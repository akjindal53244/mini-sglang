"""Chat templates and incremental decoding, isolated from GPU scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import zmq
from minisgl.message import (
    AbortMsg,
    BatchMsg,
    CancelledMsg,
    DetokenizeMsg,
    ErrorMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
    ZmqQueue,
)
from minisgl.utils import load_tokenizer
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from minisgl.config import Config


def tokenize(tokenizer: PreTrainedTokenizerBase, message: TokenizeMsg) -> list[int]:
    if isinstance(message.text, list):
        # Chat templates already insert special tokens. Tokenizing the rendered
        # string again with add_special_tokens=True can duplicate BOS tokens.
        ids = tokenizer.apply_chat_template(message.text, tokenize=True, add_generation_prompt=True)
    else:
        ids = tokenizer.encode(message.text)
    return list(ids)


@dataclass
class DecodeState:
    token_ids: list[int]
    read_offset: int = 0  # Tokens whose text has been delivered.
    context_start: int = 0  # Start of the overlap used to decode the next token.


class DetokenizeManager:
    """Turn a token stream into text deltas.

    Decoding tokens one at a time is wrong: a token can hold part of a UTF-8
    character, and leading spaces depend on the previous token. So decode a small
    overlapping window twice and emit only the text the new tokens added.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.decode_map: dict[int, DecodeState] = {}  # uid -> DecodeState
        self.tokenizer = tokenizer

    def detokenize(self, msgs: list[DetokenizeMsg]) -> list[str]:
        read_ids: list[list[int]] = []
        surr_ids: list[list[int]] = []
        for msg in msgs:
            s = self.decode_map.setdefault(msg.uid, DecodeState([]))
            if msg.finish_reason != "stop":  # A stopping EOS is not output text.
                s.token_ids.append(msg.next_token)
            read_ids.append(s.token_ids[s.context_start :])
            surr_ids.append(s.token_ids[s.context_start : s.read_offset])

        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        outputs: list[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str) :]
            # U+FFFD means the last token ended inside a character, so wait for the
            # next token. A final token has no successor: flush what we have.
            if new_text and (msg.finished or not new_text.endswith("�")):
                s.context_start, s.read_offset = s.read_offset, len(s.token_ids)
                outputs.append(new_text)
            else:
                outputs.append("")
            if msg.finished:
                del self.decode_map[msg.uid]
        return outputs


def tokenize_worker(config: Config, ready) -> None:
    tokenizer = load_tokenizer(config.model_path)
    decoder = DetokenizeManager(tokenizer)
    receive = ZmqQueue(config.ipc("tokenizer"), zmq.PULL, bind=True)
    backend = ZmqQueue(config.ipc("requests"), zmq.PUSH)
    frontend = ZmqQueue(config.ipc("replies"), zmq.PUSH)
    ready.put("Tokenizer ready")
    try:
        while True:
            message = receive.get()
            messages = message.data if isinstance(message, BatchMsg) else [message]
            to_backend, to_frontend, tokens = [], [], []
            for message in messages:
                if isinstance(message, TokenizeMsg):
                    try:
                        ids = tokenize(tokenizer, message)
                    except Exception as error:
                        # A malformed chat must fail its request, not kill the
                        # shared text worker and strand every waiting client.
                        to_frontend.append(
                            ErrorMsg(message.uid, f"Could not tokenize input: {error}")
                        )
                    else:
                        to_backend.append(UserMsg(message.uid, ids, message.sampling_params))
                elif isinstance(message, AbortMsg):
                    to_backend.append(message)
                elif isinstance(message, CancelledMsg):
                    # Release after the scheduler acknowledges cancellation;
                    # earlier tokens may still have been queued for decoding.
                    decoder.decode_map.pop(message.uid, None)
                elif isinstance(message, ErrorMsg):
                    to_frontend.append(message)
                elif isinstance(message, DetokenizeMsg):
                    tokens.append(message)
                else:
                    raise TypeError(f"Unknown tokenizer message: {type(message).__name__}")
            if tokens:
                texts = decoder.detokenize(tokens)
                to_frontend.extend(
                    UserReply(msg.uid, text, msg.finished, msg.finish_reason)
                    for msg, text in zip(tokens, texts, strict=True)
                )
            if to_backend:
                backend.put(BatchMsg(to_backend))
            if to_frontend:
                frontend.put(BatchMsg(to_frontend))
    except KeyboardInterrupt:
        pass
    finally:
        receive.stop()
        backend.stop()
        frontend.stop()
