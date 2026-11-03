"""Offline inference: encode prompts, step the scheduler, collect output IDs."""

from __future__ import annotations

import torch
from minisgl.config import Config
from minisgl.core import SamplingParams
from minisgl.message import ErrorMsg, UserMsg
from minisgl.scheduler import Scheduler
from minisgl.utils import load_tokenizer


class LLM:
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        self.tokenizer = load_tokenizer(model_path)
        self.scheduler = Scheduler(Config(model_path, dtype=dtype, **kwargs))

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: list[SamplingParams] | SamplingParams,
    ) -> list[dict]:
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("Provide exactly one SamplingParams per prompt")
        if self.scheduler is None:
            raise RuntimeError("LLM is closed")
        outputs = [[] for _ in prompts]
        next_prompt = 0
        while next_prompt < len(prompts) or not self.scheduler.idle:
            # Bound CPU tokenization work per step so large prompt lists do not
            # postpone the first GPU batch. Admission applies the actual KV budget.
            messages, input_tokens = [], 0
            while (
                next_prompt < len(prompts)
                and input_tokens < self.scheduler.config.max_extend_tokens
            ):
                prompt = prompts[next_prompt]
                if isinstance(prompt, str):
                    prompt = self.tokenizer.encode(prompt)
                ids = torch.tensor(prompt, dtype=torch.int32, device="cpu")
                input_tokens += len(ids)
                messages.append(UserMsg(next_prompt, ids, sampling_params[next_prompt]))
                next_prompt += 1
            for reply in self.scheduler.step(messages):
                if isinstance(reply, ErrorMsg):
                    # Finish submitted work before exposing an invalid batch to
                    # the caller, so a later generate() can reuse this instance.
                    while not self.scheduler.idle:
                        self.scheduler.step([])
                    raise ValueError(reply.error)
                if reply.finish_reason != "stop":
                    outputs[reply.uid].append(reply.next_token)
        return [{"text": self.tokenizer.decode(ids), "token_ids": ids} for ids in outputs]

    def shutdown(self) -> None:
        if self.scheduler is not None:
            self.scheduler.shutdown()
            self.scheduler = None  # Release model, KV pool, and graph buffers together.
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
