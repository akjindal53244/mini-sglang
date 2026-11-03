from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.message import (
    AbortMsg,
    BatchMsg,
    CancelledMsg,
    DetokenizeMsg,
    ErrorMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
    ZmqAsyncQueue,
    ZmqQueue,
    deserialize_type,
    serialize_type,
)


@dataclass
class NestedPayload:
    value: int
    children: list[NestedPayload]
    tokens: torch.Tensor


def test_nested_tensor_round_trip():
    tokens = torch.tensor([1, 2, 3], dtype=torch.int32)
    payload = NestedPayload(10, [NestedPayload(20, [], tokens)], tokens)
    decoded = deserialize_type({"NestedPayload": NestedPayload}, serialize_type(payload))
    assert decoded.value == 10 and decoded.children[0].value == 20
    assert torch.equal(decoded.tokens, tokens)
    assert torch.equal(decoded.children[0].tokens, tokens)
    assert decoded.tokens.dtype == torch.int32


@pytest.mark.parametrize(
    "message",
    [
        UserMsg(1, torch.tensor([1, 2], dtype=torch.int32), SamplingParams()),
        BatchMsg([UserMsg(1, torch.tensor([3], dtype=torch.int32), SamplingParams())]),
        TokenizeMsg(1, [{"role": "user", "content": "hello"}], SamplingParams()),
        UserReply(1, "hello", True),
        DetokenizeMsg(1, 42, False),
        BatchMsg([UserReply(1, "hello", True)]),
        AbortMsg(1),
        CancelledMsg(1),
        ErrorMsg(1, "input too long"),
    ],
)
def test_wire_message_round_trip(message):
    encoded = message.encoder()
    decoded = message.decoder(encoded)
    assert type(decoded) is type(message)
    assert decoded.encoder() == encoded


@pytest.mark.parametrize("asynchronous", [False, True])
def test_queue_preserves_nested_message_and_tensor_values(tmp_path, asynchronous):
    import asyncio

    import zmq

    queue_type = ZmqAsyncQueue if asynchronous else ZmqQueue
    address = "ipc://" + str(tmp_path / "messages.sock")
    receiver = queue_type(address, zmq.PULL, bind=True)
    sender = queue_type(address, zmq.PUSH)
    sender.socket.setsockopt(zmq.SNDTIMEO, 2000)
    receiver.socket.setsockopt(zmq.RCVTIMEO, 2000)
    message = BatchMsg([UserMsg(7, torch.tensor([2, 3], dtype=torch.int32), SamplingParams())])

    async def exchange():
        await sender.put(message)
        return await receiver.get()

    try:
        if asynchronous:
            decoded = asyncio.run(exchange())
        else:
            sender.put(message)
            decoded = receiver.get()
        assert decoded.encoder() == message.encoder()
    finally:
        sender.stop()
        receiver.stop()
