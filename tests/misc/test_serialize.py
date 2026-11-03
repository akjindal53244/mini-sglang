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
    tokens: list[int]


def test_nested_dataclass_round_trip():
    payload = NestedPayload(10, [NestedPayload(20, [], [1, 2])], [3])
    decoded = deserialize_type({"NestedPayload": NestedPayload}, serialize_type(payload))
    assert decoded == payload


def test_tensors_are_rejected_instead_of_silently_serialized():
    with pytest.raises(TypeError, match="Tensor"):
        serialize_type(UserMsg(1, torch.tensor([1, 2]), SamplingParams()))


@pytest.mark.parametrize(
    "message",
    [
        UserMsg(1, [1, 2], SamplingParams()),
        BatchMsg([UserMsg(1, [3], SamplingParams())]),
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
def test_queue_preserves_nested_messages(tmp_path, asynchronous):
    import asyncio

    import zmq

    queue_type = ZmqAsyncQueue if asynchronous else ZmqQueue
    address = "ipc://" + str(tmp_path / "messages.sock")
    receiver = queue_type(address, zmq.PULL, bind=True)
    sender = queue_type(address, zmq.PUSH)
    sender.socket.setsockopt(zmq.SNDTIMEO, 2000)
    receiver.socket.setsockopt(zmq.RCVTIMEO, 2000)
    message = BatchMsg([UserMsg(7, [2, 3], SamplingParams())])

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
