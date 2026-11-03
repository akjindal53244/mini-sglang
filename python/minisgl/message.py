"""Typed CPU messages and MessagePack transport between local processes.

Only token IDs and request settings cross these sockets, as plain lists and
numbers. KV tensors and activations stay on GPU; collectives use NCCL instead.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass

import msgpack
import zmq
import zmq.asyncio
from minisgl.core import SamplingParams


def serialize_type(value):
    if is_dataclass(value):
        return {
            "__type__": type(value).__name__,
            **{field.name: serialize_type(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, dict):
        return {key: serialize_type(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(serialize_type(item) for item in value)
    if isinstance(value, (int, float, str, bool, bytes, type(None))):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def deserialize_type(classes, value):
    if isinstance(value, (list, tuple)):
        return type(value)(deserialize_type(classes, item) for item in value)
    if not isinstance(value, dict):
        return value
    name = value.get("__type__")
    data = {
        key: deserialize_type(classes, item) for key, item in value.items() if key != "__type__"
    }
    return data if name is None else classes[name](**data)


class Message:
    def encoder(self) -> dict:
        return serialize_type(self)

    @staticmethod
    def decoder(data: dict):
        return deserialize_type(globals(), data)


@dataclass
class BatchMsg(Message):
    data: list[Message]


@dataclass
class UserMsg(Message):
    uid: int
    input_ids: list[int]
    sampling_params: SamplingParams


@dataclass
class UserReply(Message):
    uid: int
    incremental_output: str
    finished: bool
    finish_reason: str | None = None


@dataclass
class DetokenizeMsg(Message):
    uid: int
    next_token: int
    finished: bool
    finish_reason: str | None = None


@dataclass
class ErrorMsg(Message):
    uid: int
    error: str


@dataclass
class TokenizeMsg(Message):
    uid: int
    text: str | list[dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(Message):
    uid: int


@dataclass
class CancelledMsg(Message):
    """Scheduler acknowledges cancellation; detokenizer can release its state."""

    uid: int


class ZmqQueue:
    """One socket with an explicit PUSH/PULL direction."""

    _context_type = zmq.Context

    def __init__(self, address: str, kind: int, *, bind: bool = False):
        self.context = self._context_type()
        self.socket = self.context.socket(kind)
        self.socket.bind(address) if bind else self.socket.connect(address)

    def put(self, message: Message) -> None:
        self.socket.send(msgpack.packb(message.encoder(), use_bin_type=True), copy=False)

    def get(self) -> Message:
        return self.decode(self.get_raw())

    def get_raw(self) -> bytes:
        return self.socket.recv()

    @staticmethod
    def decode(raw: bytes) -> Message:
        return Message.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self) -> None:
        if not self.context.closed:
            self.socket.close(linger=0)
            self.context.term()


class ZmqAsyncQueue(ZmqQueue):
    _context_type = zmq.asyncio.Context

    async def put(self, message: Message) -> None:
        await self.socket.send(msgpack.packb(message.encoder(), use_bin_type=True), copy=False)

    async def get(self) -> Message:
        return self.decode(await self.socket.recv())
