"""HTTP/shell frontends and the local worker processes they own.

Both frontends submit TokenizeMsg and consume UserReply. One tokenizer worker
handles text; each GPU has one Scheduler. No serving code enters the model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Literal

import torch
import uvicorn
import zmq
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from minisgl.config import Config, parse_args
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    AbortMsg,
    BatchMsg,
    ErrorMsg,
    Message,
    TokenizeMsg,
    UserReply,
    ZmqAsyncQueue,
    ZmqQueue,
)
from minisgl.utils import init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = init_logger(__name__)


class WorkerUnavailable(RuntimeError):
    """The serving workers cannot complete this request."""


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[ChatMessage] | None = None
    prompt: str | None = Field(default=None, min_length=1)  # Raw-prompt extension.
    # KV for the whole budget is reserved at admission, so keep the default finite.
    max_tokens: int = Field(default=SamplingParams.max_tokens, gt=0)
    temperature: float = Field(default=1.0, ge=0)
    top_k: int = Field(default=-1, ge=-1)
    top_p: float = Field(default=1.0, gt=0, le=1)
    ignore_eos: bool = False
    stream: bool = False
    # Accept neutral SDK defaults; reject settings we cannot implement faithfully.
    n: Literal[1] = 1
    stop: Literal[None] = None
    presence_penalty: Literal[0.0] = 0.0
    frequency_penalty: Literal[0.0] = 0.0

    @model_validator(mode="after")
    def require_input(self):
        if not self.messages and not self.prompt:
            raise ValueError("Provide messages or a nonempty prompt")
        return self


@dataclass
class FrontendManager:
    config: Config
    send_tokenizer: ZmqAsyncQueue
    recv_tokenizer: ZmqAsyncQueue
    pending: dict[int, asyncio.Queue[UserReply | ErrorMsg]] = field(default_factory=dict)
    next_uid: int = 0
    workers: list = field(default_factory=list)
    background: set[asyncio.Task] = field(default_factory=set)
    worker_failed: asyncio.Event = field(default_factory=asyncio.Event)
    worker_crashed: bool = False

    def new_request(self) -> int:
        uid = self.next_uid
        self.next_uid += 1
        self.pending[uid] = asyncio.Queue()
        return uid

    async def submit(self, text, params: SamplingParams) -> int:
        if self.worker_failed.is_set():
            raise WorkerUnavailable("An inference worker exited")
        uid = self.new_request()
        sending = asyncio.create_task(self.send_tokenizer.put(TokenizeMsg(uid, text, params)))
        failed = asyncio.create_task(self.worker_failed.wait())
        try:
            # A dead tokenizer can leave PUSH waiting for a peer. Failure must
            # interrupt that send as well as requests already waiting for tokens.
            await asyncio.wait((sending, failed), return_when=asyncio.FIRST_COMPLETED)
            if self.worker_failed.is_set():
                raise WorkerUnavailable("An inference worker exited")
            await sending
            return uid
        except BaseException:
            self.pending.pop(uid, None)
            self._send_abort(uid)
            raise
        finally:
            sending.cancel()
            failed.cancel()
            await asyncio.gather(sending, failed, return_exceptions=True)

    def _send_abort(self, uid: int):
        if self.worker_failed.is_set():
            return
        # Retain the task: asyncio keeps only weak references to background work.
        task = asyncio.create_task(self.send_tokenizer.put(AbortMsg(uid)))
        self.background.add(task)
        task.add_done_callback(self.background.discard)

    async def listen(self):
        while True:
            message = await self.recv_tokenizer.get()
            for reply in message.data if isinstance(message, BatchMsg) else [message]:
                if reply.uid in self.pending:
                    self.pending[reply.uid].put_nowait(reply)

    async def responses(self, uid: int):
        replies, finished = self.pending[uid], False
        try:
            while not finished:
                reply = await replies.get()
                finished = isinstance(reply, ErrorMsg) or reply.finished
                if finished:
                    # Release before yielding: consumers often break immediately
                    # after the terminal reply without advancing this generator.
                    self.pending.pop(uid, None)
                if isinstance(reply, ErrorMsg):
                    if self.worker_failed.is_set():
                        raise WorkerUnavailable(reply.error)
                    raise ValueError(reply.error)
                yield reply
        finally:
            self.pending.pop(uid, None)
            if not finished:
                # StreamingResponse cancels its generator on disconnect. Send
                # the abort in a separate task so that cancellation cannot eat it.
                self._send_abort(uid)

    async def watch_workers(self):
        """A dead worker never replies: fail waiting clients, then stop serving."""
        while all(worker.is_alive() for worker in self.workers):
            await asyncio.sleep(1)
        # Ctrl-C also reaches the workers, which exit 0 while the server is stopping.
        self.worker_crashed = any(worker.exitcode not in (None, 0) for worker in self.workers)
        if self.worker_crashed:
            logger.error("An inference worker exited; shutting down")
        # Latch failure before waking clients. An already-accepted HTTP request
        # may finish reading its body after this loop has visited pending UIDs.
        self.worker_failed.set()
        for uid, replies in self.pending.items():
            replies.put_nowait(ErrorMsg(uid, "An inference worker exited"))
        # Not SIGINT: a second one makes Uvicorn force-quit and skip lifespan cleanup.
        signal.raise_signal(signal.SIGTERM)

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def _listen(frontend: FrontendManager):
    tasks = [
        asyncio.create_task(frontend.listen()),
        asyncio.create_task(frontend.watch_workers()),
    ]
    try:
        yield
    finally:
        tasks.extend(frontend.background)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        frontend.shutdown()


def _sse(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


async def _chat_stream(frontend: FrontendManager, uid: int, model: str):
    first = True
    try:
        async for reply in frontend.responses(uid):
            delta = {"content": reply.incremental_output}
            if first:
                delta["role"], first = "assistant", False
            yield _sse(
                {
                    "id": f"chatcmpl-{uid}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": reply.finish_reason}],
                }
            )
    except (ValueError, WorkerUnavailable) as error:
        kind = "server_error" if isinstance(error, WorkerUnavailable) else "invalid_request_error"
        yield _sse({"error": {"message": str(error), "type": kind}})
    yield b"data: [DONE]\n\n"


def create_app(frontend: FrontendManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_):
        async with _listen(frontend):
            yield

    app = FastAPI(title="Mini-SGLang", lifespan=lifespan)

    @app.exception_handler(WorkerUnavailable)
    async def worker_unavailable(_request, error):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
    async def root():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        model = frontend.config.model_path
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "mini-sglang",
                    "root": model,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def completions(request: OpenAICompletionRequest):
        if request.model != frontend.config.model_path:
            raise HTTPException(404, "This server hosts " + frontend.config.model_path)
        prompt = (
            [msg.model_dump() for msg in request.messages] if request.messages else request.prompt
        )
        params = SamplingParams(
            request.temperature,
            request.top_k,
            request.top_p,
            request.ignore_eos,
            request.max_tokens,
        )
        uid = await frontend.submit(prompt, params)
        if request.stream:
            return StreamingResponse(
                _chat_stream(frontend, uid, request.model), media_type="text/event-stream"
            )
        text, reason = [], None
        try:
            async for reply in frontend.responses(uid):
                text.append(reply.incremental_output)
                reason = reply.finish_reason
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(text)},
                    "finish_reason": reason,
                }
            ],
        }

    return app


async def shell(frontend: FrontendManager):
    session = PromptSession("$ ", completer=WordCompleter(["/exit", "/reset"]))
    params = SamplingParams(
        temperature=float(os.getenv("MINISGL_SHELL_TEMPERATURE", "0.6")),
        top_k=int(os.getenv("MINISGL_SHELL_TOP_K", "-1")),
        top_p=float(os.getenv("MINISGL_SHELL_TOP_P", "1.0")),
        max_tokens=int(os.getenv("MINISGL_SHELL_MAX_TOKENS", "2048")),
    )
    history = []
    async with _listen(frontend):
        try:
            while True:
                text = (await session.prompt_async()).strip()
                if text == "/exit":
                    break
                if text == "/reset":
                    history.clear()
                    continue
                if not text:
                    continue
                if text.startswith("/"):
                    print("Commands: /reset, /exit")
                    continue
                history.append({"role": "user", "content": text})
                output = []
                try:
                    uid = await frontend.submit(history, params)
                    async for reply in frontend.responses(uid):
                        print(reply.incremental_output, end="", flush=True)
                        output.append(reply.incremental_output)
                except (ValueError, WorkerUnavailable) as error:
                    history.pop()
                    print(f"Error: {error}")
                    continue
                print()
                history.append({"role": "assistant", "content": "".join(output)})
        except EOFError:
            pass


def _receive_requests(receiver, group, primary: bool, idle: bool) -> list[Message]:
    raw = []
    if primary:
        if idle:
            # Sleep until work arrives. With several ranks, wake every second anyway:
            # the others wait in the Gloo broadcast below, which fails after a timeout.
            receiver.socket.poll(timeout=None if group is None else 1000)
        while not receiver.empty():
            raw.append(receiver.get_raw())
    if group is not None:
        # Keep all TP or PP ranks in lockstep. Empty decode steps broadcast only a
        # count; variable-size request packets use Gloo's object broadcast.
        count = torch.tensor(len(raw))
        torch.distributed.broadcast(count, src=0, group=group)
        if not int(count):
            return []
        packets = [raw]
        torch.distributed.broadcast_object_list(packets, src=0, group=group)
        raw = packets[0]
    return [ZmqQueue.decode(packet) for packet in raw]


@torch.inference_mode()
def _run_scheduler(config: Config, ready):
    from minisgl.scheduler import Scheduler

    scheduler = Scheduler(config)
    primary = config.world.is_primary()
    receiver = ZmqQueue(config.ipc("requests"), zmq.PULL, bind=True) if primary else None
    sender = ZmqQueue(config.ipc("tokenizer"), zmq.PUSH) if primary else None
    group = scheduler.engine.cpu_group
    if group is not None:
        group.barrier().wait()
    if primary:
        ready.put("Scheduler ready")
    if config.shell:
        logging.disable(logging.INFO)
    try:
        while True:
            if scheduler.idle:
                scheduler.cache_manager.check_integrity()
            messages = _receive_requests(receiver, group, primary, scheduler.idle)
            replies = scheduler.step(messages)
            if primary and replies:
                sender.put(BatchMsg(replies))
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl-C reaches us directly, then again from the launcher. Finish cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        scheduler.shutdown()
        if primary:
            receiver.stop()
            sender.stop()


def launch_server():
    from minisgl.tokenizer import tokenize_worker

    config = parse_args(sys.argv[1:])
    if config.shell and config.use_dummy_weight:
        raise ValueError("Shell mode requires real model weights")
    frontend = FrontendManager(
        config,
        ZmqAsyncQueue(config.ipc("tokenizer"), zmq.PUSH),
        ZmqAsyncQueue(config.ipc("replies"), zmq.PULL, bind=True),
    )
    context = mp.get_context("spawn")
    ready = context.Queue()
    field_name = "pp_info" if config.pp_info.size > 1 else "tp_info"
    workers = [
        context.Process(
            target=_run_scheduler,
            args=(
                replace(config, **{field_name: DistributedInfo(rank, config.world.size)}),
                ready,
            ),
            name=f"scheduler-{rank}",
        )
        for rank in range(config.world.size)
    ]
    workers.append(context.Process(target=tokenize_worker, args=(config, ready), name="tokenizer"))
    frontend.workers = workers
    started = []
    # Uvicorn replays the signal after serving stops. Turn SIGTERM into a
    # Python exception so the launcher's finally block still joins its workers.
    previous_sigterm = signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        for worker in workers:
            worker.start()
            started.append(worker)
        # Only the primary GPU rank and the tokenizer acknowledge readiness.
        acknowledgments = 0
        while acknowledgments < 2:
            try:
                logger.info(ready.get(timeout=1))
                acknowledgments += 1
            except queue.Empty:
                failed = [worker.name for worker in started if worker.exitcode is not None]
                if failed:
                    raise RuntimeError("Worker exited during startup: " + ", ".join(failed))
        if config.shell:
            asyncio.run(shell(frontend))
        else:
            # An unfinished HTTP body must not hold worker-failure shutdown open.
            uvicorn.run(
                create_app(frontend),
                host=config.server_host,
                port=config.server_port,
                timeout_graceful_shutdown=5,
            )
    except KeyboardInterrupt:
        pass
    finally:
        # SIGINT enters the workers' KeyboardInterrupt/finally blocks, releasing
        # sockets and multiprocessing locks. SIGTERM skips Python finalizers.
        for worker in started:
            if worker.is_alive():
                try:
                    os.kill(worker.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for worker in started:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join()
        ready.close()
        frontend.shutdown()
        signal.signal(signal.SIGTERM, previous_sigterm)
    if frontend.worker_crashed:
        raise SystemExit(1)


if __name__ == "__main__":
    launch_server()
