"""Small helpers for shapes, model downloads, logging, and memory."""

from __future__ import annotations

import copy
import functools
import json
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)


@functools.cache
def compute_capability() -> tuple[int, int]:
    """GPU architecture, e.g. (12, 0) for RTX PRO 6000; (0, 0) without a GPU.

    Comparing numbers does not guarantee that a kernel supports a GPU.
    """
    import torch

    return torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = (
                os.path.join(model_path, "chat_template.json")
                if os.path.isdir(model_path)
                else hf_hub_download(repo_id=model_path, filename="chat_template.json")
            )
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except OSError:
            pass
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    # Each caller may normalize the config. Preserve the resolved Hub revision
    # as well: a to_dict()/constructor round-trip drops the private commit hash.
    return copy.deepcopy(_load_hf_config(model_path))


def load_eos_token_ids(model_path: str, hf_config: PretrainedConfig) -> set[int]:
    """Stop tokens from both configs: generation_config.json often lists more."""
    sources = [hf_config.eos_token_id]
    try:
        sources.append(GenerationConfig.from_pretrained(model_path).eos_token_id)
    except OSError:  # The file is optional.
        pass
    return {
        i for ids in sources for i in (ids if isinstance(ids, list) else [ids]) if i is not None
    }


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    # Mistral repos add consolidated.safetensors: the same weights, other names.
    return snapshot_download(
        model_path, allow_patterns=["*.safetensors"], ignore_patterns=["consolidated*"]
    )


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1."""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def init_logger(name: str):
    """Plain logging; suppress routine duplicate messages from non-primary ranks."""
    import logging
    import sys

    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    if logger.handlers:
        return logger

    def rank_filter(record):
        from minisgl.distributed import get_world_info

        world = get_world_info()
        record.rank = f" rank={world.rank}" if world.size > 1 else ""
        return world.is_primary() or record.levelno >= logging.WARNING

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(rank_filter)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s%(rank)s] %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)
    return logger


def format_memory(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"
