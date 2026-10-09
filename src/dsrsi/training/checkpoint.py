"""Transactional checkpoints at completed microbatch/unroll boundaries.

Autograd graphs are deliberately not serialized. The caller must finish backward
and provide detached numerical life state at a recoverable boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dsrsi.utils.atomic import atomic_json, fsync_directory as _fsync_dir

FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class Progress:
    stage: str
    optimizer_step: int = 0
    micro_step: int = 0
    accumulation_index: int = 0
    student_tokens: int = 0
    teacher_tokens: int = 0

    def __post_init__(self):
        if not self.stage:
            raise ValueError("stage must be specified")
        for name, value in asdict(self).items():
            if name != "stage" and (not isinstance(value, int) or value < 0):
                raise ValueError(f"invalid progress counter: {name}")


def cpu_tree(value: Any) -> Any:
    """Copy plain state without retaining tensors' graphs or device storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise TypeError(f"state must contain tensors/plain values, got {type(value).__name__}")


def _class_name(value: Any) -> str | None:
    return None if value is None else f"{type(value).__module__}.{type(value).__qualname__}"


def frozen_auxiliary_parameters(model):
    provider = getattr(model, "frozen_auxiliary_parameters", None)
    return {} if provider is None else provider()


def _optimizer_names(model, optimizer) -> list[list[str]]:
    names = {id(param): name for name, param in model.named_parameters()}
    try:
        return [[names[id(param)] for param in group["params"]]
                for group in optimizer.param_groups]
    except KeyError as exc:
        raise CheckpointError("optimizer includes a parameter outside the model") from exc


def capture_rng(device: str | torch.device | None = None) -> dict:
    np_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "kind": np_state[0], "keys": np_state[1].tolist(),
            "position": np_state[2], "has_gauss": np_state[3], "cached_gauss": np_state[4],
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "accelerator": None,
    }
    if device is not None:
        dev = torch.device(device)
        if dev.type in ("cuda", "npu"):
            backend = getattr(torch, dev.type)
            state["accelerator"] = {
                "kind": dev.type, "state": backend.get_rng_state(dev).cpu().clone(),
            }
        elif dev.type != "cpu":
            raise ValueError(f"unsupported RNG device: {dev}")
    return state


def restore_rng(state: dict, device: str | torch.device | None = None) -> None:
    accelerator = state["accelerator"]
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if accelerator is not None and accelerator["kind"] != dev.type:
        raise CheckpointError("accelerator RNG device type mismatch")
    if accelerator is None and dev.type != "cpu":
        raise CheckpointError("checkpoint has no accelerator RNG state")
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state["kind"], np.asarray(np_state["keys"], dtype=np.uint32),
                         np_state["position"], np_state["has_gauss"], np_state["cached_gauss"]))
    torch.set_rng_state(state["torch_cpu"])
    if accelerator is not None:
        getattr(torch, dev.type).set_rng_state(accelerator["state"], dev)


def capture_training_state(
    model, optimizer, *, progress: Progress, data_cursor: dict, life_state: dict,
    rank: int = 0, world_size: int = 1, scheduler=None, scaler=None,
    device: str | torch.device | None = None,
) -> dict:
    if not 0 <= rank < world_size:
        raise ValueError("invalid rank/world_size")
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    return {
        "format_version": FORMAT_VERSION, "rank": rank, "world_size": world_size,
        "progress": asdict(progress),
        "weights": cpu_tree(params),
        "frozen_auxiliary": cpu_tree(frozen_auxiliary_parameters(model)),
        "buffers": cpu_tree(dict(model.named_buffers())),
        "gradients": cpu_tree({name: param.grad for name, param in params.items()}),
        "optimizer": {
            "class": _class_name(optimizer), "parameter_names": _optimizer_names(model, optimizer),
            "state": cpu_tree(optimizer.state_dict()),
        },
        "scheduler": {"class": _class_name(scheduler),
                      "state": None if scheduler is None else cpu_tree(scheduler.state_dict())},
        "scaler": {"class": _class_name(scaler),
                   "state": None if scaler is None else cpu_tree(scaler.state_dict())},
        "data_cursor": cpu_tree(data_cursor), "life_state": cpu_tree(life_state),
        "rng": capture_rng(device),
    }


def _check_tensors(current: dict, saved: dict, name: str) -> None:
    if set(current) != set(saved):
        raise CheckpointError(f"{name} names differ: {sorted(set(current) ^ set(saved))}")
    for key, value in current.items():
        other = saved[key]
        if not isinstance(other, torch.Tensor) or other.shape != value.shape or other.dtype != value.dtype:
            raise CheckpointError(f"{name} shape/dtype mismatch: {key}")


def restore_training_state(
    payload: dict, model, optimizer, *, scheduler=None, scaler=None,
    device: str | torch.device | None = None,
) -> tuple[Progress, dict, dict]:
    """Validate topology and state layout before altering model weights."""
    if payload["format_version"] != FORMAT_VERSION:
        raise CheckpointError("unsupported checkpoint format")
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    buffers = dict(model.named_buffers())
    _check_tensors(params, payload["weights"], "trainable parameters")
    _check_tensors(buffers, payload["buffers"], "buffers")
    frozen = frozen_auxiliary_parameters(model)
    if "frozen_auxiliary" in payload:
        _check_tensors(frozen, payload["frozen_auxiliary"], "frozen auxiliary parameters")
    elif payload["progress"]["stage"] not in {"A", "B"}:
        raise CheckpointError("new training stages require complete frozen auxiliary state")
    if set(params) != set(payload["gradients"]):
        raise CheckpointError("gradient names differ")
    for name, grad in payload["gradients"].items():
        if grad is not None and (grad.shape != params[name].shape or grad.dtype != params[name].dtype):
            raise CheckpointError(f"gradient shape/dtype mismatch: {name}")
    saved_opt = payload["optimizer"]
    if saved_opt["class"] != _class_name(optimizer):
        raise CheckpointError("optimizer class differs; an explicit stage transition is required")
    if saved_opt["parameter_names"] != _optimizer_names(model, optimizer):
        raise CheckpointError("optimizer parameter group/order mismatch")
    for name, obj in (("scheduler", scheduler), ("scaler", scaler)):
        if payload[name]["class"] != _class_name(obj):
            raise CheckpointError(f"{name} class mismatch")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    saved_acc = payload["rng"]["accelerator"]
    if (saved_acc is None and dev.type != "cpu") or (saved_acc is not None and saved_acc["kind"] != dev.type):
        raise CheckpointError("accelerator RNG type mismatch")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(payload["weights"][name].to(param.device))
            grad = payload["gradients"][name]
            param.grad = None if grad is None else grad.to(param.device).clone()
        for name, buf in buffers.items():
            buf.copy_(payload["buffers"][name].to(buf.device))
        for name, value in payload.get("frozen_auxiliary", {}).items():
            frozen[name].copy_(value.to(frozen[name].device))
    optimizer.load_state_dict(saved_opt["state"])
    for name, obj in (("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None:
            obj.load_state_dict(payload[name]["state"])
    restore_rng(payload["rng"], device)
    return Progress(**payload["progress"]), payload["data_cursor"], payload["life_state"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointStore:
    """Rank files become visible as a checkpoint only after manifest commit."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _inside(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        if resolved.parent != self.root:
            raise CheckpointError("checkpoint must be a direct child of its store")
        return resolved

    def begin(self, progress: Progress) -> Path:
        path = self.root / f"micro_{progress.micro_step:012d}_{uuid.uuid4().hex[:10]}.pending"
        path.mkdir()
        return path

    def save_rank(self, directory: str | Path, payload: dict) -> None:
        directory = self._inside(directory)
        if not directory.name.endswith(".pending") or (directory / "manifest.json").exists():
            raise CheckpointError("rank writes require an uncommitted generation")
        rank = payload["rank"]
        if not 0 <= rank < payload["world_size"]:
            raise CheckpointError("invalid rank payload")
        destination = directory / f"rank_{rank:05d}.pt"
        if destination.exists():
            raise CheckpointError("rank already wrote this generation")
        fd, temp_name = tempfile.mkstemp(prefix=f".rank_{rank}.", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, destination)
            _fsync_dir(directory)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        atomic_json(directory / f"rank_{rank:05d}.ready.json", {
            "rank": rank, "world_size": payload["world_size"],
            "progress": payload["progress"], "format_version": payload["format_version"],
            "file": destination.name, "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
        })

    def commit(
        self, directory: str | Path, *, world_size: int, identity: dict,
        progress: Progress, status: str = "checkpoint", required_student_tokens: int | None = None,
    ) -> Path:
        directory = self._inside(directory)
        if not directory.name.endswith(".pending") or world_size < 1:
            raise CheckpointError("invalid checkpoint generation")
        if status not in {"checkpoint", "paused", "stopped", "completed"}:
            raise ValueError("invalid checkpoint status")
        if status == "completed" and (required_student_tokens is None or required_student_tokens <= 0
                                      or progress.student_tokens < required_student_tokens):
            raise CheckpointError("completed status requires the configured token budget to be met")
        files = []
        for rank in range(world_size):
            ready_path = directory / f"rank_{rank:05d}.ready.json"
            if not ready_path.is_file():
                raise CheckpointError(f"rank {rank} has not completed its save")
            entry = json.loads(ready_path.read_text())
            if (entry["rank"] != rank or entry["world_size"] != world_size
                    or entry["progress"] != asdict(progress) or entry["format_version"] != FORMAT_VERSION):
                raise CheckpointError(f"rank {rank} state counters/topology differ")
            path = directory / f"rank_{rank:05d}.pt"
            if entry["file"] != path.name or path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
                raise CheckpointError(f"rank {rank} payload verification failed")
            files.append(entry)
        manifest = {
            "format_version": FORMAT_VERSION, "created_unix": time.time(),
            "status": status, "world_size": world_size, "identity": identity,
            "progress": asdict(progress), "required_student_tokens": required_student_tokens,
            "files": files,
        }
        atomic_json(directory / "manifest.json", manifest)
        final = directory.with_suffix("")
        os.replace(directory, final)
        _fsync_dir(self.root)
        atomic_json(self.root / "latest.json", {
            "directory": final.name, "manifest_sha256": file_sha256(final / "manifest.json"),
        })
        return final

    def load_committed(self) -> tuple[Path, dict]:
        """Pin and verify a generation once, also for read-only evaluation."""
        descriptor = json.loads((self.root / "latest.json").read_text())
        directory = self._inside(self.root / descriptor["directory"])
        if directory.name.endswith(".pending"):
            raise CheckpointError("latest must reference a committed generation")
        manifest_path = directory / "manifest.json"
        if file_sha256(manifest_path) != descriptor["manifest_sha256"]:
            raise CheckpointError("manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text())
        world_size = manifest["world_size"]
        if manifest["format_version"] != FORMAT_VERSION or world_size < 1:
            raise CheckpointError("format/world_size mismatch")
        if len(manifest["files"]) != world_size:
            raise CheckpointError("invalid checkpoint rank inventory")
        for expected_rank, entry in enumerate(manifest["files"]):
            path = directory / f"rank_{expected_rank:05d}.pt"
            if entry["rank"] != expected_rank or entry["file"] != path.name:
                raise CheckpointError("rank manifest mismatch")
            if not path.is_file() or path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
                raise CheckpointError(f"rank {expected_rank} payload checksum mismatch")
        return directory, manifest

    def load_rank(self, directory: Path, manifest: dict, *, rank: int) -> dict:
        directory = self._inside(directory)
        world_size = manifest["world_size"]
        if not 0 <= rank < world_size:
            raise CheckpointError("invalid checkpoint rank inventory")
        path = directory / f"rank_{rank:05d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["format_version"] != FORMAT_VERSION or payload["rank"] != rank or payload["world_size"] != world_size
                or payload["progress"] != manifest["progress"]):
            raise CheckpointError("rank payload disagrees with committed manifest")
        return payload

    def load_latest(self, *, identity: dict, rank: int = 0, world_size: int = 1) -> tuple[dict, dict]:
        directory, manifest = self.load_committed()
        if manifest["world_size"] != world_size:
            raise CheckpointError("format/world_size mismatch")
        if manifest["identity"] != identity:
            raise CheckpointError("base/config/data identity mismatch")
        payload = self.load_rank(directory, manifest, rank=rank)
        return payload, manifest
