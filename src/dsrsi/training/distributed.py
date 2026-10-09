"""Explicit Gloo/HCCL collectives for control, counters and conditional grads."""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import json

import torch
import torch.distributed as dist

from .control import ACTIONS, ControlRequest


class CollectiveContext:
    def __init__(self, *, device="cpu", enabled=None):
        self.device = torch.device(device)
        self.enabled = dist.is_initialized() if enabled is None else enabled
        if self.enabled and not dist.is_initialized():
            raise RuntimeError("collective context requires an initialized process group")
        self.rank = dist.get_rank() if self.enabled else 0
        self.world_size = dist.get_world_size() if self.enabled else 1

    @classmethod
    def initialize(cls, *, device, backend, init_method="env://", rank=None, world_size=None, timeout_seconds=60):
        kwargs = {"backend": backend, "init_method": init_method, "timeout": timedelta(seconds=timeout_seconds)}
        if rank is not None:
            kwargs.update(rank=rank, world_size=world_size)
        dist.init_process_group(**kwargs)
        return cls(device=device, enabled=True)

    def broadcast_json(self, value, *, source=0):
        if not self.enabled:
            return value
        data = json.dumps(value, ensure_ascii=False).encode() if self.rank == source else b""
        length = torch.tensor([len(data)], device=self.device, dtype=torch.int64)
        dist.broadcast(length, source)
        count = int(length.item())
        buffer = (torch.tensor(list(data), device=self.device, dtype=torch.uint8) if self.rank == source
                  else torch.empty(count, device=self.device, dtype=torch.uint8))
        dist.broadcast(buffer, source)
        return json.loads(bytes(buffer.cpu().tolist()).decode())

    def totals(self, *values):
        tensor = torch.tensor(values, device=self.device, dtype=torch.int64)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tuple(tensor.cpu().tolist())

    def control(self, local_request):
        if not self.enabled:
            return local_request
        priority = 0 if local_request is None else ACTIONS[local_request.action]
        maximum = torch.tensor([priority], device=self.device, dtype=torch.int64)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if int(maximum.item()) == 0:
            return None
        candidate = self.rank if priority == int(maximum.item()) else self.world_size
        winner = torch.tensor([candidate], device=self.device, dtype=torch.int64)
        dist.all_reduce(winner, op=dist.ReduceOp.MIN)
        source = int(winner.item())
        request = self.broadcast_json(asdict(local_request) if self.rank == source else None, source=source)
        return ControlRequest(**request)

    def require_all_success(self, local_error):
        if not self.enabled:
            if local_error:
                raise RuntimeError(local_error)
            return
        failed_rank = torch.tensor([self.rank if local_error else self.world_size], device=self.device, dtype=torch.int64)
        dist.all_reduce(failed_rank, op=dist.ReduceOp.MIN)
        source = int(failed_rank.item())
        if source != self.world_size:
            message = self.broadcast_json(local_error if self.rank == source else None, source=source)
            raise RuntimeError(f"rank {source} failed: {message}")

    def run_local(self, phase, operation):
        """Coordinate failures of local work before entering the next collective.

        operation must contain no collective calls. A failed microbatch is never
        committed; callers terminate and resume the last committed checkpoint.
        Device/communication failures still rely on the process-group timeout.
        """
        if not self.enabled:
            return operation()
        result, error = None, None
        try:
            result = operation()
        except Exception as exc:
            error = f"{phase}: {type(exc).__name__}: {exc}"
        self.require_all_success(error)
        return result

    def broadcast_parameters(self, parameters):
        if self.enabled:
            for parameter in parameters:
                dist.broadcast(parameter.data, 0)

    def synchronize_gradients(self, parameters):
        if not self.enabled:
            return
        parameters = list(parameters)
        def pack():
            active = torch.tensor([p.grad is not None for p in parameters], device=self.device, dtype=torch.int64)
            grouped = {}
            for index, parameter in enumerate(parameters):
                if parameter.grad is not None and parameter.grad.is_sparse:
                    raise ValueError("memory modules use dense gradients; sparse synchronization requires its own protocol")
                grouped.setdefault(parameter.dtype, []).append((index, parameter))
            packed = [(entries, torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1)
                                          for _, p in entries])) for entries in grouped.values()]
            return active, packed
        active, packed = self.run_local("pack gradients", pack)
        dist.all_reduce(active, op=dist.ReduceOp.SUM)
        for entries, flat in packed:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            def unpack():
                flat.div_(self.world_size)
                offset = 0
                for index, parameter in entries:
                    n = parameter.numel()
                    parameter.grad = None if active[index].item() == 0 else flat[offset:offset + n].reshape_as(parameter).clone()
                    offset += n
            self.run_local("unpack gradients", unpack)

    def close(self):
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()
