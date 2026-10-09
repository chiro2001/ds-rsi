"""Control requests are observed at recoverable boundaries, never in backward."""

from __future__ import annotations

import argparse
import json
import signal
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .checkpoint import atomic_json

ACTIONS = {"checkpoint": 1, "pause": 2, "stop": 3}


@dataclass(frozen=True)
class ControlRequest:
    request_id: str
    action: str

    def __post_init__(self):
        if not self.request_id or self.action not in ACTIONS:
            raise ValueError("invalid control request")


class ControlChannel:
    def __init__(self, run_dir: str | Path):
        self.root = Path(run_dir) / "control"
        self._signal_action: str | None = None
        self._signal_request: ControlRequest | None = None

    def request(self, action: str) -> ControlRequest:
        req = ControlRequest(uuid.uuid4().hex, action)
        atomic_json(self.root / "request.json", asdict(req))
        return req

    def install_signal_handlers(self) -> None:
        def record(action):
            def handler(_signum, _frame):
                if self._signal_action is None or ACTIONS[action] > ACTIONS[self._signal_action]:
                    self._signal_action = action
            return handler

        for name, action in (("SIGUSR1", "checkpoint"), ("SIGUSR2", "pause"),
                             ("SIGTERM", "stop"), ("SIGINT", "pause")):
            signum = getattr(signal, name, None)
            if signum is not None:
                signal.signal(signum, record(action))

    def poll(self) -> ControlRequest | None:
        candidates = []
        if self._signal_action is not None:
            if self._signal_request is None or self._signal_request.action != self._signal_action:
                self._signal_request = ControlRequest(uuid.uuid4().hex, self._signal_action)
            candidates.append(self._signal_request)
        request_path = self.root / "request.json"
        if request_path.exists():
            req = ControlRequest(**json.loads(request_path.read_text()))
            ack_path = self.root / "acknowledged.json"
            ack = json.loads(ack_path.read_text()) if ack_path.exists() else {}
            if ack.get("request_id") != req.request_id:
                candidates.append(req)
        return max(candidates, key=lambda req: ACTIONS[req.action]) if candidates else None

    def acknowledge(self, request: ControlRequest, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        if checkpoint.name.endswith(".pending") or not (checkpoint / "manifest.json").is_file():
            raise ValueError("only a committed checkpoint can acknowledge a request")
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        expected_status = {"checkpoint": "checkpoint", "pause": "paused", "stop": "stopped"}
        if manifest.get("status") != expected_status[request.action]:
            raise ValueError("checkpoint status does not match the requested action")
        atomic_json(self.root / "acknowledged.json", {
            **asdict(request), "checkpoint": str(checkpoint.resolve()),
        })
        if self._signal_request is not None and self._signal_request.request_id == request.request_id:
            self._signal_action = None
            self._signal_request = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Request a checkpoint or safe pause")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--action", required=True, choices=tuple(ACTIONS))
    args = parser.parse_args()
    print(json.dumps(asdict(ControlChannel(args.run_dir).request(args.action)), ensure_ascii=False))


if __name__ == "__main__":
    main()
