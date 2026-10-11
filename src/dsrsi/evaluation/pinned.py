"""Pin an explicit committed checkpoint generation for read-only evaluation."""

import json
from pathlib import Path

from dsrsi.training.checkpoint import FORMAT_VERSION, CheckpointError, CheckpointStore, file_sha256


def load_pinned(store: CheckpointStore, name: str | None):
    """Latest generation, or an explicit immutable one verified the same way."""
    if name is None:
        return store.load_committed()
    root = Path(store.root).resolve()
    directory = (root / name).resolve()
    if directory.parent != root:
        raise CheckpointError("pinned checkpoint must be a direct child of the store")
    if not directory.is_dir() or directory.name.endswith(".pending"):
        raise CheckpointError("pinned checkpoint must be a committed generation directory")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise CheckpointError("pinned checkpoint has no manifest")
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
    if not manifest.get("progress", {}).get("stage"):
        raise CheckpointError("pinned checkpoint progress is incomplete")
    return directory, manifest
