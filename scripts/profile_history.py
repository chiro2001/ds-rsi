#!/usr/bin/env python3
"""Measure Python working allocation when reading a growing durable history."""

import argparse
import gc
import json
from pathlib import Path
import time
import tracemalloc

from dsrsi.harness.memory_tools import HarnessStore


def measure(operation):
    gc.collect()
    tracemalloc.start()
    started = time.monotonic()
    value = operation()
    elapsed = time.monotonic() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, {"seconds": elapsed, "python_peak_bytes": peak}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[1000, 12000])
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for count in args.counts:
        path = root / f"history_{count}.sqlite"
        if count < 1 or (path.exists() and not args.reuse) or (args.reuse and not path.exists()):
            raise ValueError("invalid count or history reuse condition")
        if not args.reuse:
            store = HarnessStore(path, "life", database_id=f"history-profile-{count}")
            for index in range(count):
                text = f"记录{index}校验码：" + "独立的验收过程已记录，继续检查原始资料。" * 100
                store.append_history("user", text, clock=index)
            store.close()
        store, restore = measure(lambda: HarnessStore(path, "life", database_id=f"history-profile-{count}"))
        first = store.connection.execute("SELECT event_id FROM history ORDER BY revision LIMIT 1").fetchone()[0]
        opened, open_stats = measure(lambda: store.execute("history.open", {"id": "h_" + first}, clock=count))
        results, search_stats = measure(lambda: store.execute("history.search", {"query": "校验码"}, clock=count))
        assert opened["found"] and len(results["results"]) == min(count, 8)
        store.close()
        records.append({"history_records": count, "restore": restore, "open": open_stats, "search": search_stats})
    report = {"status": "synthetic_history_python_allocation_profile", "records": records,
              "measured_scope": "tracemalloc Python allocations; excludes SQLite C allocator and model tensors"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
