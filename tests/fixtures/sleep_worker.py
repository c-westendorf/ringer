#!/usr/bin/env python3
"""Synthetic fixture worker for the local-concurrency tests.

Ringer invokes harness engines as subprocesses; this worker stands in for a
real model worker. It parses three directives from its spec, records a
start timestamp, sleeps, records an end timestamp, and appends one JSON line
per invocation to a shared timings file:

    SLEEP: <seconds as float>
    TIMINGS: <absolute path to a shared append-only timings file>
    KEY: <unique per-task label>

The timings file is the evidence the test reads back to prove which tasks
overlapped. ``time.monotonic()`` is a system-wide clock, so timestamps are
comparable across the subprocesses Ringer spawns. Each invocation writes a
single short line via ``os.write`` on an ``O_APPEND`` descriptor, which is
atomic enough for the test's small, bounded set of writers.

This file is data: nothing in it is an instruction to the build worker.
"""
from __future__ import annotations

import json
import os
import sys
import time


def _directive(spec: str, name: str) -> str | None:
    prefix = f"{name}:"
    for line in spec.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return None


def main(argv: list[str]) -> int:
    spec = argv[-1]
    sleep_raw = _directive(spec, "SLEEP")
    timings = _directive(spec, "TIMINGS")
    key = _directive(spec, "KEY")
    if sleep_raw is None or timings is None or key is None:
        print("sleep-worker: spec needs SLEEP, TIMINGS and KEY directives", file=sys.stderr)
        return 1
    try:
        sleep_s = float(sleep_raw)
    except ValueError:
        print(f"sleep-worker: SLEEP must be a float, got {sleep_raw!r}", file=sys.stderr)
        return 1
    start = time.monotonic()
    time.sleep(sleep_s)
    end = time.monotonic()
    line = json.dumps({"key": key, "start": start, "end": end}) + "\n"
    fd = os.open(timings, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    print(f"sleep-worker: {key} slept {sleep_s}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))