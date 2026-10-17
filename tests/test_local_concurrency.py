#!/usr/bin/env python3
"""Local-model concurrency: ollama/ tasks share one GPU, so Ringer caps them
separately from the run-wide max_parallel.

These tests drive RingerRunner in-process with a synthetic "sleep" engine
(tests/fixtures/sleep_worker.py). Each worker records a start/end timestamp
to a shared timings file; the tests read it back to prove which tasks
overlapped. ``time.monotonic()`` is a system-wide clock, so timestamps are
comparable across the subprocesses Ringer spawns.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ringer import (  # noqa: E402
    AppConfig,
    ArtifactConfig,
    EngineConfig,
    EvalConfig,
    LocalModelsConfig,
    Manifest,
    RingerRunner,
    TaskSpec,
)
from ringer import load_local_models_config  # noqa: E402

SLEEP_WORKER = ROOT / "tests" / "fixtures" / "sleep_worker.py"


def _sleep_spec(key: str, timings_path: Path, seconds: float) -> str:
    return (
        f"SLEEP: {seconds}\n"
        f"TIMINGS: {timings_path}\n"
        f"KEY: {key}\n"
    )


def _read_timings(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _max_overlap(rows: list[dict[str, object]]) -> int:
    """Largest number of intervals mutually overlapping at any instant."""
    events: list[tuple[float, int]] = []
    for row in rows:
        start = float(row["start"])  # type: ignore[index]
        end = float(row["end"])  # type: ignore[index]
        events.append((start, 1))
        events.append((end + 1e-9, -1))
    events.sort()
    running = 0
    peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    return peak


def _any_overlap(rows: list[dict[str, object]]) -> bool:
    return _max_overlap(rows) >= 2


def _no_overlap(rows: list[dict[str, object]]) -> bool:
    """True when no two intervals overlap (serialized)."""
    ordered = sorted(rows, key=lambda r: float(r["start"]))  # type: ignore[index]
    for a, b in zip(ordered, ordered[1:]):
        if float(a["end"]) > float(b["start"]) + 1e-6:  # type: ignore[index]
            return False
    return True


class _LocalConcurrencyHarness:
    """Builds an in-process RingerRunner against a synthetic sleep engine."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ringer-local-conc-")
        self.root = Path(self._tmp.name)
        self.state_dir = self.root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.workdir = self.root / "work"
        self.jsonl_path = self.root / "runs.jsonl"
        self.timings_path = self.root / "timings.jsonl"

    def cleanup(self) -> None:
        self._tmp.cleanup()

    def config(
        self,
        *,
        local_models: LocalModelsConfig = LocalModelsConfig(),
    ) -> AppConfig:
        return AppConfig(
            path=None,
            identity_default=None,
            state_dir=self.state_dir,
            dashboard_port_base=18787,
            hud_port=8700,
            hud_app_path=None,
            allow_full_access=False,
            eval=EvalConfig(backend="jsonl", jsonl_path=self.jsonl_path),
            engines={
                "sleep": EngineConfig(
                    name="sleep",
                    bin=sys.executable,
                    args_template=(str(SLEEP_WORKER), "{spec}"),
                    full_access_args=(),
                    sandbox_args=(),
                    token_regex=None,
                    token_parser=None,
                    model_report_regex=None,
                    model_default="",
                )
            },
            artifact=ArtifactConfig(
                enabled=False,
                out_template=str(self.state_dir / "artifacts" / "{run_id}.html"),
                report_template=str(self.state_dir / "artifacts" / "{run_id}-report.html"),
                index_out=self.state_dir / "artifacts" / "index.html",
            ),
            local_models=local_models,
        )

    def manifest(
        self,
        tasks: tuple[TaskSpec, ...],
        *,
        max_parallel: int,
        max_local_parallel: int | None,
    ) -> Manifest:
        return Manifest(
            run_name="local-conc",
            workdir=self.workdir,
            max_parallel=max_parallel,
            worktrees=False,
            repo=None,
            tasks=tasks,
            max_local_parallel=max_local_parallel,
        )

    def run(self, manifest: Manifest, config: AppConfig) -> dict[str, object]:
        runner = RingerRunner(
            manifest,
            config=config,
            identity="local-conc-test",
            dashboard_enabled=False,
        )
        rc = asyncio.run(runner.run())
        state_path = self.state_dir / "runs" / f"{runner.run_id}.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["__exit_rc"] = rc
        return state

    def task(self, key: str, model: str, *, seconds: float) -> TaskSpec:
        return TaskSpec(
            key=key,
            spec=_sleep_spec(key, self.timings_path, seconds),
            check="true",
            engine="sleep",
            model=model,
            timeout_s=30,
            max_attempts=1,
        )


class LocalConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._harness = _LocalConcurrencyHarness()
        self.addCleanup(self._harness.cleanup)

    def test_local_serialized_nonlocal_overlap(self) -> None:
        # 4 local + 4 non-local, max_parallel=4, max_local_parallel=1: local
        # tasks take the local slot one at a time, non-local tasks run at the
        # run-wide cap (4) so they overlap with each other.
        harness = self._harness
        local = tuple(
            harness.task(f"local-{i}", "ollama/local-model", seconds=0.35)
            for i in range(4)
        )
        cloud = tuple(
            harness.task(f"cloud-{i}", "openrouter/openai/gpt-oss", seconds=0.35)
            for i in range(4)
        )
        manifest = harness.manifest(
            local + cloud, max_parallel=4, max_local_parallel=1
        )
        config = harness.config()
        state = harness.run(manifest, config)

        timings = _read_timings(harness.timings_path)
        self.assertEqual(8, len(timings), timings)
        local_rows = [r for r in timings if str(r["key"]).startswith("local-")]
        cloud_rows = [r for r in timings if str(r["key"]).startswith("cloud-")]
        self.assertEqual(4, len(local_rows), local_rows)
        self.assertEqual(4, len(cloud_rows), cloud_rows)

        # Local lane cap 1: no two local tasks overlap.
        self.assertTrue(_no_overlap(local_rows), f"local overlapped: {local_rows}")
        # Non-local lane cap 4: non-local tasks overlap with each other.
        self.assertTrue(_any_overlap(cloud_rows), f"cloud did not overlap: {cloud_rows}")
        # Run state JSON records the effective local cap.
        self.assertEqual(1, state["max_local_parallel"])

    def test_cloud_slug_is_not_local(self) -> None:
        # An Ollama Cloud slug (contains "cloud") runs remotely and must NOT
        # take a local slot, so two such tasks overlap even with
        # max_local_parallel=1.
        harness = self._harness
        tasks = (
            harness.task("a", "ollama/x:480b-cloud", seconds=0.3),
            harness.task("b", "ollama/x:480b-cloud", seconds=0.3),
        )
        manifest = harness.manifest(tasks, max_parallel=2, max_local_parallel=1)
        config = harness.config()
        state = harness.run(manifest, config)

        timings = _read_timings(harness.timings_path)
        self.assertEqual(2, len(timings), timings)
        self.assertTrue(_any_overlap(timings), f"cloud slugs serialized: {timings}")
        self.assertEqual(1, state["max_local_parallel"])

    def test_manifest_override_wins_over_config(self) -> None:
        # Config allows 2 concurrent local; the manifest pins it to 1. Two
        # local tasks then serialize (cap 1), proving the override won, and
        # state JSON records the manifest value (1), not the config's (2).
        harness = self._harness
        tasks = (
            harness.task("local-a", "ollama/local-model", seconds=0.3),
            harness.task("local-b", "ollama/local-model", seconds=0.3),
        )
        manifest = harness.manifest(tasks, max_parallel=2, max_local_parallel=1)
        config = harness.config(
            local_models=LocalModelsConfig(prefixes=("ollama/",), max_concurrent=2)
        )
        state = harness.run(manifest, config)

        timings = _read_timings(harness.timings_path)
        self.assertEqual(2, len(timings), timings)
        self.assertTrue(_no_overlap(timings), f"local overlapped under cap 1: {timings}")
        self.assertEqual(1, state["max_local_parallel"])

    def test_config_default_without_manifest_override_allows_two(self) -> None:
        # Symmetric control: with config max_concurrent=2 and no manifest
        # override, two local tasks DO overlap, confirming the config default
        # is what the override replaced in the test above.
        harness = self._harness
        tasks = (
            harness.task("local-a", "ollama/local-model", seconds=0.3),
            harness.task("local-b", "ollama/local-model", seconds=0.3),
        )
        manifest = harness.manifest(tasks, max_parallel=2, max_local_parallel=None)
        config = harness.config(
            local_models=LocalModelsConfig(prefixes=("ollama/",), max_concurrent=2)
        )
        state = harness.run(manifest, config)

        timings = _read_timings(harness.timings_path)
        self.assertEqual(2, len(timings), timings)
        self.assertTrue(_any_overlap(timings), f"local did not overlap under cap 2: {timings}")
        self.assertEqual(2, state["max_local_parallel"])


class ManifestMaxLocalParallelValidationTests(unittest.TestCase):
    def _manifest_obj(self, max_local_parallel: object) -> dict[str, object]:
        return {
            "run_name": "validate",
            "workdir": "/tmp/ringer-validate",
            "max_parallel": 2,
            "tasks": [
                {
                    "key": "a",
                    "spec": "Do the work described here in full detail.",
                    "check": "true",
                }
            ],
            "max_local_parallel": max_local_parallel,
        }

    def test_absent_is_allowed(self) -> None:
        manifest = Manifest.from_obj(
            {
                "run_name": "validate",
                "workdir": "/tmp/ringer-validate",
                "max_parallel": 2,
                "tasks": [
                    {
                        "key": "a",
                        "spec": "Do the work described here in full detail.",
                        "check": "true",
                    }
                ],
            }
        )
        self.assertIsNone(manifest.max_local_parallel)

    def test_positive_int_is_accepted(self) -> None:
        manifest = Manifest.from_obj(self._manifest_obj(3))
        self.assertEqual(3, manifest.max_local_parallel)

    def test_rejects_string(self) -> None:
        with self.assertRaises(ValueError):
            Manifest.from_obj(self._manifest_obj("1"))

    def test_rejects_float(self) -> None:
        with self.assertRaises(ValueError):
            Manifest.from_obj(self._manifest_obj(1.5))

    def test_rejects_bool(self) -> None:
        with self.assertRaises(ValueError):
            Manifest.from_obj(self._manifest_obj(True))

    def test_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            Manifest.from_obj(self._manifest_obj(0))

    def test_rejects_negative(self) -> None:
        with self.assertRaises(ValueError):
            Manifest.from_obj(self._manifest_obj(-1))


class LocalModelsConfigValidationTests(unittest.TestCase):
    def test_defaults_when_absent(self) -> None:
        cfg = load_local_models_config(None)
        self.assertEqual(("ollama/",), cfg.prefixes)
        self.assertEqual(1, cfg.max_concurrent)

    def test_accepts_valid(self) -> None:
        cfg = load_local_models_config(
            {"prefixes": ["ollama/", "llamacpp/"], "max_concurrent": 4}
        )
        self.assertEqual(("ollama/", "llamacpp/"), cfg.prefixes)
        self.assertEqual(4, cfg.max_concurrent)

    def test_rejects_non_table(self) -> None:
        with self.assertRaises(ValueError):
            load_local_models_config("nope")

    def test_rejects_max_concurrent_zero(self) -> None:
        with self.assertRaises(ValueError):
            load_local_models_config({"max_concurrent": 0})

    def test_rejects_max_concurrent_string(self) -> None:
        with self.assertRaises(ValueError):
            load_local_models_config({"max_concurrent": "1"})

    def test_rejects_max_concurrent_bool(self) -> None:
        with self.assertRaises(ValueError):
            load_local_models_config({"max_concurrent": True})

    def test_rejects_prefixes_not_list(self) -> None:
        with self.assertRaises(ValueError):
            load_local_models_config({"prefixes": "ollama/"})

    def test_rejects_empty_prefix_string(self) -> None:
        with self.assertRaises(ValueError):
            load_local_models_config({"prefixes": ["ollama/", ""]})

    def test_rejects_non_string_prefix(self) -> None:
        with self.assertRaises(ValueError):
            load_local_models_config({"prefixes": ["ollama/", 5]})


if __name__ == "__main__":
    unittest.main(verbosity=2)