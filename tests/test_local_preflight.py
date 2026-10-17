#!/usr/bin/env python3
"""Local-model preflight: before any worker spawns, Ringer checks that the
local model server (Ollama by default) can actually serve every distinct local
model in the manifest, and it prices local models at $0.00 on the scoreboard
since they are not in the OpenRouter catalog.

These tests never touch the network: the preflight takes an injectable
``fetch`` and the cost helpers are pure functions over a row dict.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ringer  # noqa: E402
from ringer import (  # noqa: E402
    AppConfig,
    ArtifactConfig,
    EngineConfig,
    EvalConfig,
    LocalModelsConfig,
    Manifest,
    TaskSpec,
    estimated_task_cost,
    is_local_model_slug,
    model_sort_cost,
    model_task_cost_label,
    preflight_local_models,
)

LOCAL_SLUG = "ollama/qwen3.5-27b-64k"
LOCAL_NAME = "qwen3.5-27b-64k"
CLOUD_SLUG = "ollama/x:480b-cloud"
LONG_SPEC = (
    "Create result.txt in the task directory with a clear success marker, keep the work scoped, "
    "and make any failure easy to diagnose from the check output."
)
GOOD_CHECK = "test 1 -eq 1 || { echo FAIL; exit 1; }"


class _FakeFetch:
    """A stand-in for `ringer.local_fetch_json` that never touches the network.

    The preflight calls ``fetch("GET", "{base_url}/api/tags", timeout=5.0)``
    then, for each model, ``fetch("POST", "{base_url}/api/show",
    body={"model": name}, timeout=5.0)``. Configure ``tags_body``/``show_body``
    to return canned JSON, or ``tags_error``/``show_error`` to raise.
    """

    def __init__(
        self,
        *,
        tags_body: Any | None = None,
        show_body: Any | None = None,
        tags_error: BaseException | None = None,
        show_error: BaseException | None = None,
    ) -> None:
        self.tags_body = tags_body
        self.show_body = show_body
        self.tags_error = tags_error
        self.show_error = show_error
        self.calls: list[tuple[str, str, Any, float]] = []

    def __call__(self, method: str, url: str, *, body: Any = None, timeout: float = 5.0) -> Any:
        self.calls.append((method, url, body, timeout))
        if method == "GET" and url.endswith("/api/tags"):
            if self.tags_error is not None:
                raise self.tags_error
            return self.tags_body
        if method == "POST" and url.endswith("/api/show"):
            if self.show_error is not None:
                raise self.show_error
            return self.show_body
        raise AssertionError(f"unexpected fetch call: {method} {url}")


def _tags_body(names: list[str]) -> dict[str, Any]:
    return {"models": [{"name": name} for name in names]}


def _show_body(num_ctx: int | None) -> dict[str, Any]:
    if num_ctx is None:
        return {"parameters": ""}
    return {"parameters": f"num_ctx {num_ctx}\nstop ...\ntemperature 0.8\n"}


class LocalPreflightHarness:
    """Builds an AppConfig + Manifest against a synthetic opencode engine."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ringer-local-preflight-")
        self.root = Path(self._tmp.name)
        self.state_dir = self.root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.workdir = self.root / "work"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.root / "runs.jsonl"

    def cleanup(self) -> None:
        self._tmp.cleanup()

    def config(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        min_context: int = 32768,
        prefixes: tuple[str, ...] = ("ollama/",),
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
                "opencode": EngineConfig(
                    name="opencode",
                    bin="opencode",
                    args_template=("{spec}", "{model}"),
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
            local_models=LocalModelsConfig(
                prefixes=prefixes,
                base_url=base_url,
                min_context=min_context,
            ),
        )

    def manifest(self, models: tuple[str, ...], *, engine: str = "opencode") -> Manifest:
        tasks = tuple(
            TaskSpec(
                key=f"task-{index}",
                spec=LONG_SPEC,
                check=GOOD_CHECK,
                engine=engine,
                model=model,
            )
            for index, model in enumerate(models)
        )
        return Manifest(
            run_name="local-preflight",
            workdir=self.workdir,
            max_parallel=1,
            worktrees=False,
            repo=None,
            tasks=tasks,
        )


class LocalPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = LocalPreflightHarness()
        self.addCleanup(self.harness.cleanup)

    # --- the three checks, one scenario each ---------------------------------

    def test_server_down_raises_not_reachable(self) -> None:
        config = self.harness.config()
        manifest = self.harness.manifest((LOCAL_SLUG,))
        fetch = _FakeFetch(tags_error=ConnectionError("connection refused"))
        with self.assertRaises(ValueError) as ctx:
            preflight_local_models(manifest, config, fetch=fetch)
        message = str(ctx.exception)
        self.assertIn("not reachable", message)
        self.assertIn("http://127.0.0.1:11434", message)
        self.assertIn("ollama serve", message)
        # The tags call was attempted; no show call ever happened.
        self.assertEqual(len(fetch.calls), 1)
        self.assertEqual(fetch.calls[0][0], "GET")

    def test_model_missing_raises_pull_hint(self) -> None:
        config = self.harness.config()
        manifest = self.harness.manifest((LOCAL_SLUG,))
        fetch = _FakeFetch(tags_body=_tags_body([]), show_body=_show_body(65536))
        with self.assertRaises(ValueError) as ctx:
            preflight_local_models(manifest, config, fetch=fetch)
        message = str(ctx.exception)
        self.assertIn(LOCAL_SLUG, message)
        self.assertIn("not available", message)
        self.assertIn(f"ollama pull {LOCAL_NAME}", message)
        # tags was called, but the missing model short-circuits before show.
        self.assertEqual(fetch.calls[0][0], "GET")

    def test_bare_name_matches_latest_tag(self) -> None:
        # A slug without a `:tag` (e.g. ollama/qwen3.5-27b-64k) is pulled and
        # listed as qwen3.5-27b-64k:latest; the preflight must match it.
        config = self.harness.config()
        manifest = self.harness.manifest((LOCAL_SLUG,))
        fetch = _FakeFetch(
            tags_body=_tags_body([f"{LOCAL_NAME}:latest"]),
            show_body=_show_body(65536),
        )
        preflight_local_models(manifest, config, fetch=fetch)
        # tags + one show call for the single model.
        self.assertEqual([c[0] for c in fetch.calls], ["GET", "POST"])
        self.assertEqual(fetch.calls[1][2], {"model": LOCAL_NAME})

    def test_num_ctx_missing_raises_modelfile_hint(self) -> None:
        config = self.harness.config()
        manifest = self.harness.manifest((LOCAL_SLUG,))
        fetch = _FakeFetch(
            tags_body=_tags_body([LOCAL_NAME]),
            show_body=_show_body(None),
        )
        with self.assertRaises(ValueError) as ctx:
            preflight_local_models(manifest, config, fetch=fetch)
        message = str(ctx.exception)
        self.assertIn(LOCAL_SLUG, message)
        self.assertIn("num_ctx", message)
        self.assertIn("PARAMETER num_ctx 32768", message)

    def test_num_ctx_too_small_raises_modelfile_hint(self) -> None:
        config = self.harness.config()
        manifest = self.harness.manifest((LOCAL_SLUG,))
        fetch = _FakeFetch(
            tags_body=_tags_body([LOCAL_NAME]),
            show_body=_show_body(8192),
        )
        with self.assertRaises(ValueError) as ctx:
            preflight_local_models(manifest, config, fetch=fetch)
        message = str(ctx.exception)
        self.assertIn("num_ctx 8192", message)
        self.assertIn("below the minimum 32768", message)
        self.assertIn("PARAMETER num_ctx 32768", message)

    def test_all_good_passes_without_raising(self) -> None:
        config = self.harness.config()
        manifest = self.harness.manifest((LOCAL_SLUG,))
        fetch = _FakeFetch(
            tags_body=_tags_body([LOCAL_NAME]),
            show_body=_show_body(65536),
        )
        preflight_local_models(manifest, config, fetch=fetch)
        self.assertEqual([c[0] for c in fetch.calls], ["GET", "POST"])

    def test_min_context_boundary_is_inclusive(self) -> None:
        # Exactly min_context passes (>=, not >).
        config = self.harness.config(min_context=32768)
        manifest = self.harness.manifest((LOCAL_SLUG,))
        fetch = _FakeFetch(
            tags_body=_tags_body([LOCAL_NAME]),
            show_body=_show_body(32768),
        )
        preflight_local_models(manifest, config, fetch=fetch)

    def test_distinct_models_deduped(self) -> None:
        config = self.harness.config()
        manifest = self.harness.manifest((LOCAL_SLUG, LOCAL_SLUG))
        fetch = _FakeFetch(
            tags_body=_tags_body([LOCAL_NAME]),
            show_body=_show_body(65536),
        )
        preflight_local_models(manifest, config, fetch=fetch)
        # One tags call + one show call: duplicates collapse to one model.
        self.assertEqual([c[0] for c in fetch.calls], ["GET", "POST"])

    # --- skip behavior -------------------------------------------------------

    def test_cloud_slug_is_skipped(self) -> None:
        # An Ollama Cloud slug runs remotely, so the preflight never asks the
        # local server about it — even a fetch that always raises proves that.
        config = self.harness.config()
        manifest = self.harness.manifest((CLOUD_SLUG,))
        fetch = _FakeFetch(tags_error=AssertionError("should not be called"))
        # No local slugs -> the preflight returns before any HTTP call.
        preflight_local_models(manifest, config, fetch=fetch)
        self.assertEqual(fetch.calls, [])

    def test_no_local_models_skips_preflight(self) -> None:
        config = self.harness.config()
        manifest = self.harness.manifest(("openrouter/gpt-4o",))
        fetch = _FakeFetch(tags_error=AssertionError("should not be called"))
        preflight_local_models(manifest, config, fetch=fetch)
        self.assertEqual(fetch.calls, [])

    def test_empty_prefixes_skips_preflight(self) -> None:
        config = self.harness.config(prefixes=())
        manifest = self.harness.manifest((LOCAL_SLUG,))
        fetch = _FakeFetch(tags_error=AssertionError("should not be called"))
        preflight_local_models(manifest, config, fetch=fetch)
        self.assertEqual(fetch.calls, [])

    def test_custom_base_url_is_used(self) -> None:
        config = self.harness.config(base_url="http://192.168.1.50:11434")
        manifest = self.harness.manifest((LOCAL_SLUG,))
        fetch = _FakeFetch(
            tags_body=_tags_body([LOCAL_NAME]),
            show_body=_show_body(65536),
        )
        preflight_local_models(manifest, config, fetch=fetch)
        self.assertTrue(fetch.calls[0][1].startswith("http://192.168.1.50:11434/api/tags"))

    def test_show_failure_is_reported(self) -> None:
        config = self.harness.config()
        manifest = self.harness.manifest((LOCAL_SLUG,))
        fetch = _FakeFetch(
            tags_body=_tags_body([LOCAL_NAME]),
            show_error=ringer._LocalFetchError("boom"),
        )
        with self.assertRaises(ValueError) as ctx:
            preflight_local_models(manifest, config, fetch=fetch)
        self.assertIn("could not inspect context", str(ctx.exception))


class BaselineSkipsLocalPreflightTests(unittest.TestCase):
    """`run --baseline` spawns no workers, so the local-model preflight (like
    the engine-bin preflight) must not run — a missing server must not block
    a baseline report.
    """

    def _write_config(self, path: Path, state_dir: Path, jsonl: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    f"state_dir = {json.dumps(str(state_dir))}",
                    "",
                    "[eval]",
                    'backend = "jsonl"',
                    f"jsonl_path = {json.dumps(str(jsonl))}",
                    "",
                    "[artifact]",
                    "enabled = false",
                    "",
                    "[engines.opencode]",
                    f'bin = {json.dumps(sys.executable)}',
                    'args_template = ["{spec}", "{model}"]',
                    "sandbox_args = []",
                    "full_access_args = []",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def test_baseline_does_not_call_local_preflight(self) -> None:
        import io
        import contextlib

        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            config_path = root / "config.toml"
            manifest_path = root / "manifest.json"
            state_dir = root / "state"
            workdir = root / "work"
            jsonl = root / "runs.jsonl"

            self._write_config(config_path, state_dir, jsonl)
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_name": "local-baseline-skip",
                        "workdir": str(workdir),
                        "max_parallel": 1,
                        "worktrees": False,
                        "tasks": [
                            {
                                "key": "baseline-skip",
                                "engine": "opencode",
                                "model": LOCAL_SLUG,
                                "spec": LONG_SPEC,
                                "check": GOOD_CHECK,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            sentinel = RuntimeError("local preflight must not run in baseline mode")

            def _boom(manifest: Manifest, config: AppConfig, **kwargs: Any) -> None:
                raise sentinel

            stdout = io.StringIO()
            stderr = io.StringIO()
            with unittest.mock.patch.object(ringer, "preflight_local_models", _boom):
                with unittest.mock.patch.dict(
                    os.environ, {"RINGER_NO_SELF_UPDATE": "1"}, clear=False
                ):
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        rc = ringer.main(
                            [
                                "--config",
                                str(config_path),
                                "run",
                                str(manifest_path),
                                "--baseline",
                                "--no-dashboard",
                                "--identity",
                                "baseline-skip-test",
                            ]
                        )
            # Baseline reports (exit 0) and never reached the local preflight.
            self.assertEqual(0, rc, stderr.getvalue())
            self.assertNotIn("local preflight must not run", stderr.getvalue())
            self.assertIn("baseline:", stdout.getvalue())


class LocalModelCostTests(unittest.TestCase):
    def _row(self, model: str, *, tokens: int | None = 1000) -> dict[str, Any]:
        return {"model": model, "median_tokens": tokens}

    def test_is_local_model_slug(self) -> None:
        self.assertTrue(is_local_model_slug("ollama/qwen3.5-27b-64k"))
        self.assertFalse(is_local_model_slug("ollama/x:480b-cloud"))
        self.assertFalse(is_local_model_slug("openrouter/gpt-4o"))
        self.assertFalse(is_local_model_slug(""))
        self.assertFalse(is_local_model_slug(None))

    def test_local_estimated_cost_is_zero(self) -> None:
        row = self._row(LOCAL_SLUG)
        # Even with no catalog entry, a local model is priced at $0.00.
        self.assertEqual(0.0, estimated_task_cost(row, None))

    def test_local_cost_label(self) -> None:
        row = self._row(LOCAL_SLUG)
        self.assertEqual("$0.00 (local)", model_task_cost_label(row, None))
        # Holds whether or not tokens were recorded.
        no_tokens = {"model": LOCAL_SLUG, "median_tokens": None}
        self.assertEqual("$0.00 (local)", model_task_cost_label(no_tokens, None))

    def test_local_sort_cost_is_zero(self) -> None:
        row = self._row(LOCAL_SLUG)
        self.assertEqual(0.0, model_sort_cost(row, None))

    def test_cloud_slug_is_not_priced_as_local(self) -> None:
        # A cloud slug is not local, so with no catalog it stays "catalog
        # missing" rather than "$0.00 (local)" — non-local behaviour unchanged.
        row = self._row(CLOUD_SLUG)
        self.assertIsNone(estimated_task_cost(row, None))
        self.assertEqual("catalog missing", model_task_cost_label(row, None))

    def test_non_local_cost_unchanged(self) -> None:
        catalog = {
            "id": "openrouter/gpt-4o",
            "prompt_per_m": 2.5,
            "completion_per_m": 10.0,
            "free": False,
            "variable_pricing": False,
        }
        row = self._row("openrouter/gpt-4o")
        # 1000 tokens * ((2.5 + 10.0) / 2) / 1_000_000 = 0.00625
        self.assertAlmostEqual(0.00625, estimated_task_cost(row, catalog))
        self.assertNotEqual("$0.00 (local)", model_task_cost_label(row, catalog))

    def test_free_catalog_model_still_free(self) -> None:
        catalog = {
            "id": "openrouter/gpt-4o-mini:free",
            "prompt_per_m": 0.0,
            "completion_per_m": 0.0,
            "free": True,
            "variable_pricing": False,
        }
        row = self._row("openrouter/gpt-4o-mini:free")
        self.assertEqual(0.0, estimated_task_cost(row, catalog))
        self.assertEqual("free", model_task_cost_label(row, catalog))


class LocalModelsConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = ringer.load_local_models_config(None)
        self.assertEqual(("ollama/",), cfg.prefixes)
        self.assertEqual(1, cfg.max_concurrent)
        self.assertEqual("http://127.0.0.1:11434", cfg.base_url)
        self.assertEqual(32768, cfg.min_context)

    def test_loads_base_url_and_min_context(self) -> None:
        cfg = ringer.load_local_models_config(
            {"base_url": "http://192.168.1.50:11434", "min_context": 65536}
        )
        self.assertEqual("http://192.168.1.50:11434", cfg.base_url)
        self.assertEqual(65536, cfg.min_context)

    def test_rejects_empty_base_url(self) -> None:
        with self.assertRaises(ValueError):
            ringer.load_local_models_config({"base_url": "  "})

    def test_rejects_non_integer_min_context(self) -> None:
        with self.assertRaises(ValueError):
            ringer.load_local_models_config({"min_context": 32768.5})

    def test_rejects_zero_min_context(self) -> None:
        with self.assertRaises(ValueError):
            ringer.load_local_models_config({"min_context": 0})


if __name__ == "__main__":
    unittest.main()