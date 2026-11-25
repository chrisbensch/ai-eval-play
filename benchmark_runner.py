#!/usr/bin/env python3
"""Batch coding benchmark runner and HTML scorecards."""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import shlex
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.client import HTTPException
from pathlib import Path

from benchmark_config import connection_settings
from benchmark_suite import ModelReply

# ==============================================================================
# CONFIGURATION: Select models here. See README.md for connection settings.
# ==============================================================================

BASE_URL, API_KEY, OMLX_SSH_HOST = connection_settings()

MODELS_TO_TEST = [
    #"blm-omlx/occamy-1.0-abliterated-mtp-mlx-4bit",
    "opencode-go/deepseek-v4.1-flash",
    "blm-omlx/Qwen3.8-27B-NVFP4-QAD",
    "blm-omlx/Qwopus3.8-27B-Coder-Flash-8bit",
    "blm-omlx/Inkling-2-Mini-Unbound",
    "blm-omlx/Qwen3.8-27B-Defiance-Uncensored-Q6_K_P",
    "nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b",
]

MAX_TOKENS = 32768
THINKING_BUDGET = 4096
TEMPERATURE = 0.6
REQUEST_TIMEOUT = 1500.0
REPAIR_REQUEST_TIMEOUT = 3600.0
MAX_ARCHITECTURE_WORDS = 150
OUTPUT_DIR = Path("benchmark_results")
RUNTIME_IMAGE = "python:3.12-slim"  # Load once with: docker pull python:3.12-slim
OMLX_PROVIDER_PREFIX = "blm-omlx/"
# ==============================================================================

BENCHMARK_PROMPT = f"""You are a senior systems engineer. Build a complete, production-grade, single-file Python 3.12+ module implementing a persistent, crash-safe, asynchronous task scheduler and worker pool with dependency DAG resolution.

### Requirements
1. **Zero External Dependencies:** Use ONLY the Python 3.12+ standard library (`asyncio`, `sqlite3`, `json`, `time`, `uuid`, `dataclasses`, `typing`, `logging`, etc.).
2. **DAG & Dependency Management:**
   - Tasks can declare upstream `depends_on: list[str]` (task IDs).
   - Detect and reject circular dependencies immediately upon submission with a custom `CycleDetectedError`.
   - A task may only transition to `READY` when all upstream dependencies have reached `SUCCEEDED`.
   - If an upstream task fails permanently, all downstream dependent tasks must automatically cascade to `CANCELLED` with a clear failure reason.
3. **Persistence & Crash Recovery:**
   - Persist all task states, payloads, retry counts, and timestamps in SQLite using WAL mode.
   - Implement an atomic lease mechanism (`lease_owner`, `lease_expires_at`) so multiple worker loops cannot claim the same task concurrently.
   - Upon scheduler startup, automatically recover any tasks stuck in `RUNNING` whose lease has expired, returning them to `READY` (or `FAILED` if retries are exhausted).
4. **Execution, Retries & Timeouts:**
   - Support both `async def` and synchronous callables (run sync functions via `asyncio.to_thread`).
   - Enforce per-task execution timeouts (`timeout_seconds`).
   - Implement exponential backoff with jitter for failed/timed-out tasks up to `max_retries`. Keep retry delays at least 50 ms and the first two retry delays below 1 second for the benchmark.
5. **Graceful Shutdown:**
   - Provide an `async def shutdown(grace_period: float)` method that stops claiming new tasks, waits for in-flight tasks up to `grace_period`, and cancels/re-queues any remaining tasks cleanly.

### Public API used by the benchmark
- Export `Scheduler(db_path, handlers, *, workers=2, lease_seconds=5.0)` and `CycleDetectedError`. `handlers` maps names to sync or async callables that each take one JSON-compatible payload.
- Provide `async submit(task_id, handler, payload, *, depends_on=None, timeout_seconds=5.0, max_retries=0)`. Allow forward dependency IDs so cycles can be tested on submission. `max_retries` counts extra attempts after the initial attempt.
- Provide `async start()` to initialize storage, recover expired leases, and launch workers in the background. The benchmark awaits `start()` before the first `submit()` on each scheduler instance.
- Provide `async status(task_id)` to return the uppercase state string and the `async shutdown(grace_period)` method above.

### Output Format
Provide your response in two strict sections:
1. **`scheduler.py`:** The complete, importable implementation in one fenced Python code block. Include type hints and docstrings. Omit tests, demos, and a `__main__` block. The benchmark supplies the tests.
2. **`architecture.md`:** After the closing Python fence, explain your SQLite locking and lease strategy, cycle detection, and cascade cancellation in at most {MAX_ARCHITECTURE_WORDS} words."""


@dataclass
class CriterionResult:
    name: str
    short_label: str
    passed: bool
    points: int
    max_points: int
    details: str


@dataclass
class ModelBenchmarkSummary:
    model: str
    score: int
    max_score: int
    gen_seconds: float
    completion_tokens: int
    tokens_per_sec: float
    exec_seconds: float
    status: str
    criteria: list[CriterionResult] = field(default_factory=list)
    architecture_prose: str = ""
    generated_code: str = ""
    raw_response: str = ""


def request_chat_completion(
    model: str, messages: list[dict[str, str]] | None = None, *, repair: bool = False,
) -> ModelReply:
    """Query the OpenAI-compatible chat completions endpoint."""
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages if messages is not None else [
            {
                "role": "system",
                "content": (
                    "Start with scheduler.py in a complete fenced Python code block. Follow it "
                    "with the short architecture.md note. Omit opening remarks and test code."
                ),
            },
            {"role": "user", "content": BENCHMARK_PROMPT},
        ],
        "temperature": TEMPERATURE,
        "stream": False,
        "tool_choice": "none",
    }
    if not repair:
        payload["max_tokens"] = MAX_TOKENS
        if model.startswith(OMLX_PROVIDER_PREFIX):
            payload["thinking_budget"] = THINKING_BUDGET
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    start = time.perf_counter()
    timeout = REPAIR_REQUEST_TIMEOUT if repair else REQUEST_TIMEOUT
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_body = resp.read().decode("utf-8")
    elapsed = time.perf_counter() - start

    data = json.loads(raw_body)
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("API response has no completion choices.")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("API response has no completion message.")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ValueError("API response content is not text.")

    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    if type(input_tokens) is not int or input_tokens < 0:
        input_tokens = None
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    estimated_output = type(completion_tokens) is not int or completion_tokens < 0
    if estimated_output:
        reasoning_text = message.get("reasoning_content")
        reasoning_text = reasoning_text if isinstance(reasoning_text, str) else ""
        completion_tokens = max(1, (len(content) + len(reasoning_text)) // 4)

    finish_reason = choice.get("finish_reason")
    if message.get("tool_calls") or message.get("function_call"):
        finish_reason = "tool_calls"
    return ModelReply(
        content, completion_tokens, elapsed, str(finish_reason or "unknown"),
        input_tokens=input_tokens, estimated_output=estimated_output,
    )


def call_local_model(
    model: str, messages: list[dict[str, str]] | None = None, *, repair: bool = False,
) -> tuple[str, int, float, str]:
    reply = request_chat_completion(model, messages, repair=repair)
    return reply.text, reply.output_tokens, reply.seconds, reply.finish_reason


OMLX_UNLOAD_SCRIPT = """
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def unload(model):
    settings = json.loads((Path.home() / ".omlx" / "settings.json").read_text())
    port = settings["server"]["port"]
    api_key = settings.get("auth", {}).get("api_key")
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    base_url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + 60

    def request(path, method="GET"):
        req = Request(base_url + path, headers=headers, method=method)
        return urlopen(req, timeout=max(0.1, deadline - time.monotonic()))

    unload_sent = False
    while time.monotonic() < deadline:
        with request("/status") as response:
            models = json.load(response)["models"]
        entry = next((item for item in models if item["id"] == model), None)
        if entry is None:
            entry = next((item for item in models if item.get("model_alias") == model), None)
        if entry is None:
            raise ValueError(f"Model {model!r} was not found in oMLX's status API.")
        if not entry["loaded"] and not entry.get("is_loading", False):
            print("Unload confirmed." if unload_sent else "Already unloaded.")
            return
        if not unload_sent and not entry.get("is_loading", False):
            model_id = entry.get("source_model_id") or entry["id"]
            try:
                with request(f"/{quote(model_id, safe='')}/unload", method="POST"):
                    unload_sent = True
            except HTTPError as exc:
                # The model can finish loading or be evicted between status and POST.
                if exc.code not in (400, 409):
                    raise
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"oMLX still reports {model!r} loaded or loading after 60 seconds.")


try:
    unload(sys.argv[1])
except (OSError, ValueError, KeyError) as exc:
    sys.exit(str(exc))
"""


def unload_previous_model(model: str) -> None:
    """Ask the oMLX server on the OmniRoute host to release this model."""
    if not model.startswith(OMLX_PROVIDER_PREFIX):
        return
    if not OMLX_SSH_HOST:
        raise RuntimeError("Set OMLX_SSH_HOST to the Mac running oMLX before switching local models.")
    model_id = model.removeprefix(OMLX_PROVIDER_PREFIX)
    print(f"    Releasing {model_id} from oMLX memory...", flush=True)
    command = shlex.join(["python3", "-", model_id])
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
         "-o", "ConnectTimeout=10", OMLX_SSH_HOST, command],
        input=OMLX_UNLOAD_SCRIPT,
        capture_output=True,
        text=True,
        timeout=75,
    )
    if result.returncode:
        diagnostic = (result.stderr or result.stdout).strip()
        raise RuntimeError(diagnostic or f"SSH exited with status {result.returncode}.")
    print(f"    {result.stdout.strip()}")


def extract_artifacts(raw_text: str) -> tuple[str, str]:
    """Extract architecture prose and largest Python code block."""
    cleaned = re.sub(r"<think>.*?</think>", "",
                     raw_text, flags=re.DOTALL).strip()
    python_fence = re.compile(
        r"^[ \t]*(?P<fence>`{3,}|~{3,})python[ \t]*\r?\n"
        r"(.*?)^[ \t]*(?P=fence)[ \t]*\r?$",
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    code_blocks = [match[2] for match in python_fence.finditer(cleaned)]
    if not code_blocks:
        raise ValueError("No ```python ... ``` code block found in output.")
    scheduler_code = max(code_blocks, key=len).strip()
    prose = python_fence.sub("", cleaned)
    prose = re.sub(r"(?m)^[ \t]*(?:`{3,}|~{3,})(?:markdown|md)?[ \t]*$", "",
                   prose, flags=re.IGNORECASE)
    prose = "\n".join(
        line for line in prose.splitlines()
        if line.strip(" #*`:\t").lower() not in {"architecture.md", "scheduler.py"}
    )
    return prose.strip(), scheduler_code


def check_architecture_prose(prose: str) -> CriterionResult:
    if not prose:
        return CriterionResult("1. Architecture Section", "Arch", False, 0, 10, "No architecture prose found.")
    word_count = len(prose.split())
    passed = word_count <= MAX_ARCHITECTURE_WORDS
    return CriterionResult(
        "1. Architecture Section",
        "Arch",
        passed,
        10 if passed else 4,
        10,
        f"Word count: {word_count} words (target ≤{MAX_ARCHITECTURE_WORDS}).",
    )


def check_static_requirements(code: str) -> list[CriterionResult]:
    results: list[CriterionResult] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [
            CriterionResult("2. Valid Python Syntax", "Syntax", False,
                            0, 15, f"SyntaxError line {exc.lineno}: {exc.msg}")
        ]

    results.append(CriterionResult("2. Valid Python Syntax",
                   "Syntax", True, 15, 15, "AST parsed cleanly."))

    stdlib_allowed = set(sys.stdlib_module_names) | {"__future__"}
    third_party: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in stdlib_allowed:
                    third_party.append(root)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".")[0]
            if root not in stdlib_allowed:
                third_party.append(root)

    no_ext = len(third_party) == 0
    results.append(
        CriterionResult(
            "3. Zero External Dependencies",
            "Stdlib",
            no_ext,
            10 if no_ext else 0,
            10,
            "Stdlib only." if no_ext else f"Forbidden imports: {sorted(set(third_party))}",
        )
    )

    code_lower = code.lower()
    checks = [
        ("CycleDetectedError", "cycledetectederror" in code_lower),
        ("SQLite WAL mode", "wal" in code_lower),
        ("Atomic lease / transaction", ("returning" in code_lower)
         or ("begin immediate" in code_lower)),
        ("Thread offloading (to_thread)", "to_thread" in code_lower),
        ("Type annotations", sum(isinstance(n, ast.arg)
         and n.annotation is not None for n in ast.walk(tree)) >= 5),
    ]
    passed_checks = [label for label, ok in checks if ok]
    failed_checks = [label for label, ok in checks if not ok]
    results.append(
        CriterionResult(
            "4. Structural & Concurrency Primitives",
            "Primitives",
            len(failed_checks) == 0,
            len(passed_checks) * 5,
            25,
            f"Matched {len(passed_checks)}/5" +
            (f" | Missing: {', '.join(failed_checks)}" if failed_checks else ""),
        )
    )
    return results


RUNTIME_TESTS = """
import asyncio
import sys
import threading
import time
import unittest
from pathlib import Path

from scheduler import CycleDetectedError, Scheduler


async def wait_for_status(scheduler, task_id, expected):
    async with asyncio.timeout(4):
        while await scheduler.status(task_id) != expected:
            await asyncio.sleep(0.02)


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def start_scheduler(self, scheduler):
        await scheduler.start()
        self.addAsyncCleanup(scheduler.shutdown, 0.2)

    async def test_diamond_and_sync_worker(self):
        completed = []
        loop_thread = threading.get_ident()
        graph_submitted = asyncio.Event()

        async def record(task_id):
            if task_id == "A":
                await graph_submitted.wait()
            if task_id == "C":
                self.assertIn("A", completed)
            if task_id == "D":
                self.assertIn("B", completed)
                self.assertIn("C", completed)
            await asyncio.sleep(0.05)
            completed.append(task_id)

        def record_sync(task_id):
            self.assertNotEqual(threading.get_ident(), loop_thread)
            self.assertIn("A", completed)
            completed.append(task_id)

        scheduler = Scheduler("diamond.db", {"async": record, "sync": record_sync}, workers=2)
        await self.start_scheduler(scheduler)
        await scheduler.submit("A", "async", "A")
        await scheduler.submit("B", "sync", "B", depends_on=["A"])
        await scheduler.submit("C", "async", "C", depends_on=["A"])
        await scheduler.submit("D", "async", "D", depends_on=["B", "C"])
        graph_submitted.set()
        await wait_for_status(scheduler, "D", "SUCCEEDED")
        self.assertEqual(completed[0], "A")
        self.assertCountEqual(completed[1:-1], ["B", "C"])
        self.assertEqual(completed[-1], "D")

    async def test_rejects_cycle_on_submission(self):
        scheduler = Scheduler("cycle.db", {"unused": lambda _: None})
        await self.start_scheduler(scheduler)
        await scheduler.submit("A", "unused", None, depends_on=["B"])
        with self.assertRaises(CycleDetectedError):
            await scheduler.submit("B", "unused", None, depends_on=["A"])

    async def test_permanent_failure_cancels_all_descendants(self):
        ran = []
        graph_submitted = asyncio.Event()

        async def fail(_):
            await graph_submitted.wait()
            raise RuntimeError("expected failure")

        scheduler = Scheduler("cascade.db", {"fail": fail, "record": ran.append})
        await self.start_scheduler(scheduler)
        await scheduler.submit("A", "fail", None)
        await scheduler.submit("B", "record", "B", depends_on=["A"])
        await scheduler.submit("C", "record", "C", depends_on=["B"])
        graph_submitted.set()
        await wait_for_status(scheduler, "C", "CANCELLED")
        self.assertEqual(await scheduler.status("A"), "FAILED")
        self.assertEqual(await scheduler.status("B"), "CANCELLED")
        self.assertEqual(ran, [])

    async def test_timeout_and_delayed_retry(self):
        timed_out = []

        async def handler(_):
            if not timed_out:
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    timed_out.append(time.monotonic())
                    raise
            else:
                timed_out.append(time.monotonic())

        scheduler = Scheduler("retry.db", {"work": handler}, workers=1)
        await self.start_scheduler(scheduler)
        await scheduler.submit("A", "work", None, timeout_seconds=0.15, max_retries=1)
        await wait_for_status(scheduler, "A", "SUCCEEDED")
        self.assertEqual(len(timed_out), 2)
        self.assertGreaterEqual(timed_out[1] - timed_out[0], 0.04)

    async def test_restart_recovers_expired_lease(self):
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-B", __file__, "--crash-worker",
        )
        try:
            async with asyncio.timeout(4):
                while not Path("lease-claimed").exists():
                    await asyncio.sleep(0.02)
        finally:
            if child.returncode is None:
                child.kill()
            await child.wait()

        await asyncio.sleep(1.1)
        ran = []
        scheduler = Scheduler("crash.db", {"work": ran.append}, workers=1, lease_seconds=1.0)
        await self.start_scheduler(scheduler)
        await wait_for_status(scheduler, "A", "SUCCEEDED")
        self.assertEqual(ran, ["recovered"])


async def crash_worker():
    async def hold(_):
        Path("lease-claimed").touch()
        await asyncio.Event().wait()

    scheduler = Scheduler("crash.db", {"work": hold}, workers=1, lease_seconds=1.0)
    await scheduler.start()
    await scheduler.submit("A", "work", "recovered", max_retries=1)
    await asyncio.Event().wait()


if __name__ == "__main__":
    if sys.argv[1:] == ["--crash-worker"]:
        asyncio.run(crash_worker())
    else:
        result = unittest.main(verbosity=2, exit=False).result
        if not result.wasSuccessful() or result.testsRun != 5 or result.skipped:
            sys.exit(1)
        Path("tests.passed").touch()
"""


RUNTIME_BOOTSTRAP = """
import json
import resource
import subprocess
import sys
from pathlib import Path

payload = json.load(sys.stdin)
Path("scheduler.py").write_text(payload["code"], encoding="utf-8")
Path("check_scheduler.py").write_text(payload["tests"], encoding="utf-8")
limit = 4 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
with open("test.log", "w+b") as log:
    result = subprocess.run(
        [sys.executable, "-B", "check_scheduler.py"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    log.seek(max(0, log.tell() - 4096))
    print(log.read().decode("utf-8", errors="replace"))
if result.returncode != 0:
    sys.exit(1)
if not Path("tests.passed").is_file():
    sys.exit("Scheduler tests exited before completing.")
"""


def check_runtime_execution(code: str, timeout_sec: float = 45.0) -> tuple[CriterionResult, float]:
    container = f"bench-scheduler-{uuid.uuid4().hex}"
    command = [
        "docker", "run", "--rm", "--interactive", "--init", "--name", container,
        "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--user=65534:65534",
        "--pids-limit=64", "--memory=256m", "--cpus=1", "--log-driver=none",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=32m,mode=1777", "--workdir=/tmp",
        "--entrypoint=timeout", RUNTIME_IMAGE, "--kill-after=2s", f"{timeout_sec:g}s",
        "python", "-I", "-B", "-c", RUNTIME_BOOTSTRAP,
    ]

    start = time.perf_counter()
    passed = False
    try:
        proc = subprocess.run(
            command,
            input=json.dumps({"code": code, "tests": RUNTIME_TESTS}),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec + 5,
        )
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        try:
            subprocess.run(
                ["docker", "rm", "--force", container],
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if isinstance(exc, KeyboardInterrupt):
            raise
        details = f"Runtime tests timed out after {timeout_sec:g}s."
    except OSError as exc:
        details = f"Sandbox unavailable: {exc}"
    else:
        diagnostic = " | ".join((proc.stderr or proc.stdout).strip().splitlines()[-8:])
        if proc.returncode == 0:
            passed = True
            details = "All five runner-owned scheduler tests passed."
        elif proc.returncode in (124, 137):
            details = f"Sandbox stopped the tests (exit {proc.returncode}, timeout or resource limit)."
        elif proc.returncode in (125, 126, 127):
            details = f"Sandbox unavailable (requires Docker and {RUNTIME_IMAGE}): {diagnostic}"
        else:
            details = f"Runtime tests failed (exit {proc.returncode}): {diagnostic}"

    criterion = CriterionResult(
        "5. End-to-End Test Execution", "Runtime", passed, 40 if passed else 0, 40, details
    )
    return criterion, time.perf_counter() - start


def safe_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_")


def run_benchmark_for_model(model: str) -> ModelBenchmarkSummary:
    slug = safe_slug(model)
    model_dir = OUTPUT_DIR / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Testing model: {model}")
    print(f"    Querying {BASE_URL} (timeout {REQUEST_TIMEOUT:.0f}s)...")

    try:
        raw_response, tokens, gen_elapsed, finish_reason = call_local_model(model)
    except (OSError, ValueError, HTTPException) as exc:
        print(f"    [ERROR] API call failed: {exc}")
        return ModelBenchmarkSummary(model, 0, 100, 0.0, 0, 0.0, 0.0, f"API Error: {exc}")

    tps = tokens / gen_elapsed if gen_elapsed > 0 else 0.0
    (model_dir / "response.md").write_text(raw_response, encoding="utf-8")
    print(
        f"    Generated ~{tokens} tokens in {gen_elapsed:.1f}s "
        f"({tps:.1f} tok/s, finish_reason={finish_reason}).")

    try:
        prose, code = extract_artifacts(raw_response)
    except ValueError as exc:
        if finish_reason in {"tool_calls", "function_call"}:
            status = "Unexpected tool call"
        elif finish_reason == "length":
            status = "Token limit before complete Python block"
        else:
            status = "No Python block"
        print(f"    [FAIL] {status} (finish_reason={finish_reason}). {exc}")
        return ModelBenchmarkSummary(
            model, 0, 100, gen_elapsed, tokens, tps, 0.0, status, raw_response=raw_response
        )

    (model_dir / "scheduler.py").write_text(code, encoding="utf-8")

    results: list[CriterionResult] = [check_architecture_prose(prose)]
    static_results = check_static_requirements(code)
    results.extend(static_results)

    exec_elapsed = 0.0
    if any(r.name == "2. Valid Python Syntax" and r.passed for r in static_results):
        runtime_res, exec_elapsed = check_runtime_execution(code)
        results.append(runtime_res)

    total = sum(r.points for r in results)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"    [{status}] {r.name:<38} {r.points:>2}/{r.max_points:<2} | {r.details}")

    final_status = "PASS" if total >= 80 else "FAIL"
    return ModelBenchmarkSummary(
        model=model,
        score=total,
        max_score=100,
        gen_seconds=gen_elapsed,
        completion_tokens=tokens,
        tokens_per_sec=tps,
        exec_seconds=exec_elapsed,
        status=final_status,
        criteria=results,
        architecture_prose=prose,
        generated_code=code,
        raw_response=raw_response,
    )


def build_stacked_bar_svg(summaries: list[ModelBenchmarkSummary]) -> str:
    """Generate an inline SVG stacked horizontal bar chart of rubric scores."""
    colors = {
        "Arch": "#38bdf8",       # Sky blue
        "Syntax": "#818cf8",     # Indigo
        "Stdlib": "#c084fc",     # Purple
        "Primitives": "#f59e0b",  # Amber
        "Runtime": "#10b981",    # Emerald
    }
    row_h = 44
    pad_top = 36
    pad_left = 260
    chart_w = 500
    svg_w = pad_left + chart_w + 80
    svg_h = pad_top + max(1, len(summaries)) * row_h + 24

    parts = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" class="chart-svg" role="img" aria-label="Score Breakdown Chart">']

    # Legend at top
    lx = pad_left
    for label, col in colors.items():
        parts.append(
            f'<rect x="{lx}" y="8" width="12" height="12" rx="3" fill="{col}"/>')
        parts.append(
            f'<text x="{lx + 17}" y="18" fill="#cbd5e1" font-size="12">{label}</text>')
        lx += 92

    # Grid lines (0, 25, 50, 75, 100)
    for tick in (0, 25, 50, 75, 100):
        gx = pad_left + (tick / 100.0) * chart_w
        parts.append(
            f'<line x1="{gx}" y1="{pad_top - 6}" x2="{gx}" y2="{svg_h - 20}" stroke="#334155" stroke-dasharray="3,3"/>')
        parts.append(
            f'<text x="{gx}" y="{svg_h - 5}" fill="#94a3b8" font-size="11" text-anchor="middle">{tick}</text>')

    for idx, s in enumerate(summaries):
        y = pad_top + idx * row_h + 8
        short_name = s.model if len(s.model) <= 34 else s.model[:31] + "..."
        parts.append(
            f'<text x="{pad_left - 12}" y="{y + 17}" fill="#f1f5f9" font-size="12.5" '
            f'font-family="ui-monospace, monospace" text-anchor="end">{html.escape(short_name)}</text>'
        )
        # Background track
        parts.append(
            f'<rect x="{pad_left}" y="{y}" width="{chart_w}" height="24" rx="5" fill="#1e293b"/>')

        cur_x = float(pad_left)
        for c in s.criteria:
            if c.points > 0:
                seg_w = (c.points / 100.0) * chart_w
                col = colors.get(c.short_label, "#64748b")
                parts.append(
                    f'<rect x="{cur_x:.1f}" y="{y}" width="{seg_w:.1f}" height="24" fill="{col}">'
                    f'<title>{html.escape(c.name)}: {c.points}/{c.max_points}</title></rect>'
                )
                cur_x += seg_w

        parts.append(
            f'<text x="{pad_left + chart_w + 10}" y="{y + 17}" fill="#f8fafc" font-size="13" '
            f'font-weight="bold">{s.score}/100</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def build_scatter_svg(summaries: list[ModelBenchmarkSummary]) -> str:
    """Generate an inline SVG scatter plot comparing Tokens/Sec vs Rubric Score."""
    svg_w, svg_h = 520, 300
    pad_l, pad_r, pad_t, pad_b = 55, 35, 25, 45
    plot_w = svg_w - pad_l - pad_r
    plot_h = svg_h - pad_t - pad_b

    max_tps = max([s.tokens_per_sec for s in summaries] + [10.0]) * 1.15
    palette = ["#38bdf8", "#10b981", "#f59e0b",
               "#c084fc", "#f43f5e", "#2dd4bf"]

    parts = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" class="chart-svg" role="img" aria-label="Speed vs Score Scatter Plot">']

    # Pass threshold line (80 pts)
    pass_y = pad_t + plot_h * (1.0 - 0.80)
    parts.append(
        f'<line x1="{pad_l}" y1="{pass_y:.1f}" x2="{svg_w - pad_r}" y2="{pass_y:.1f}" stroke="#10b981" stroke-opacity="0.45" stroke-dasharray="4,4"/>')
    parts.append(
        f'<text x="{svg_w - pad_r - 4}" y="{pass_y - 5:.1f}" fill="#10b981" font-size="10" text-anchor="end">Pass Threshold (80)</text>')

    # Y-axis grid (0..100)
    for score_tick in (0, 25, 50, 75, 100):
        yy = pad_t + plot_h * (1.0 - score_tick / 100.0)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{svg_w - pad_r}" y2="{yy:.1f}" stroke="#334155" stroke-width="1"/>')
        parts.append(
            f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">{score_tick}</text>')

    # X-axis ticks
    for i in range(5):
        frac = i / 4.0
        xx = pad_l + frac * plot_w
        val = frac * max_tps
        parts.append(
            f'<text x="{xx:.1f}" y="{svg_h - 22}" fill="#94a3b8" font-size="11" text-anchor="middle">{val:.0f}</text>')

    parts.append(
        f'<text x="{pad_l + plot_w / 2}" y="{svg_h - 5}" fill="#cbd5e1" font-size="11.5" text-anchor="middle">Generation Speed (tokens / sec)</text>')
    parts.append(
        f'<text x="15" y="{pad_t + plot_h / 2}" fill="#cbd5e1" font-size="11.5" text-anchor="middle" transform="rotate(-90 15 {pad_t + plot_h / 2})">Rubric Score</text>')

    for idx, s in enumerate(summaries):
        cx = pad_l + (s.tokens_per_sec / max_tps) * plot_w
        cy = pad_t + plot_h * (1.0 - s.score / 100.0)
        col = palette[idx % len(palette)]
        short_label = s.model.split("/")[-1][:18]
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{col}" stroke="#0f172a" stroke-width="2">'
            f'<title>{html.escape(s.model)}: {s.score}/100 @ {s.tokens_per_sec:.1f} tok/s</title></circle>'
        )
        label_y = cy - 11 if cy > pad_t + 20 else cy + 18
        parts.append(
            f'<text x="{cx:.1f}" y="{label_y:.1f}" fill="#e2e8f0" font-size="10.5" text-anchor="middle">{html.escape(short_label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def generate_html_report(summaries: list[ModelBenchmarkSummary], out_path: Path) -> None:
    """Render a comprehensive, self-contained HTML report with charts and code inspection."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_model = summaries[0].model if summaries else "N/A"
    best_score = f"{summaries[0].score}/100" if summaries else "-"
    fastest = max(
        summaries, key=lambda s: s.tokens_per_sec) if summaries else None
    fastest_str = f"{fastest.tokens_per_sec:.1f} tok/s" if fastest else "-"
    fastest_name = fastest.model.split("/")[-1] if fastest else "-"
    pass_count = sum(1 for s in summaries if s.status == "PASS")
    total_time = sum(s.gen_seconds + s.exec_seconds for s in summaries)

    bar_svg = build_stacked_bar_svg(summaries)
    scatter_svg = build_scatter_svg(summaries)

    table_rows = []
    detail_cards = []

    for rank, s in enumerate(summaries, start=1):
        badge_cls = "badge-pass" if s.status == "PASS" else "badge-fail"
        exec_str = f"{s.exec_seconds:.2f}s" if s.exec_seconds > 0 else "—"
        table_rows.append(f"""
        <tr>
          <td><strong>#{rank}</strong></td>
          <td class="mono">{html.escape(s.model)}</td>
          <td><strong>{s.score}</strong><span class="muted">/{s.max_score}</span></td>
          <td>{s.tokens_per_sec:.1f} <span class="muted">tok/s</span></td>
          <td>{s.completion_tokens:,}</td>
          <td>{s.gen_seconds:.1f}s</td>
          <td>{exec_str}</td>
          <td><span class="badge {badge_cls}">{html.escape(s.status)}</span></td>
        </tr>""")

        crit_rows = []
        for c in s.criteria:
            c_badge = "badge-pass" if c.passed else "badge-fail"
            c_label = "PASS" if c.passed else "FAIL"
            crit_rows.append(f"""
            <tr>
              <td>{html.escape(c.name)}</td>
              <td><span class="badge {c_badge}">{c_label}</span></td>
              <td class="mono">{c.points}/{c.max_points}</td>
              <td>{html.escape(c.details)}</td>
            </tr>""")

        criteria_html = (
            f"""<table class="sub-table">
              <thead><tr><th>Rubric Criterion</th><th>Result</th><th>Points</th><th>Diagnostics</th></tr></thead>
              <tbody>{''.join(crit_rows)}</tbody>
            </table>"""
            if crit_rows
            else f'<p class="muted">No criteria evaluated ({html.escape(s.status)}).</p>'
        )

        arch_block = (
            f'<div class="prose-box">{html.escape(s.architecture_prose)}</div>'
            if s.architecture_prose
            else '<p class="muted">No architecture prose extracted.</p>'
        )

        code_block = (
            f'<pre class="code-box"><code>{html.escape(s.generated_code)}</code></pre>'
            if s.generated_code
            else '<p class="muted">No Python code extracted.</p>'
        )

        open_attr = " open" if rank == 1 else ""
        detail_cards.append(f"""
        <details class="model-card"{open_attr}>
          <summary>
            <span class="summary-left">
              <span class="rank-pill">#{rank}</span>
              <span class="mono">{html.escape(s.model)}</span>
            </span>
            <span class="summary-right">
              <span class="muted">{s.tokens_per_sec:.1f} tok/s</span>
              <strong>{s.score}/100</strong>
              <span class="badge {badge_cls}">{html.escape(s.status)}</span>
            </span>
          </summary>
          <div class="card-body">
            <h4>Rubric Breakdown</h4>
            {criteria_html}
            <h4>Extracted Architecture Summary</h4>
            {arch_block}
            <h4>Generated <code>scheduler.py</code> ({len(s.generated_code.splitlines())} lines)</h4>
            {code_block}
          </div>
        </details>""")

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Local LLM Coding Benchmark Scorecard</title>
  <style>
    :root {{
      --bg: #0b1120;
      --surface: #111827;
      --surface-2: #1e293b;
      --border: #273549;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --pass-bg: rgba(16, 185, 129, 0.15);
      --pass-fg: #34d399;
      --fail-bg: rgba(244, 63, 94, 0.15);
      --fail-fg: #fb7185;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 2rem;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.5;
    }}
    .container {{ max-width: 1240px; margin: 0 auto; }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1rem;
      margin-bottom: 1.75rem;
      flex-wrap: wrap;
      gap: 1rem;
    }}
    h1 {{ margin: 0; font-size: 1.65rem; letter-spacing: -0.02em; }}
    h2 {{ font-size: 1.15rem; margin: 0 0 1rem 0; color: #e2e8f0; }}
    h4 {{ font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin: 1.25rem 0 0.5rem; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.88rem; }}
    .muted {{ color: var(--muted); font-size: 0.88rem; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1.75rem;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.1rem 1.25rem;
    }}
    .kpi-label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .kpi-value {{ font-size: 1.5rem; font-weight: 700; margin: 0.25rem 0; color: var(--accent); }}
    .kpi-sub {{ font-size: 0.82rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .charts-grid {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 1.25rem;
      margin-bottom: 1.75rem;
    }}
    @media (max-width: 960px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      margin-bottom: 1.75rem;
    }}
    .chart-svg {{ width: 100%; height: auto; display: block; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ padding: 0.7rem 0.85rem; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    tr:last-child td {{ border-bottom: none; }}
    .sub-table th, .sub-table td {{ padding: 0.5rem 0.75rem; font-size: 0.85rem; }}
    .badge {{
      display: inline-block;
      padding: 0.18rem 0.55rem;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.03em;
    }}
    .badge-pass {{ background: var(--pass-bg); color: var(--pass-fg); border: 1px solid rgba(52, 211, 153, 0.3); }}
    .badge-fail {{ background: var(--fail-bg); color: var(--fail-fg); border: 1px solid rgba(251, 113, 133, 0.3); }}
    .model-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 1rem;
      overflow: hidden;
    }}
    .model-card summary {{
      padding: 0.95rem 1.25rem;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--surface-2);
      user-select: none;
      gap: 1rem;
    }}
    .model-card summary:hover {{ background: #263449; }}
    .summary-left, .summary-right {{ display: flex; align-items: center; gap: 0.75rem; }}
    .rank-pill {{
      background: #0f172a;
      border: 1px solid var(--border);
      padding: 0.15rem 0.5rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 700;
    }}
    .card-body {{ padding: 1.25rem; border-top: 1px solid var(--border); }}
    .prose-box {{
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.9rem 1rem;
      font-size: 0.88rem;
      white-space: pre-wrap;
      color: #cbd5e1;
    }}
    .code-box {{
      background: #070b14;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      max-height: 460px;
      overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.8rem;
      line-height: 1.45;
      color: #e2e8f0;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Local LLM Coding Benchmark Scorecard</h1>
        <div class="muted">Task: Persistent SQLite DAG Scheduler &amp; Worker Pool (One-Shot) &bull; Endpoint: <span class="mono">{html.escape(BASE_URL)}</span></div>
      </div>
      <div class="muted">Generated: {timestamp}</div>
    </header>

    <section class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Top Model</div>
        <div class="kpi-value">{best_score}</div>
        <div class="kpi-sub mono" title="{html.escape(best_model)}">{html.escape(best_model)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Peak Throughput</div>
        <div class="kpi-value">{fastest_str}</div>
        <div class="kpi-sub mono">{html.escape(fastest_name)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Models Passing (&ge;80)</div>
        <div class="kpi-value">{pass_count} / {len(summaries)}</div>
        <div class="kpi-sub">Temp: {TEMPERATURE} &bull; Max Tokens: {MAX_TOKENS:,} &bull; Request Timeout: {REQUEST_TIMEOUT:g}s</div>
        <div class="kpi-sub">Requested oMLX Thinking Budget: {THINKING_BUDGET:,} tokens</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Total Suite Time</div>
        <div class="kpi-value">{total_time:.1f}s</div>
        <div class="kpi-sub">Generation + Sandbox Execution</div>
      </div>
    </section>

    <section class="charts-grid">
      <div class="panel" style="margin-bottom:0;">
        <h2>Rubric Score Breakdown (100 Points Max)</h2>
        {bar_svg}
      </div>
      <div class="panel" style="margin-bottom:0;">
        <h2>Throughput vs. Code Quality</h2>
        {scatter_svg}
      </div>
    </section>

    <section class="panel">
      <h2>Leaderboard Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Model ID</th>
            <th>Score</th>
            <th>Speed</th>
            <th>Tokens</th>
            <th>Gen Time</th>
            <th>Test Time</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Detailed Model Scorecards &amp; Artifacts</h2>
      {''.join(detail_cards)}
    </section>
  </div>
</body>
</html>"""

    out_path.write_text(page_html, encoding="utf-8")


def run_scheduler_benchmark() -> None:
    if not MODELS_TO_TEST:
        print(
            "MODELS_TO_TEST is empty. Add at least one model ID at the top of the script.")
        sys.exit(1)

    try:
        subprocess.run(
            ["docker", "image", "inspect", RUNTIME_IMAGE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        diagnostic = getattr(exc, "stderr", None) or str(exc)
        sys.exit(
            f"Runtime sandbox requires a running Docker daemon and the {RUNTIME_IMAGE} image.\n"
            f"{diagnostic.strip()}\nLoad the image with: docker pull {RUNTIME_IMAGE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[ModelBenchmarkSummary] = []

    unload_error = None
    for index, model in enumerate(MODELS_TO_TEST):
        if index:
            previous = MODELS_TO_TEST[index - 1]
            try:
                unload_previous_model(previous)
            except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
                unload_error = f"Could not unload {previous} before testing {model}: {exc}"
                break
        summaries.append(run_benchmark_for_model(model))

    summaries.sort(key=lambda s: (s.score, s.tokens_per_sec), reverse=True)

    report_path = OUTPUT_DIR / "report.html"
    generate_html_report(summaries, report_path)

    print("\n" + "=" * 92)
    print(f"{'MODEL':<42} {'SCORE':>7} {'SPEED':>11} {'GEN TIME':>10} {'TEST TIME':>10} {'STATUS':>8}")
    print("-" * 92)
    for s in summaries:
        score_str = f"{s.score}/{s.max_score}"
        speed_str = f"{s.tokens_per_sec:.1f} tok/s"
        gen_str = f"{s.gen_seconds:.1f}s"
        exec_str = f"{s.exec_seconds:.2f}s" if s.exec_seconds > 0 else "-"
        print(
            f"{s.model:<42} {score_str:>7} {speed_str:>11} {gen_str:>10} {exec_str:>10} {s.status:>8}")
    print("=" * 92)
    print(f"HTML Dashboard Report generated at: {report_path.resolve()}\n")
    if unload_error:
        sys.exit(unload_error)


def main() -> None:
    from functools import partial

    from benchmark_suite import MAX_REPAIR_ATTEMPTS, run_suite
    from benchmark_tasks import TASKS

    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--tracks", nargs="+", choices=tuple(TASKS),
                         help="Run selected tracks (default: all tracks).")
    choices.add_argument("--scheduler", action="store_true",
                         help="Run the original SQLite scheduler benchmark.")
    parser.add_argument("--repair-attempts", type=int, choices=range(MAX_REPAIR_ATTEMPTS + 1),
                        help=f"Revisions for each failed suite track (default: {MAX_REPAIR_ATTEMPTS}). "
                             "Use 0 for the first pass only.")
    args = parser.parse_args()
    if args.scheduler:
        if args.repair_attempts is not None:
            parser.error("--repair-attempts applies to the coding suite.")
        run_scheduler_benchmark()
        return

    tracks = list(dict.fromkeys(args.tracks)) if args.tracks else list(TASKS)
    repair_attempts = MAX_REPAIR_ATTEMPTS if args.repair_attempts is None else args.repair_attempts
    try:
        report = run_suite(
            MODELS_TO_TEST, tracks, request_chat_completion, unload_previous_model, safe_slug,
            repair_chat=partial(request_chat_completion, repair=True), repair_attempts=repair_attempts,
        )
    except RuntimeError as exc:
        sys.exit(str(exc))
    print(f"\nHTML report: {report}")


if __name__ == "__main__":
    main()
