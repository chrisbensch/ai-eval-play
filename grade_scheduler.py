#!/usr/bin/env python3
"""Automated rubric grader for the one-shot SQLite DAG Scheduler benchmark."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CriterionResult:
    name: str
    passed: bool
    points: int
    max_points: int
    details: str


def extract_artifacts(raw_text: str) -> tuple[str, str]:
    """Extract architecture.md text and scheduler.py code block from model output."""
    code_blocks = re.findall(r"```python\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if not code_blocks:
        # Fallback if the input file is already raw Python code
        if "def " in raw_text and "sqlite3" in raw_text:
            return "", raw_text
        raise ValueError("No ```python ... ``` code block found in model response.")

    # Take the largest Python block as scheduler.py
    scheduler_code = max(code_blocks, key=len).strip()

    # Extract prose outside code blocks for architecture check
    prose = re.sub(r"```.*?```", "", raw_text, flags=re.DOTALL).strip()
    return prose, scheduler_code


def check_architecture_prose(prose: str) -> CriterionResult:
    """Check if architecture section exists and stays concise (<= 300 words buffer)."""
    if not prose:
        return CriterionResult("1. Architecture Section", False, 0, 10, "No architecture prose found (raw code input?).")
    words = prose.split()
    word_count = len(words)
    # Allow a 50-word grace buffer over the strict 250-word prompt limit
    passed = 30 <= word_count <= 320
    return CriterionResult(
        "1. Architecture Section",
        passed,
        10 if passed else 4,
        10,
        f"Prose word count: {word_count} words (target: <=250 words).",
    )


def check_static_requirements(code: str) -> list[CriterionResult]:
    """Inspect AST and source patterns for forbidden dependencies and required primitives."""
    results: list[CriterionResult] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [
            CriterionResult("2. Valid Python Syntax", False, 0, 15, f"SyntaxError at line {exc.lineno}: {exc.msg}")
        ]

    results.append(CriterionResult("2. Valid Python Syntax", True, 15, 15, "AST parsed cleanly."))

    # Check stdlib-only imports
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
            no_ext,
            10 if no_ext else 0,
            10,
            "All imports from Python stdlib." if no_ext else f"Forbidden imports: {sorted(set(third_party))}",
        )
    )

    # Check required architectural constructs
    code_lower = code.lower()
    checks = [
        ("CycleDetectedError", "cycledetectederror" in code_lower),
        ("SQLite WAL mode", "wal" in code_lower),
        ("Atomic lease / transaction", ("returning" in code_lower) or ("begin immediate" in code_lower)),
        ("Thread offloading (to_thread)", "to_thread" in code_lower),
        ("Type annotations", sum(isinstance(n, ast.arg) and n.annotation is not None for n in ast.walk(tree)) >= 5),
    ]
    passed_checks = [label for label, ok in checks if ok]
    failed_checks = [label for label, ok in checks if not ok]
    pts = len(passed_checks) * 5
    results.append(
        CriterionResult(
            "4. Structural & Concurrency Primitives",
            len(failed_checks) == 0,
            pts,
            25,
            f"Matched {len(passed_checks)}/5" + (f" | Missing: {', '.join(failed_checks)}" if failed_checks else ""),
        )
    )
    return results


def check_runtime_execution(code: str, timeout_sec: float = 45.0) -> CriterionResult:
    """Execute scheduler.py in an isolated temp directory and measure runtime & exit status."""
    with tempfile.TemporaryDirectory(prefix="bench_sched_") as tmpdir:
        script_path = Path(tmpdir) / "scheduler.py"
        script_path.write_text(code, encoding="utf-8")

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            elapsed = time.perf_counter() - start
        except subprocess.TimeoutExpired:
            return CriterionResult(
                "5. End-to-End Test Execution",
                False,
                0,
                40,
                f"Timed out after {timeout_sec:.0f}s (likely deadlock or infinite retry loop).",
            )

        if proc.returncode == 0:
            # Verify the script actually executed assertions/tests rather than exiting empty
            has_assertions = ("assert " in code) or ("unittest" in code)
            pts = 40 if has_assertions else 25
            note = f"Exited 0 in {elapsed:.2f}s."
            if not has_assertions:
                note += " (Warning: no assert/unittest statements detected in __main__)."
            return CriterionResult("5. End-to-End Test Execution", True, pts, 40, note)

        err_tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        return CriterionResult(
            "5. End-to-End Test Execution",
            False,
            0,
            40,
            f"Exit code {proc.returncode} after {elapsed:.2f}s:\n    " + "\n    ".join(err_tail),
        )


def grade(file_path: Path) -> int:
    raw = file_path.read_text(encoding="utf-8")
    prose, code = extract_artifacts(raw)

    results: list[CriterionResult] = [check_architecture_prose(prose)]
    static_results = check_static_requirements(code)
    results.extend(static_results)

    # Only run execution if syntax passed
    if any(r.name == "2. Valid Python Syntax" and r.passed for r in static_results):
        results.append(check_runtime_execution(code))

    total = sum(r.points for r in results)
    max_total = 100

    print(f"\n=== Benchmark Scorecard: {file_path.name} ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name:<38} {r.points:>2}/{r.max_points:<2} | {r.details}")
    print("-" * 78)
    print(f"TOTAL SCORE: {total}/{max_total}\n")
    return 0 if total >= 80 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 grade_scheduler.py <model_output.md_or_scheduler.py>")
        sys.exit(2)
    sys.exit(grade(Path(sys.argv[1])))
