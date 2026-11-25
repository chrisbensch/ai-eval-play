"""Execute and report on the coding benchmark suite."""

from __future__ import annotations

import base64
import difflib
import html
import json
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.client import HTTPException
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from benchmark_tasks import AGENT_EDITABLE, AGENT_FILES, TASKS, Task


REQUEST_ERRORS = (OSError, ValueError, HTTPException)
MAX_AGENT_TURNS = 20
MAX_REPAIR_ATTEMPTS = 3
MAX_FILE_BYTES = 48 * 1024
SUITE_OUTPUT = Path("benchmark_results_suite")


@dataclass(frozen=True)
class ModelReply:
    text: str
    output_tokens: int
    seconds: float
    finish_reason: str
    input_tokens: int | None = None
    estimated_output: bool = False


Chat = Callable[[str, list[dict[str, str]]], ModelReply | tuple[str, int, float, str]]


@dataclass
class Result:
    status: str = ""
    checks: list[tuple[str, bool]] = field(default_factory=list)
    tokens: int = 0
    seconds: float = 0.0
    test_seconds: float = 0.0
    diagnostics: str = ""
    elapsed_seconds: float | None = None
    input_tokens: int = 0
    requests: int = 0
    output_counts: int = 0
    input_counts: int = 0
    estimated_output: bool = False

    @property
    def score(self) -> int:
        return 20 * sum(passed for _, passed in self.checks)

    @property
    def needs_repair(self) -> bool:
        return self.status != "PASS" and "Sandbox unavailable" not in self.status


Submission = tuple[Result, Path]


def request_model(chat: Chat, model: str, messages: list[dict[str, str]], result: Result) -> ModelReply:
    result.requests += 1
    start = time.perf_counter()
    try:
        reply = chat(model, messages)
    except REQUEST_ERRORS:
        result.seconds += time.perf_counter() - start
        raise
    if not isinstance(reply, ModelReply):
        reply = ModelReply(*reply)
    result.seconds += reply.seconds
    result.tokens += reply.output_tokens
    result.output_counts += 1
    result.estimated_output |= reply.estimated_output
    if reply.input_tokens is not None:
        result.input_tokens += reply.input_tokens
        result.input_counts += 1
    return reply


@dataclass
class AgentWorkspace:
    files: dict[str, str] = field(default_factory=lambda: AGENT_FILES.copy())
    inspected: bool = False
    tested: bool = False


def grader_feedback(task: Task, result: Result) -> str:
    checks = dict(result.checks)
    outcomes = "\n".join(
        f"- {case}: {'not run' if case not in checks else 'passed' if checks[case] else 'failed'}"
        for case in task.cases
    )
    return (
        f"Submission grade: {result.score}/100 · {result.status}\n"
        f"{outcomes}\n\nGrader diagnostics:\n{result.diagnostics or '(none)'}"
    )


def require_images(images: list[str]) -> None:
    """Fail before calling a model if one of the selected test environments is missing."""
    try:
        engine = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        if not engine.stdout.strip():
            raise RuntimeError("Docker did not report its server version.")
        missing = []
        for image in sorted(set(images)):
            inspected = subprocess.run(
                ["docker", "image", "inspect", image],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )
            if inspected.returncode:
                missing.append(image)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"A running Docker daemon is required. {detail.strip()}") from exc
    if missing:
        pulls = "\n".join(f"docker pull {image}" for image in missing)
        raise RuntimeError(f"Load the selected benchmark images before running:\n{pulls}")


def shell_fixture(files: dict[str, str]) -> str:
    """Materialize only trusted paths; candidate contents are always passed as data."""
    commands = ["set -eu", "umask 077", "ulimit -f 16384"]
    for path, content in files.items():
        parent = shlex.quote(str(Path(path).parent))
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        commands.extend((
            f"mkdir -p {parent}",
            f"base64 -d > {shlex.quote(path)} <<'BENCHMARK_FILE_CONTENT'",
            encoded,
            "BENCHMARK_FILE_CONTENT",
        ))
    return "\n".join(commands) + "\n"


def isolated(
    image: str, script: str, seconds: int, scratch_size: str = "96m",
) -> subprocess.CompletedProcess[str]:
    container = f"coding-benchmark-{uuid.uuid4().hex}"
    command = [
        "docker", "run", "--rm", "--interactive", "--init", "--name", container,
        "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--user=65534:65534",
        "--pids-limit=64", "--memory=768m", "--cpus=1", "--log-driver=none",
        f"--tmpfs=/tmp:rw,nosuid,nodev,exec,size={scratch_size},mode=1777", "--workdir=/tmp",
        "--entrypoint=timeout", image, "--kill-after=2s", f"{seconds}s", "bash", "-s",
    ]
    try:
        return subprocess.run(
            command, input=script, capture_output=True, encoding="utf-8",
            errors="replace", timeout=seconds + 5,
        )
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            subprocess.run(
                ["docker", "rm", "--force", container],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        raise


def fenced_code(response: str, task: Task) -> str:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    aliases = "|".join(re.escape(lang) for lang in task.languages)
    matches = list(re.finditer(
        rf"^[ \t]*(?P<fence>`{{3,}}|~{{3,}})(?:{aliases})[ \t]*\r?\n"
        r"(?P<code>.*?)^[ \t]*(?P=fence)[ \t]*\r?$",
        response, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ))
    if len(matches) != 1 or not matches[0]["code"].strip():
        raise ValueError(f"Expected exactly one complete, nonempty {task.languages[0]} code block.")
    return matches[0]["code"]


def grade(task: Task, files: dict[str, str], result: Result) -> None:
    marker = f"BENCHMARK_{uuid.uuid4().hex}"
    body = shell_fixture({**files, task.grader_filename: task.grader})
    body += f"""
if timeout --kill-after=1s {task.build_seconds}s bash -c {shlex.quote(task.compile_command)} \
    </dev/null >compile.log 2>&1; then
    printf '%s\\n' '{marker} compiled'
else
    printf '%s\\n' '{marker} compile_failed'
    head -c 4500 compile.log
    printf '\\n'
    exit 0
fi
for case_name in {shlex.join(task.cases)}; do
    rm -f .case-passed
    if (export case_name; timeout --kill-after=1s 8s bash -c {shlex.quote(task.test_command)}) \
        </dev/null >case.log 2>&1 && test -f .case-passed; then
        printf '%s %s %s\\n' '{marker}' "$case_name" pass
    else
        code=$?
        printf '%s %s %s\\n' '{marker}' "$case_name" fail
        printf '  %s exited %s (124/137 indicates timeout or termination; 1 can also mean an absent pass marker).\\n' "$case_name" "$code"
        tail -c 1800 case.log | sed 's/^/  /'
        printf '\\n'
    fi
done
"""
    start = time.perf_counter()
    try:
        process = isolated(task.image, body, task.build_seconds + 60, task.scratch_size)
    except subprocess.TimeoutExpired:
        result.status = "Sandbox timed out"
        return
    except OSError as exc:
        result.status = "Sandbox unavailable"
        result.diagnostics = str(exc)
        return
    finally:
        result.test_seconds = time.perf_counter() - start

    records = []
    logs = [process.stderr.strip()] if process.stderr.strip() else []
    for line in process.stdout.splitlines():
        if line.startswith(marker + " "):
            records.append(line.removeprefix(marker + " ").split())
        elif line.strip():
            logs.append(line)
    result.diagnostics = "\n".join(logs)
    outcomes = {item[0]: item[1] for item in records if len(item) == 2}
    result.checks = [(name, outcomes[name] == "pass") for name in task.cases if name in outcomes]
    if ["compile_failed"] in records:
        result.status = "Build failed"
    elif process.returncode in (125, 126, 127) or not records:
        result.status = "Sandbox unavailable" if process.returncode in (125, 126, 127) else "Sandbox interrupted"
    elif len(result.checks) != len(task.cases) or process.returncode:
        result.status = "Sandbox interrupted"
    else:
        result.status = "PASS" if result.score == 100 else "Tests failed"


def run_one_shot(
    model: str, task: Task, directory: Path, chat: Chat, previous: Submission | None = None,
    *, latest: Result | None = None,
) -> Result:
    result = Result()
    messages = [
        {"role": "system", "content": "Solve the task in one response. Follow the requested format."},
        {"role": "user", "content": task.prompt},
    ]
    if previous:
        verdict, submitted_directory = previous
        response = submitted_directory / "response.md"
        if response.is_file():
            messages.append({"role": "assistant", "content": response.read_text(encoding="utf-8")})
        feedback = grader_feedback(task, verdict)
        if latest is not None and latest is not verdict:
            feedback += f"\n\nA later repair did not produce usable source. {latest.status}: {latest.diagnostics}"
        messages.append({
            "role": "user",
            "content": (
                "Revise your previous work using this grading feedback. Keep the parts that "
                "already work and fix the failures. If the previous reply was cut off or "
                "missing, complete the implementation from the task above. Return the full "
                "replacement in one fenced code block.\n\n" + feedback
            ),
        })
    saved_prompt = task.prompt if previous is None else "\n\n".join(
        f"{message['role'].title()}:\n{message['content']}" for message in messages
    )
    directory.joinpath("prompt.md").write_text(saved_prompt, encoding="utf-8")
    try:
        response = request_model(chat, model, messages, result)
    except REQUEST_ERRORS as exc:
        result.status = "API error"
        result.diagnostics = str(exc)
        return result
    reply, finish = response.text, response.finish_reason
    directory.joinpath("response.md").write_text(reply, encoding="utf-8")
    try:
        code = fenced_code(reply, task)
    except ValueError as exc:
        result.status = (
            "Token limit" if finish == "length" else
            "Unexpected tool call" if finish == "tool_calls" else "Invalid submission"
        )
        result.diagnostics = f"{exc} finish_reason={finish}"
        return result
    directory.joinpath(task.filename).write_text(code, encoding="utf-8")
    grade(task, {task.filename: code}, result)
    result.diagnostics = f"finish_reason={finish}\n{result.diagnostics}".strip()
    return result


AGENT_PROTOCOL = f"""You are working in a small virtual repository through the following tools.
Return exactly one JSON object per turn and wait for the tool result. Do not use
Markdown, offer a plan in prose, or emit multiple actions. Available actions:
{{"action":"list"}}
{{"action":"read","path":"relative/path"}}
{{"action":"write","path":"relative/path","content":"entire replacement file"}}
{{"action":"test"}}
{{"action":"finish","summary":"brief summary of the patch"}}
The test action runs the current public tests in a fresh environment. You have
{MAX_AGENT_TURNS} turns total. Inspect the repo promptly, edit source, and run the public tests
after your last edit. The runner grades the final patch using separate tests.
"""


def public_tests(task: Task, files: dict[str, str]) -> str:
    script = shell_fixture(files) + """
if timeout --kill-after=1s 8s python -B -m unittest discover -s tests -v </dev/null >public.log 2>&1 \
    && grep -qx OK public.log; then
    printf 'Public tests completed successfully.\\n'
else
    printf 'Public test run failed.\\n'
fi
tail -c 2500 public.log
"""
    try:
        process = isolated(task.image, script, 14)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Could not run the public tests: {exc}"
    if process.returncode:
        return f"Public test sandbox exited {process.returncode}: {process.stderr[-800:]}"
    return process.stdout


def decode_action(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    action = json.loads(match[1] if match else text)
    if not isinstance(action, dict):
        raise ValueError("Expected a JSON object.")
    return action


def run_agent(
    model: str, task: Task, directory: Path, chat: Chat, *,
    workspace: AgentWorkspace | None = None, previous: Result | None = None,
) -> Result:
    result = Result()
    if workspace is None:
        workspace = AgentWorkspace()
    files = workspace.files
    prompt = task.prompt
    if previous:
        prompt += (
            "\n\nContinue in the repository from your last graded attempt. Your edits are "
            "already present. Inspect and revise the current files using the tool protocol. "
            "Use the new turn budget to run public tests after any further edits, then finish.\n\n"
            + grader_feedback(task, previous)
        )
    directory.joinpath("prompt.md").write_text(
        AGENT_PROTOCOL + "\n" + prompt, encoding="utf-8",
    )
    messages = [{"role": "system", "content": AGENT_PROTOCOL},
                {"role": "user", "content": prompt}]
    transcript = []
    finished = False
    failure = ""
    for turn in range(1, MAX_AGENT_TURNS + 1):
        print(f"      Agent turn {turn}/{MAX_AGENT_TURNS}", flush=True)
        try:
            response = request_model(chat, model, messages, result)
        except REQUEST_ERRORS as exc:
            failure = f"API error: {exc}"
            break
        reply, finish_reason = response.text, response.finish_reason
        messages.append({"role": "assistant", "content": reply})
        try:
            action = decode_action(reply)
            name = action.get("action")
            path = action.get("path")
            if name == "list":
                feedback = "\n".join(sorted(files))
            elif name == "read" and isinstance(path, str) and path in files:
                workspace.inspected = True
                feedback = files[path] or "(empty file)"
            elif name == "write" and isinstance(path, str) and path in AGENT_EDITABLE:
                replacement = action.get("content")
                if not isinstance(replacement, str) or len(replacement.encode("utf-8")) > MAX_FILE_BYTES:
                    raise ValueError("Write requires a text content field of at most 48 KiB.")
                files[path] = replacement
                workspace.tested = False
                feedback = f"Updated {path}."
            elif name == "test":
                feedback = public_tests(task, files)
                workspace.tested = feedback.startswith(("Public tests completed", "Public test run failed"))
            elif name == "finish":
                changed = any(files[path] != AGENT_FILES[path] for path in AGENT_EDITABLE)
                if workspace.inspected and changed and workspace.tested:
                    finished = True
                    feedback = "Patch submitted."
                else:
                    feedback = "Read a repository file, make a source edit, then run public tests before finish."
            else:
                raise ValueError("Unknown action, missing file, or path outside the editable repository.")
        except (ValueError, TypeError) as exc:
            feedback = f"Invalid tool request ({finish_reason}): {exc}"
        transcript.append({
            "turn": turn, "response": reply, "tool_result": feedback, "finish_reason": finish_reason,
            "model_seconds": response.seconds, "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens, "estimated_output": response.estimated_output,
        })
        if finished:
            break
        messages.append({"role": "user", "content": f"Tool result:\n{feedback}"})

    directory.joinpath("transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    patch = []
    for path in sorted(AGENT_EDITABLE):
        destination = directory / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(files[path], encoding="utf-8")
        patch.extend(difflib.unified_diff(
            AGENT_FILES[path].splitlines(keepends=True), files[path].splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}",
        ))
    directory.joinpath("submission.diff").write_text("".join(patch), encoding="utf-8")
    grade(task, files, result)
    if not finished:
        failure = failure or "Turn limit reached before finish."
        result.status = f"Incomplete workflow · {result.status}"
    result.diagnostics = "\n".join(part for part in (failure, result.diagnostics) if part)
    return result


REPORT_STYLE = """
:root{color-scheme:dark;--bg:#0b1120;--panel:#121d30;--raised:#19263b;--line:#2b3b53;
  --text:#f1f5f9;--muted:#a7b7ce;--mono:"SF Mono",Consolas,"Liberation Mono",monospace}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(ellipse at top,#172b46 0%,var(--bg) 43%);
  color:var(--text);font:16px/1.55 "Avenir Next",Avenir,"Segoe UI",sans-serif}
main{width:min(1200px,calc(100% - 3rem));margin:3rem auto}
h1,h2,h3,p{margin:0}
h1{font-size:clamp(2rem,5vw,3.4rem);line-height:1.04;letter-spacing:-.055em}
h2{font-size:1.2rem;letter-spacing:-.025em;margin-bottom:1rem}
h3{font-size:1rem}
a{color:inherit;text-underline-offset:.15em}
:focus-visible{outline:2px solid #7dd3fc;outline-offset:4px}
.header{border-bottom:1px solid var(--line);padding:0 0 1.4rem}
.header p,.note{color:var(--muted);font-size:.9rem}
.header p{margin-top:.65rem}
.error{padding:.85rem 1rem;border:1px solid #924b56;background:#472436;border-radius:.55rem}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:.8rem;margin:1.25rem 0}
.metric,.panel{background:var(--panel);border:1px solid var(--line);border-radius:.8rem}
.metric{padding:.85rem 1rem;border-top:3px solid var(--accent,#7dd3fc)}
.metric-label,.eyebrow{font:600 .72rem/1.4 var(--mono);color:var(--muted);
  text-transform:uppercase;letter-spacing:.07em}
.metric-value{display:block;font-size:1.63rem;font-weight:750;line-height:1.15;margin:.35rem 0;
  color:var(--accent,#7dd3fc);font-variant-numeric:tabular-nums}
.panel{padding:1.15rem;margin:1rem 0}
.web{--accent:#38bdf8}.rust{--accent:#fbbf24}.cpp{--accent:#c4b5fd}.agent{--accent:#5eead4}
.python-web{--accent:#fda4af}.go{--accent:#bef264}
.legend{display:flex;gap:.4rem 1.15rem;flex-wrap:wrap;font-size:.86rem;color:var(--muted);margin-bottom:1rem}
.legend span::before{content:"";display:inline-block;width:.68em;height:.68em;
  border-radius:2px;background:var(--accent);margin-right:.4em}
.chart{display:grid;gap:.85rem}.bar-row{display:grid;grid-template-columns:minmax(9rem,1.2fr) 3fr 8rem;
  gap:.6rem;align-items:center}
.name{font:500 .8rem/1.45 var(--mono);overflow-wrap:anywhere}
.bar{display:flex;gap:3px;height:1.15rem;border-radius:4px;overflow:hidden}
.lane{flex:1;min-width:0;background:#24344c}.lane span{display:block;height:100%;background:var(--accent)}
.lane.pending{background:repeating-linear-gradient(135deg,#25354c 0 4px,#1a2639 4px 8px)}
.points{font:650 .83rem/1.4 var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.bar-row .points{justify-self:end}
.comparison{display:grid;gap:1rem;margin-top:1rem}
.comparison-row{display:grid;grid-template-columns:minmax(9rem,1.2fr) 3fr 7rem;
  align-items:center;gap:.55rem 1rem}
.comparison-bars{display:grid;gap:.35rem}
.comparison-bar{display:grid;grid-template-columns:6rem 1fr 4.5rem;align-items:center;gap:.5rem}
.comparison-bar>span:first-child{font-size:.76rem;color:var(--muted)}
.comparison-rail{height:.65rem;border-radius:.18rem;background:#24344c;overflow:hidden}
.comparison-rail i{display:block;height:100%;background:#60a5fa}
.comparison-final .comparison-rail i{background:#5eead4}
.gain{justify-self:end;white-space:nowrap;color:#bef264;background:#263920;border-radius:.25rem;
  padding:.25rem .5rem;font:650 .8rem/1.3 var(--mono)}
.gain.unchanged{color:var(--muted);background:var(--raised)}
.table-scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;text-align:left;min-width:550px}
th,td{border-bottom:1px solid var(--line);padding:.65rem .6rem;vertical-align:top}
thead th{font-size:.82rem;color:var(--accent,var(--muted));white-space:nowrap}
tbody tr:hover{background:var(--raised)}
tbody tr:last-child>*{border-bottom:0}
.score-link{display:block;width:max-content;text-decoration:none}
.score-link:hover,.score-link:focus-visible{text-decoration:underline}
.badge{display:inline-block;border-radius:.28rem;padding:.13em .52em;font:650 .69rem/1.5 var(--mono)}
.pass{color:#86efac;background:#204238}.fail{color:#fecaca;background:#472837}
.partial{color:#fde68a;background:#493c23}.pending-text{color:var(--muted)}
.cards{display:grid;gap:.5rem}.model{margin-bottom:1.3rem}.model:last-child{margin-bottom:0}
.model h3{margin:0 0 .45rem;color:#d8e2f1}
details{background:var(--raised);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:.45rem;scroll-margin-top:1rem}
summary{padding:.6rem .85rem;cursor:pointer}.summary-content{display:inline-flex;gap:.45rem .75rem;
  flex-wrap:wrap;align-items:baseline;margin-left:.35rem}.track-name{color:var(--accent);font-weight:700}
.detail-body{border-top:1px solid var(--line);padding:.75rem .9rem}
.revision{margin-top:.8rem}.revision summary{font-size:.87rem}
.detail-meta{display:flex;gap:.35rem 1rem;flex-wrap:wrap;color:var(--muted);font-size:.85rem;margin-bottom:.5rem}
.detail-meta strong{color:var(--text);font-variant-numeric:tabular-nums}
.usage-line{display:flex;gap:.25rem .65rem;flex-wrap:wrap;align-items:baseline;
  font-size:.78rem;color:var(--muted)}
.usage-line strong{color:var(--text);font-variant-numeric:tabular-nums}
.detail-body>.usage-line{margin-bottom:.75rem}
.model>.usage-line{margin-bottom:.55rem}
.checks{list-style:none;padding:0;margin:.5rem 0 .8rem;display:grid;
  grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.25rem .9rem}
.checks li{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;font-size:.83rem}
.links{display:flex;gap:.45rem .85rem;flex-wrap:wrap;font-size:.84rem;color:#bae6fd}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:24rem;overflow:auto;background:#0c1525;
  border:1px solid var(--line);border-radius:.35rem;padding:.65rem;margin:.65rem 0 0;
  color:#dbe7f7;font:normal .78rem/1.5 var(--mono)}
@media(max-width:640px){
  main{width:calc(100% - 1.4rem);margin:1.5rem auto}
  .panel{padding:.85rem}.bar-row{grid-template-columns:1fr auto;gap:.3rem .6rem}
  .bar{grid-column:1/-1;grid-row:2}.bar-row .points{grid-column:2;grid-row:1}
  .comparison-row{grid-template-columns:minmax(0,1fr) auto;gap:.3rem .5rem}
  .comparison-bars{grid-column:1/-1;grid-row:2}
  .comparison-bar{grid-template-columns:5.4rem minmax(0,1fr) 4.3rem}
  .gain{grid-column:2;grid-row:1}
}
"""


def report_badge(status: str) -> str:
    tone = "pass" if status == "PASS" else "partial" if status.startswith("Incomplete") else "fail"
    return f'<span class="badge {tone}">{html.escape(status)}</span>'


def token_total(amount: int, counts: int, requests: int, *, estimated: bool = False) -> str:
    if not counts and not amount:
        return "—"
    text = f"{'≈' if estimated else ''}{amount:,}"
    return text + (" (partial)" if counts < requests else "")


def usage_numbers(results: list[Result]) -> tuple[str, str, str]:
    elapsed = [r.elapsed_seconds for r in results if r.elapsed_seconds is not None]
    wall = f"{sum(elapsed):,.1f}s" if elapsed else "—"
    if elapsed and len(elapsed) < len(results):
        wall += " (partial)"
    requests = sum(r.requests for r in results)
    output = token_total(
        sum(r.tokens for r in results), sum(r.output_counts for r in results), requests,
        estimated=any(r.estimated_output for r in results),
    )
    inputs = token_total(
        sum(r.input_tokens for r in results), sum(r.input_counts for r in results), requests,
    )
    return wall, output, inputs


def usage_line(label: str, results: list[Result]) -> str:
    wall, output, inputs = usage_numbers(results)
    return (
        f'<p class="usage-line"><strong>{label}</strong>'
        f'<span><strong>{wall}</strong> elapsed</span>'
        f'<span><strong>{output}</strong> output tokens</span>'
        f'<span><strong>{inputs}</strong> input tokens</span></p>'
    )


def attempt_details(report_dir: Path, track: str, submission: Submission) -> str:
    task = TASKS[track]
    result, directory = submission
    outcomes = dict(result.checks)
    checks = []
    for case in task.cases:
        outcome = outcomes.get(case)
        badge = ('<span class="pending-text">Not run</span>' if outcome is None
                 else report_badge("PASS" if outcome else "FAIL"))
        label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", case.removeprefix("Test"))
        label = label.removeprefix("test_").replace("_", " ")
        label = html.escape(label[:1].upper() + label[1:])
        checks.append(f"<li><span>{label}</span>{badge}</li>")
    artifacts = [("prompt.md", "Prompt"), ("metrics.json", "Metrics")]
    if track == "agent":
        artifacts += [("transcript.json", "Transcript"), ("submission.diff", "Patch")]
    else:
        artifacts += [("response.md", "Response"), (task.filename, "Submitted code")]
    links = []
    for file, label in artifacts:
        saved = directory / file
        if saved.is_file():
            href = html.escape(quote(saved.relative_to(report_dir).as_posix()))
            links.append(f'<a href="{href}">{label}</a>')
    diagnostic = f'<pre tabindex="0">{html.escape(result.diagnostics)}</pre>' if result.diagnostics else ""
    elapsed, output, inputs = usage_numbers([result])
    return f"""
    <div class="detail-meta">
      <span><strong>{elapsed}</strong> elapsed</span>
      <span><strong>{output}</strong> output tokens</span>
      <span><strong>{inputs}</strong> input tokens</span>
      <span><strong>{result.seconds:.1f}s</strong> model requests</span>
      <span><strong>{result.test_seconds:.1f}s</strong> grading in {html.escape(task.image)}</span>
    </div>
    <ul class="checks">{''.join(checks)}</ul>
    <div class="links">{''.join(links)}</div>
    {diagnostic}
"""


def write_report(
    path: Path, tracks: list[str], runs: list[tuple[str, dict[str, Submission]]],
    error: str = "", *,
    repair_history: dict[tuple[int, str], list[Submission]] | None = None,
    repair_limit: int = 0,
) -> None:
    repair_history = repair_history or {}
    repair_count = sum(map(len, repair_history.values()))
    headings = "".join(f'<th scope="col" class="{key}">{html.escape(TASKS[key].title)}</th>' for key in tracks)
    legend = "".join(f'<span class="{key}">{html.escape(TASKS[key].title)}</span>' for key in tracks)
    limit = 100 * len(tracks)
    ranked = sorted(
        enumerate(runs),
        key=lambda row: (len(row[1][1]) == len(tracks), sum(r.score for r, _ in row[1][1].values())),
        reverse=True,
    )
    scored = [(key, result) for _, entries in runs for key, (result, _) in entries.items()]
    passed = sum(ok for _, result in scored for _, ok in result.checks)
    expected = sum(len(TASKS[key].cases) for key, _ in scored)
    repair_seconds = sum(r.seconds for attempts in repair_history.values() for r, _ in attempts)
    metrics = [("First-pass checks passed", f"{passed}/{expected}", "cpp")]
    if ranked and len(ranked[0][1][1]) == len(tracks):
        best = sum(r.score for r, _ in ranked[0][1][1].values())
        metrics.insert(0, ("Highest first-pass score", f"{best}/{limit}", "web"))
    if "agent" in tracks:
        agents = [r for key, r in scored if key == "agent"]
        finished = sum(not r.status.startswith("Incomplete workflow") for r in agents)
        metrics.append(("First-pass agent finishes", f"{finished}/{len(agents)}", "agent"))
    metrics.append(("First-pass model time", f"{sum(r.seconds for _, r in scored):,.1f}s", "rust"))
    if repair_limit:
        metrics.append(("Repair attempts", str(repair_count), "agent"))
        metrics.append(("Repair model time", f"{repair_seconds:,.1f}s", "go"))
    cards = "".join(
        f'<div class="metric {color}"><span class="metric-label">{label}</span>'
        f'<strong class="metric-value">{value}</strong></div>'
        for label, value, color in metrics
    )

    rows, chart, comparison, groups = [], [], [], []
    for index, (model, results) in ranked:
        model_html = html.escape(model)
        total = sum(result.score for result, _ in results.values())
        best_total = total
        repairs_complete = True
        model_attempts = []
        tracks_passed = 0
        cells, lanes, details = [], [], []
        for track in tracks:
            if track not in results:
                repairs_complete = False
                cells.append('<td class="pending-text">Pending</td>')
                lanes.append('<div class="lane pending"></div>')
                continue
            submission = results[track]
            result, _ = submission
            task = TASKS[track]
            target = f"model-{index}-{track}"
            title = html.escape(task.title)
            badge = report_badge(result.status)
            revisions = repair_history.get((index, track), [])
            latest = revisions[-1][0] if revisions else result
            repair_pending = latest.needs_repair and len(revisions) < repair_limit
            if repair_pending:
                repairs_complete = False
            track_attempts = [result]
            for verdict, _ in revisions:
                if track_attempts[-1].status == "PASS":
                    break
                track_attempts.append(verdict)
            track_passed = track_attempts[-1].status == "PASS"
            tracks_passed += track_passed
            model_attempts.extend(track_attempts)
            total_usage = usage_line(
                "To pass" if track_passed else "Spent so far" if repair_pending else "Spent · no pass",
                track_attempts,
            )
            revision_note = ""
            revision_cards = []
            if revisions:
                best = max(
                    [submission, *revisions],
                    key=lambda item: (item[0].score, item[0].status == "PASS"),
                )[0]
                best_total += best.score - result.score
                revision_note = (
                    f'<div class="note">Best: {best.score}/100 · '
                    f'{len(revisions)}/{repair_limit} repairs</div>'
                )
                if best.status != result.status:
                    revision_note += report_badge(best.status)
            for number, revision in enumerate(revisions, start=1):
                verdict, _ = revision
                wall, output, _ = usage_numbers([verdict])
                revision_cards.append(
                    f'<details class="revision {track}"><summary>Repair {number} · '
                    f'{verdict.score}/100 {report_badge(verdict.status)} '
                    f'<span class="note">· {wall} elapsed · {output} output tokens</span></summary>'
                    f'<div class="detail-body">{attempt_details(path.parent, track, revision)}</div></details>'
                )
            lanes.append(f'<div class="lane {track}"><span style="width:{result.score}%"></span></div>')
            cells.append(
                f'<td><a class="score-link points" href="#{target}" aria-label="{title}: '
                f'{result.score} out of 100 on first pass, view attempts">{result.score}/100</a>'
                f'{badge}{revision_note}</td>'
            )
            details.append(f"""
<details id="{target}" class="{track}">
  <summary><span class="summary-content"><span class="track-name">{title}</span>
    <span class="points">{result.score}/100</span>{badge}</span></summary>
  <div class="detail-body">
    {total_usage}
    <p class="eyebrow">First pass</p>
    {attempt_details(path.parent, track, submission)}
    {''.join(revision_cards)}
  </div>
</details>""")
        total_display = f"{total}/{limit}" if len(results) == len(tracks) else f"{total}/{limit} so far"
        best_cell = f'<td class="points">{best_total}/{limit}</td>' if repair_limit else ""
        rows.append(
            f'<tr><th scope="row" class="name">{model_html}</th>'
            f'<td class="points">{total_display}</td>{best_cell}{"".join(cells)}</tr>'
        )
        chart.append(
            f'<div class="bar-row"><span class="name">{model_html}</span>'
            f'<div class="bar" aria-hidden="true">{"".join(lanes)}</div>'
            f'<span class="points">{total_display}</span></div>'
        )
        if tracks_passed == len(tracks):
            model_usage_label = "To pass all tracks"
        else:
            spent = "Spent" if repairs_complete else "Spent so far"
            model_usage_label = f"{spent} · {tracks_passed}/{len(tracks)} tracks passed"
        model_usage = usage_line(model_usage_label, model_attempts)
        if repair_limit and len(results) == len(tracks):
            gain = best_total - total
            final_label = "Final" if repairs_complete else "Best so far"
            gain_label = f"+{gain} pts"
            comparison.append(f"""
<div class="comparison-row">
  <span class="name">{model_html}</span>
  <div class="comparison-bars">
    <div class="comparison-bar">
      <span>First pass</span><span class="comparison-rail" aria-hidden="true">
        <i style="width:{total / limit:.2%}"></i></span>
      <span class="points">{total}/{limit}</span>
    </div>
    <div class="comparison-bar comparison-final">
      <span>{final_label}</span><span class="comparison-rail" aria-hidden="true">
        <i style="width:{best_total / limit:.2%}"></i></span>
      <span class="points">{best_total}/{limit}</span>
    </div>
    {model_usage}
  </div>
  <span class="gain{' unchanged' if not gain else ''}">{gain_label}</span>
</div>""")
        groups.append(f'<section class="model"><h3 class="name">{model_html}</h3>{model_usage}'
                      f'<div class="cards">{"".join(details)}</div></section>')
    notice = f'<p class="error" role="alert">{html.escape(error)}</p>' if error else ""
    notes = [
        "Each track has five 20-point checks.",
        "Elapsed time covers each attempt, including tests. Totals exclude time between attempts.",
        "≈ marks estimated output tokens. Partial counts omit requests that returned no usage.",
    ]
    if "web" in tracks:
        notes.append("Web grading executes HTTP requests and inspects the HTML and JavaScript source.")
    if "python-web" in tracks:
        notes.append("The Python reservation app also exercises server-rendered forms over HTTP.")
    if "agent" in tracks:
        notes.append(f"The agentic track allows up to {MAX_AGENT_TURNS} turns and reports workflow completion separately.")
    best_heading = '<th scope="col">Best with repairs</th>' if repair_limit else ""
    if repair_limit:
        notes.append(f"Failed tracks get up to {repair_limit} repairs after the first pass. "
                     "The leaderboard ranks first-pass scores. Best with repairs sums each track’s best attempt. "
                     "Repairs use the model server’s token limits.")
    comparison_panel = ""
    if repair_limit:
        comparison_body = (
            f'<div class="comparison">{"".join(comparison)}</div>' if comparison else
            '<p class="note">Models appear here after completing their first pass.</p>'
        )
        comparison_panel = f"""
<section class="panel" aria-labelledby="comparison-title">
  <h2 id="comparison-title">First pass vs. final grade</h2>
  <p class="note">The final grade keeps the best score for each track. The gain is the difference from the first pass.</p>
  {comparison_body}
</section>"""
    chart_body = f'<div class="chart">{"".join(chart)}</div>' if chart else '<p class="note">No tracks have been graded.</p>'
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Coding benchmark scorecard</title>
<style>{REPORT_STYLE}</style>
</head>
<body>
<main>
<header class="header">
  <h1>Coding benchmark scorecard</h1>
  <p>{' '.join(notes)}</p>
</header>
<section class="metrics" aria-label="Suite totals">{cards}</section>
{notice}
{comparison_panel}
<section class="panel" aria-labelledby="breakdown-title">
  <h2 id="breakdown-title">First-pass score breakdown</h2>
  <div class="legend">{legend}</div>
  {chart_body}
</section>
<section class="panel" aria-labelledby="leaderboard-title">
  <h2 id="leaderboard-title">Leaderboard</h2>
  <div class="table-scroll" role="region" aria-label="Model scores by track" tabindex="0">
    <table><thead><tr><th scope="col">Model</th><th scope="col">First pass</th>{best_heading}{headings}</tr></thead>
    <tbody>{''.join(rows)}</tbody></table>
  </div>
</section>
<section class="panel" aria-labelledby="details-title">
  <h2 id="details-title">Checks and artifacts</h2>
  {''.join(groups)}
</section>
</main>
<script>
function openLinkedTrack() {{
  const track = document.getElementById(location.hash.slice(1));
  if (track instanceof HTMLDetailsElement) {{
    track.open = true;
    track.scrollIntoView({{block: "start"}});
  }}
}}
addEventListener("hashchange", openLinkedTrack);
openLinkedTrack();
</script>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def save_attempt_metrics(directory: Path, result: Result, started: float) -> None:
    result.elapsed_seconds = time.perf_counter() - started
    metrics = {
        "status": result.status,
        "score": result.score,
        "elapsed_seconds": result.elapsed_seconds,
        "model_seconds": result.seconds,
        "grading_seconds": result.test_seconds,
        "requests": result.requests,
        "output_tokens": result.tokens if result.output_counts else None,
        "output_counts": result.output_counts,
        "estimated_output": result.estimated_output,
        "input_tokens": result.input_tokens if result.input_counts else None,
        "input_counts": result.input_counts,
    }
    directory.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def run_suite(
    models: list[str], tracks: list[str], chat: Chat, unload_model: Callable[[str], None],
    slug: Callable[[str], str], *, repair_chat: Chat | None = None, repair_attempts: int = 0,
) -> Path:
    if not models:
        raise RuntimeError("MODELS_TO_TEST is empty.")
    if not 0 <= repair_attempts <= MAX_REPAIR_ATTEMPTS:
        raise ValueError(f"Repair attempts must be between 0 and {MAX_REPAIR_ATTEMPTS}.")
    if repair_attempts and repair_chat is None:
        raise ValueError("Repair attempts require a repair chat callback.")
    require_images([TASKS[name].image for name in tracks])
    output = SUITE_OUTPUT / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    report = output / "report.html"
    runs: list[tuple[str, dict[str, Submission]]] = []
    repairs: dict[tuple[int, str], list[Submission]] = {}
    agent_workspaces: dict[int, AgentWorkspace] = {}
    active_model = None

    def save(error: str = "") -> None:
        write_report(report, tracks, runs, error, repair_history=repairs, repair_limit=repair_attempts)

    def switch_model(model: str) -> None:
        nonlocal active_model
        if active_model is not None and model != active_model:
            try:
                unload_model(active_model)
            except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
                error = f"Could not unload {active_model}: {exc}. Partial report: {report.resolve()}"
                save(error)
                raise RuntimeError(error) from exc
        active_model = model

    for index, model in enumerate(models):
        switch_model(model)
        results: dict[str, Submission] = {}
        runs.append((model, results))
        for track in tracks:
            task = TASKS[track]
            directory = output / f"{index + 1:02d}-{slug(model)}" / track
            directory.mkdir(parents=True)
            print(f"\n>>> {model} · {task.title}", flush=True)
            started = time.perf_counter()
            if track == "agent":
                workspace = AgentWorkspace()
                result = run_agent(model, task, directory, chat, workspace=workspace)
                if repair_attempts and result.needs_repair:
                    agent_workspaces[index] = workspace
            else:
                result = run_one_shot(model, task, directory, chat)
            save_attempt_metrics(directory, result, started)
            results[track] = (result, directory)
            print(f"    {result.score}/100 · {result.status}", flush=True)
            save()

    if not repair_attempts or repair_chat is None:
        return report.resolve()

    for index, (model, results) in enumerate(runs):
        for track in tracks:
            result, baseline_dir = results[track]
            if not result.needs_repair:
                continue
            switch_model(model)
            task = TASKS[track]
            attempts = repairs[(index, track)] = []
            source_submission = results[track]
            for number in range(1, repair_attempts + 1):
                directory = baseline_dir / f"repair-{number}"
                directory.mkdir()
                print(f"\n>>> {model} · {task.title} · Repair {number}/{repair_attempts}", flush=True)
                started = time.perf_counter()
                if track == "agent":
                    result = run_agent(
                        model, task, directory, repair_chat,
                        workspace=agent_workspaces[index], previous=result,
                    )
                else:
                    result = run_one_shot(
                        model, task, directory, repair_chat, source_submission, latest=result,
                    )
                    _, source_dir = source_submission
                    response = directory / "response.md"
                    if (directory / task.filename).is_file():
                        source_submission = (result, directory)
                    elif not (source_dir / task.filename).is_file() and response.is_file():
                        text = response.read_text(encoding="utf-8")
                        if text.strip() and (result.status == "Token limit" or
                                             not (source_dir / "response.md").is_file()):
                            source_submission = (result, directory)
                save_attempt_metrics(directory, result, started)
                attempts.append((result, directory))
                print(f"    {result.score}/100 · {result.status}", flush=True)
                save()
                if not result.needs_repair:
                    break
    return report.resolve()
