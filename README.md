# AI Eval Play

A Python runner for comparing models on one-shot coding and multi-turn
repository work. The six tracks cover a web issue tracker, a Python reservation
app, a Rust journal, a C++20 channel, a Go cache, and agentic maintenance.
Runner-owned tests grade each submission in network-disabled Docker containers.

## Setup

Use Python 3.10 or newer and a running Docker daemon. Load the images for the
tracks you plan to run.

```sh
docker pull python:3.12-slim
docker pull rust:1.85-bookworm
docker pull golang:1.25.14-bookworm
```

Select model IDs in `MODELS_TO_TEST` in `benchmark_runner.py`. Set
`OMNIROUTE_BASE_URL` to your OpenAI-compatible base URL, including `/v1`.
It defaults to `http://127.0.0.1:20128/v1`. Set `OMNIROUTE_API_KEY` if your
endpoint requires authentication. For the configured oMLX model prefix, set
`OMLX_SSH_HOST` to an SSH host or alias that can reach oMLX on loopback.
The runner unloads the previous oMLX model before switching.

You can also store these settings in the ignored `.benchmark.local.json`
file next to the scripts. Environment variables take precedence.

```json
{
  "base_url": "http://127.0.0.1:20128/v1",
  "api_key": "",
  "omlx_ssh_host": ""
}
```

## Run

```sh
python3 benchmark_runner.py
python3 benchmark_runner.py --tracks python-web go
python3 benchmark_runner.py --repair-attempts 0
python3 benchmark_runner.py --scheduler
```

The default suite gives failed tracks up to three repair attempts after the
first pass. Repairs omit the runner's output and thinking token caps. Your
endpoint can still apply its own limits.

Open `benchmark_results_suite/<run>/report.html` to compare first-pass and
final scores, per-check outcomes, and the time and tokens used by each
attempt. The original scheduler benchmark writes to `benchmark_results/`.
Generated reports, replies, and local credentials stay out of Git.
