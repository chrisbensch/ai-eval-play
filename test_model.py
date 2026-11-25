#!/usr/bin/env python3
"""Quick connectivity and inference smoke test for a local OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

from benchmark_config import connection_settings


# ==============================================================================
# CONFIGURATION: Select the model to probe.
# ==============================================================================
BASE_URL, API_KEY, _ = connection_settings()
TEST_MODEL = "blm-omlx/occamy-1.0-abliterated-mtp-mlx-4bit"
# ==============================================================================


def request_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    base = BASE_URL.rstrip("/")

    # 1. Check /v1/models endpoint
    print(f"1. Checking available models at {base}/models ...")
    try:
        models_resp = request_json(f"{base}/models", timeout=10.0)
        available = [m.get("id", "<unknown>") for m in models_resp.get("data", [])]
        print(f"   [OK] Server reachable. Available models ({len(available)}):")
        for m_id in available:
            marker = " <-- target" if m_id == TEST_MODEL else ""
            print(f"     - {m_id}{marker}")
        if available and TEST_MODEL not in available:
            print(f"   [WARN] '{TEST_MODEL}' not in /models list (server may still auto-load it).")
    except Exception as exc:
        print(f"   [FAIL] Could not reach {base}/models: {exc}")
        sys.exit(1)

    # 2. Send a tiny 16-token completion request
    print(f"\n2. Testing chat completion with '{TEST_MODEL}' ...")
    payload = {
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Reply with the exact word: READY"}],
        "max_tokens": 16,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    try:
        chat_resp = request_json(f"{base}/chat/completions", payload=payload, timeout=120.0)
        elapsed = time.perf_counter() - start
        choice = chat_resp["choices"][0]["message"]
        reply = (choice.get("content") or choice.get("reasoning_content") or "").strip()
        print(f"   [OK] Response received in {elapsed:.2f}s: {reply!r}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"   [FAIL] HTTP {exc.code}: {body}")
        sys.exit(1)
    except Exception as exc:
        print(f"   [FAIL] Inference request failed: {exc}")
        sys.exit(1)

    print("\nAll checks passed! You are ready to run benchmark_runner.py.")


if __name__ == "__main__":
    main()
