"""Read local connection settings without storing credentials in source."""

import json
import os
from pathlib import Path


def connection_settings() -> tuple[str, str, str]:
    local_file = Path(__file__).with_name(".benchmark.local.json")
    try:
        local = json.loads(local_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        local = {}
    if not isinstance(local, dict) or any(not isinstance(value, str) for value in local.values()):
        raise ValueError(f"{local_file.name} must contain a JSON object with string values.")
    return (
        os.environ.get("OMNIROUTE_BASE_URL", local.get("base_url", "http://127.0.0.1:20128/v1")),
        os.environ.get("OMNIROUTE_API_KEY", local.get("api_key", "")),
        os.environ.get("OMLX_SSH_HOST", local.get("omlx_ssh_host", "")),
    )
