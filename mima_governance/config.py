"""mima config — credential and workspace configuration stored at ~/.mima/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


_CONFIG_DIR = Path.home() / ".mima"
_CONFIG_FILE = _CONFIG_DIR / "config.json"


def _ensure_dir() -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load() -> dict:
    """Load config from ~/.mima/config.json. Returns empty dict if missing."""
    if not _CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(_CONFIG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save(config: dict) -> None:
    """Persist config to ~/.mima/config.json with owner-only (0o600) permissions."""
    import os
    _ensure_dir()
    _CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")
    # Restrict to owner read/write — API key must not be world-readable.
    os.chmod(_CONFIG_FILE, 0o600)


def get_api_key() -> Optional[str]:
    """Return stored API key, or None."""
    return load().get("api_key")


def get_workspace_id() -> Optional[str]:
    """Return stored workspace ID, or None."""
    return load().get("workspace_id")


def get_base_url() -> str:
    """Return stored base URL, defaulting to https://api.mima.ai."""
    return load().get("base_url", "https://api.mima.ai")


def set_credentials(api_key: str, workspace_id: str, base_url: str = "https://api.mima.ai") -> None:
    """Store credentials after successful login."""
    config = load()
    config["api_key"] = api_key
    config["workspace_id"] = workspace_id
    config["base_url"] = base_url
    save(config)
