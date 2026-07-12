"""Persisted user settings, stored at ./.tuffy/settings.json (gitignored,
same as .tuffy/mcp.json). Today this only holds the user's chosen default
model id, set via '/models default <id>', so it survives restarts without
editing code."""

import json
import os

SETTINGS_PATH = os.path.join(".tuffy", "settings.json")


def _load() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_default_model() -> str | None:
    """Returns the user's persisted default model id, or None if never set
    (first run) - caller falls back to the hardcoded DEFAULT_MODEL."""
    return _load().get("default_model")


def set_default_model(model_id: str) -> None:
    data = _load()
    data["default_model"] = model_id
    _save(data)
