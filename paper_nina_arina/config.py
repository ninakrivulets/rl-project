import copy
import json
from pathlib import Path


def _parse_value(value):
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_nested(config, key, value):
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(config, overrides):
    resolved = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must look like key=value, got {override}")
        key, value = override.split("=", 1)
        _set_nested(resolved, key, _parse_value(value))
    return resolved


def load_config(path, overrides=None):
    path = Path(path)
    with path.open() as f:
        config = json.load(f)
    return apply_overrides(config, overrides)

