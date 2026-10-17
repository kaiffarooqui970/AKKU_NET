"""Loading config.yaml.

All paths in the config are relative to the config file's own folder, so the
project works no matter which directory you run a script from.
"""
from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Read the YAML config and turn every entry under `paths` into an absolute Path."""
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base = config_path.parent
    cfg["paths"] = {name: (base / p).resolve() for name, p in cfg["paths"].items()}
    cfg["config_path"] = config_path
    return cfg
