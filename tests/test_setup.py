"""Tests for Stage 0: config loading and device selection."""
from pathlib import Path

from akkuwatch.config import load_config
from akkuwatch.device import get_device

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_loads_and_paths_are_absolute():
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg["classes"] == ["vape", "power_bank", "battery_pack", "tool_battery", "loose_cell"]
    assert all(p.is_absolute() for p in cfg["paths"].values())
    assert cfg["paths"]["raw_videos"] == REPO_ROOT / "data" / "raw_videos"


def test_device_auto_returns_known_device():
    assert get_device("auto") in {"cuda", "mps", "cpu"}


def test_device_can_be_forced():
    assert get_device("cpu") == "cpu"
