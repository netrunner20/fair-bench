"""YAML config loading with dotted command-line overrides (``--set metrics.median_k=7``)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_configs(config_dir: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    d = Path(config_dir) if config_dir else CONFIG_DIR
    cfg = {name: load_yaml(d / f"{name}.yaml") for name in ("dataset", "generation", "models", "metrics")}
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def set_dotted(cfg: dict, key: str, value) -> None:
    cur = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def resolve(path: str | Path, base: Path = REPO_ROOT) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else base / p


def config_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def model_entries(models_cfg: dict) -> dict:
    """Model entries with shared defaults applied."""
    defaults = models_cfg.get("defaults", {})
    out = {}
    for name, entry in (models_cfg.get("models") or {}).items():
        e = copy.deepcopy(defaults)
        e.update(entry or {})
        out[name] = e
    return out
