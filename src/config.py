"""
src/config.py
─────────────
Loads config.yaml and exposes a single Config object used by all modules.

Why this exists:
  Instead of every file opening config.yaml itself, they all import from here.
  This means there is exactly one place to change if config.yaml moves or
  its format changes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# The project root is always the parent of the `src/` folder
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file, returning a plain Python dict."""
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. "
            "Make sure you are running commands from the edgechat/ project root."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.yaml must be a YAML mapping, got {type(data)}")
    return data


class Config:
    """
    Thin wrapper around the YAML config dict.

    Usage:
        from src.config import config
        model_name = config.embedding["model_name"]
        top_k = config.retrieval["top_k"]
    """

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self._data = _load_yaml(path)
        self._resolve_paths()
        self._apply_env_overrides()
        logger.debug("Config loaded from %s", path)

    # ── Sections (typed accessors) ────────────────────────────────────────

    @property
    def paths(self) -> dict[str, str]:
        return self._data["paths"]

    @property
    def embedding(self) -> dict[str, Any]:
        return self._data["embedding"]

    @property
    def chunking(self) -> dict[str, int]:
        return self._data["chunking"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self._data["retrieval"]

    @property
    def llm(self) -> dict[str, Any]:
        return self._data["llm"]

    @property
    def generation(self) -> dict[str, Any]:
        return self._data["generation"]

    @property
    def eval(self) -> dict[str, str]:
        return self._data["eval"]

    @property
    def offline(self) -> dict[str, Any]:
        return self._data.get("offline", {})

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_paths(self) -> None:
        """
        Make all path values absolute relative to the project root.
        This means the app works no matter which directory you run it from.
        """
        for key, value in self._data["paths"].items():
            abs_path = (PROJECT_ROOT / value).resolve()
            self._data["paths"][key] = str(abs_path)

    def _apply_env_overrides(self) -> None:
        """
        Allow environment variables to override config values.
        HF_HUB_OFFLINE=1 is the most important one for offline mode.
        """
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            self._data["offline"]["hf_hub_offline"] = True
            logger.info("Offline mode active (HF_HUB_OFFLINE=1)")

    def get_docs_dir(self) -> Path:
        """Return the docs directory as a Path object."""
        return Path(self._data["paths"]["docs_dir"])

    def get_index_dir(self) -> Path:
        """Return the index directory as a Path object."""
        return Path(self._data["paths"]["index_dir"])


# ── Module-level singleton ────────────────────────────────────────────────
# Import this in other modules: `from src.config import config`
config = Config()
