"""Runtime paths that work both from a checkout and from a Docker wheel."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    configured = os.getenv("API_DIAGNOSIS_PROJECT_ROOT")
    if configured:
        candidate = Path(configured).resolve()
        if candidate.exists():
            return candidate
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "data").exists() and (package_root / "specs").exists():
        return package_root
    return Path.cwd()


__all__ = ["project_root"]
