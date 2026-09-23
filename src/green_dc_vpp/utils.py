from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    """Return the repository root when called from the installed source tree."""

    return Path(__file__).resolve().parents[2]
