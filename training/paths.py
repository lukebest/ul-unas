"""Repo-relative path helpers so manifests stay portable (no /home/... prefixes)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repo_rel(path: str | Path, repo: Path | None = None) -> str:
    """Return a posix path relative to the repo root when possible."""
    repo = (repo or ROOT).resolve()
    p = Path(path)
    if not p.is_absolute():
        p = (repo / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(repo).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_audio(path: str | Path, repo: Path | None = None) -> Path:
    """Resolve a manifest path against cwd first, then the repo root."""
    p = Path(path)
    if p.exists():
        return p
    repo = repo or ROOT
    cand = repo / path
    if cand.exists():
        return cand
    return p
