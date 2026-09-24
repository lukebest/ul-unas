"""Resolve the continue-train init checkpoint for domain-expand fine-tunes.

Priority (verified on main@5f1acaf):
1. `training/outputs/finetune_vbdemand/ulunas_finetuned.pt` (PR #1 ship)
2. `checkpoints/model_trained_on_dns3.tar` (official DNS3)

Do not resume `training/outputs/finetune/ulunas_finetuned.pt` (16-step smoke).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FINETUNED_VBDEMAND = ROOT / "training" / "outputs" / "finetune_vbdemand" / "ulunas_finetuned.pt"
OFFICIAL_DNS3 = ROOT / "checkpoints" / "model_trained_on_dns3.tar"
SMOKE_FINETUNE = ROOT / "training" / "outputs" / "finetune" / "ulunas_finetuned.pt"


def resolve_continue_init(repo: Path | None = None) -> Path:
    repo = (repo or ROOT).resolve()
    candidates = [
        repo / "training" / "outputs" / "finetune_vbdemand" / "ulunas_finetuned.pt",
        repo / "checkpoints" / "model_trained_on_dns3.tar",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            if path.resolve() == (repo / SMOKE_FINETUNE.relative_to(ROOT)).resolve():
                continue
            return path
    raise FileNotFoundError(
        "no continue-train init: expected "
        "training/outputs/finetune_vbdemand/ulunas_finetuned.pt or "
        "checkpoints/model_trained_on_dns3.tar"
    )


def describe_init(repo: Path | None = None) -> dict:
    repo = (repo or ROOT).resolve()
    chosen = resolve_continue_init(repo)
    rel = chosen.relative_to(repo).as_posix() if chosen.is_relative_to(repo) else str(chosen)
    kind = "finetune_vbdemand" if chosen.name == "ulunas_finetuned.pt" else "official_dns3"
    return {"path": rel, "kind": kind, "bytes": chosen.stat().st_size}
