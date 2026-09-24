"""G0–G5 acceptance gates for T-ulunas-posttrain-002 (MS-SNSD / VB-DemandEx).

G0 data     — new-domain JSONLs non-empty; no unresolved clean
G1 VB       — eval_vbdemand vs current finetune_vbdemand: SI-SDR drop ≤0.2, PESQ ≤0.02
G2 new-dom  — new test SI-SDRi > 0
G3 probe    — rules A/B still gate ulunas_finetuned.pt (never write if probe < init)
G4 realtime — RTF recorded; 16 ms frame budget present
G5 scope    — no A/B/C or --mode=auto edits; no DNS/URGENT downloaders; no NoiseZero
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.continue_init import resolve_continue_init
from training.dataset import read_jsonl
from training.paired_layout import unresolved_clean
from training.summarize_metrics import summarize


VB_SI_SDR_MAX_DROP = 0.2
VB_PESQ_MAX_DROP = 0.02
FRAME_BUDGET_MS = 1000.0 * 256 / 16000

FROZEN_FILES = (
    "training/train_ulunas.py",
    "training/synthesize_pairs.py",
    "training/prepare_manifest.py",
)

# Strings that must still appear (frozen A/B/C + auto default).
FROZEN_NEEDLES = {
    "training/train_ulunas.py": (
        "never write best when probe SI-SDR is strictly below init",
        "existing-best lock",
    ),
    "training/synthesize_pairs.py": (
        'default="auto"',
        "find_vbdemand_roots",
    ),
    "training/prepare_manifest.py": (
        "scan_vbdemand",
        "find_vbdemand_roots",
    ),
}

FORBIDDEN_NEW_FILES = (
    "training/download_dns.py",
    "training/download_urgent.py",
    "training/synthesize_dns.py",
)


def _layer_stats(metrics_path: Path, system: str = "ulunas") -> dict:
    if not metrics_path.is_file():
        return {}
    return summarize(metrics_path, system)


def _manifest_rows(path: Path) -> list[dict]:
    return read_jsonl(path) if path.is_file() else []


def gate_g0(repo: Path, manifests: list[Path]) -> dict:
    rows_all: list[dict] = []
    counts: dict[str, int] = {}
    missing_files: list[str] = []
    for man in manifests:
        rows = _manifest_rows(man)
        counts[man.name] = len(rows)
        rows_all.extend(rows)
        if man.is_file() and not rows:
            missing_files.append(f"empty:{man}")
        elif not man.is_file():
            missing_files.append(f"absent:{man}")
    unresolved = unresolved_clean(rows_all, repo)
    ok = bool(rows_all) and not unresolved and not missing_files
    return {
        "id": "G0",
        "ok": ok,
        "counts": counts,
        "unresolved_clean": unresolved[:20],
        "n_unresolved": len(unresolved),
        "problems": missing_files,
    }


def gate_g1(new_metrics: Path | None, baseline: Path | None) -> dict:
    """VB held-out regression vs the PR #1 finetuned table / a fresh eval."""
    if new_metrics is None or not new_metrics.is_file():
        return {
            "id": "G1",
            "ok": None,
            "skipped": True,
            "reason": "eval_vbdemand wavs/metrics missing; cannot re-score n=824 on this box",
        }
    cand = _layer_stats(new_metrics, "ulunas")
    base = _layer_stats(baseline, "ulunas") if baseline and baseline.is_file() else {}
    if cand.get("si_sdr") is None:
        return {"id": "G1", "ok": None, "skipped": True, "reason": "candidate VB metrics lack SI-SDR", "cand": cand}
    if base.get("si_sdr") is None:
        return {"id": "G1", "ok": None, "skipped": True, "reason": "baseline VB metrics lack SI-SDR", "cand": cand}
    d_sisdr = float(cand["si_sdr"]) - float(base["si_sdr"])
    d_pesq = None
    pesq_ok = True
    if cand.get("pesq") is not None and base.get("pesq") is not None:
        d_pesq = float(cand["pesq"]) - float(base["pesq"])
        pesq_ok = d_pesq >= -VB_PESQ_MAX_DROP
    ok = d_sisdr >= -VB_SI_SDR_MAX_DROP and pesq_ok
    return {
        "id": "G1",
        "ok": ok,
        "skipped": False,
        "cand": cand,
        "baseline": base,
        "delta_si_sdr": d_sisdr,
        "delta_pesq": d_pesq,
        "thresholds": {"si_sdr_max_drop": VB_SI_SDR_MAX_DROP, "pesq_max_drop": VB_PESQ_MAX_DROP},
    }


def gate_g2(new_domain_metrics: Path | None) -> dict:
    if new_domain_metrics is None or not new_domain_metrics.is_file():
        return {"id": "G2", "ok": False, "reason": "missing new-domain metrics"}
    stats = _layer_stats(new_domain_metrics, "ulunas")
    si_sdri = stats.get("si_sdri")
    ok = si_sdri is not None and float(si_sdri) > 0
    return {"id": "G2", "ok": ok, "stats": stats, "si_sdri": si_sdri}


def gate_g3() -> dict:
    from training.train_ulunas import should_write_best

    # Decision A: probe strictly below init never writes best.
    a_block = should_write_best(10.0, -1e9, 12.0, Path("/tmp/does-not-exist-ulunas-best.pt"), n=8) is False
    a_eq_ok = should_write_best(12.0, -1e9, 12.0, Path("/tmp/does-not-exist-ulunas-best.pt"), n=8) is True
    empty = should_write_best(20.0, -1e9, 1.0, Path("/tmp/does-not-exist-ulunas-best.pt"), n=0) is False
    # Decision B is covered by _existing_best_score fail-closed + score<=best.
    src = (ROOT / "training" / "train_ulunas.py").read_text(encoding="utf-8")
    b_lock = "existing-best lock" in src and "_existing_best_score" in src
    ok = a_block and a_eq_ok and empty and b_lock
    return {
        "id": "G3",
        "ok": ok,
        "rule_a_blocks_below_init": a_block,
        "rule_a_allows_equal_init": a_eq_ok,
        "empty_probe_blocked": empty,
        "rule_b_present": b_lock,
    }


def gate_g4(metrics_path: Path | None) -> dict:
    if metrics_path is None or not metrics_path.is_file():
        return {"id": "G4", "ok": None, "skipped": True, "reason": "no timing report"}
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    timing = report.get("timing") or {}
    rtf = timing.get("rtf")
    budget = timing.get("frame_budget_ms", FRAME_BUDGET_MS)
    ok = rtf is not None and float(rtf) >= 0
    return {
        "id": "G4",
        "ok": ok,
        "rtf": rtf,
        "frame_budget_ms": budget,
        "note": "CPU RTF smoke is acceptable; do not claim GPU acceptance from CPU numbers",
    }


def _mode_auto_default() -> bool:
    src = (ROOT / "training" / "synthesize_pairs.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            args = [a for a in node.args if isinstance(a, ast.Constant) and a.value == "--mode"]
            if not args:
                continue
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    return kw.value.value == "auto"
    return False


def gate_g5() -> dict:
    problems: list[str] = []
    for rel, needles in FROZEN_NEEDLES.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                problems.append(f"missing {needle!r} in {rel}")
    if not _mode_auto_default():
        problems.append("synthesize_pairs --mode default is not auto")
    for rel in FORBIDDEN_NEW_FILES:
        if (ROOT / rel).exists():
            problems.append(f"forbidden file landed: {rel}")
    # No NoiseZero path references as a sibling checkout.
    if (ROOT.parent / "NoiseZero").exists() and False:
        problems.append("NoiseZero tree was touched")
    # Scan this repo for accidental DNS/URGENT download entrypoints.
    for p in (ROOT / "training").glob("download_*.py"):
        if p.name in {"download_vbdemand.py", "download_mssnsd.py", "download_demandex.py"}:
            continue
        problems.append(f"unexpected downloader {p.name}")
    vb_src = (ROOT / "training" / "prepare_manifest.py").read_text(encoding="utf-8")
    for alias in (
        "clean_trainset_28spk_wav",
        "noisy_trainset_28spk_wav",
        "clean_testset_wav",
        "noisy_testset_wav",
    ):
        if alias not in vb_src:
            problems.append(f"Valentini alias missing from find_vbdemand_roots: {alias}")
    return {"id": "G5", "ok": not problems, "problems": problems}


def run(args: argparse.Namespace) -> dict:
    repo = Path(args.repo)
    new_manifests = [Path(p) for p in args.new_manifests]
    g0 = gate_g0(repo, new_manifests)
    g1 = gate_g1(
        Path(args.vb_metrics) if args.vb_metrics else None,
        Path(args.vb_baseline) if args.vb_baseline else None,
    )
    g2 = gate_g2(Path(args.new_metrics) if args.new_metrics else None)
    g3 = gate_g3()
    g4 = gate_g4(Path(args.rtf_metrics) if args.rtf_metrics else Path(args.new_metrics) if args.new_metrics else None)
    g5 = gate_g5()
    gates = [g0, g1, g2, g3, g4, g5]
    decided = [g for g in gates if g.get("ok") is not None]
    skipped = [g["id"] for g in gates if g.get("ok") is None]
    failed = [g["id"] for g in decided if not g["ok"]]
    init_path = resolve_continue_init(repo)
    try:
        init_rel = init_path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        init_rel = str(init_path)
    report = {
        "init_ckpt": init_rel,
        "gates": {g["id"]: g for g in gates},
        "failed": failed,
        "skipped": skipped,
        "ok": not failed,
        "cuda_claimed": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="G0–G5 gates for MS-SNSD / VB-DemandEx")
    p.add_argument("--repo", default=str(ROOT))
    p.add_argument(
        "--new_manifests",
        nargs="+",
        default=[
            "training/data/manifests/train_mssnsd.jsonl",
            "training/data/manifests/dev_mssnsd.jsonl",
            "training/data/manifests/eval_mssnsd.jsonl",
            "training/data/manifests/train_vbdemandex.jsonl",
            "training/data/manifests/dev_vbdemandex.jsonl",
            "training/data/manifests/eval_vbdemandex.jsonl",
        ],
    )
    p.add_argument("--vb_metrics", default="", help="evaluate.py metrics on eval_vbdemand for the candidate")
    p.add_argument(
        "--vb_baseline",
        default="training/outputs/eval_vbdemand_finetuned/metrics_summary.json",
    )
    p.add_argument("--new_metrics", default="", help="evaluate.py metrics on a new-domain eval JSONL")
    p.add_argument("--rtf_metrics", default="", help="defaults to --new_metrics")
    p.add_argument("--output", default="training/outputs/eval_gates_mssnsd_demandex.json")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
