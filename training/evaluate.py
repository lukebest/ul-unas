"""Layered evaluation, loudness-matched A/B, and offline RTF baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.audio_io import db_fs, load_mono, match_rms, write_wav
from training.metrics import (
    Timer,
    clip_rate,
    proxy_sig_bak,
    residual_energy,
    rtf,
    si_sdr,
    si_sdri,
    summarize_rows,
    try_estoi,
    try_pesq,
)
from training.model_utils import enhance_numpy, load_ulunas
from training.paths import resolve_audio


SCENE_MAP = {
    "75": "game_music",
    "85": "gunshot",
    "90": "mmo_bgm",
}


def infer_layer(name: str) -> str:
    if "clean" in name or name.endswith("_clean"):
        return "clean_only"
    if "75" in name:
        return "game_music"
    if "85" in name:
        return "gunshot"
    if "90" in name:
        return "mmo_bgm"
    if name.startswith("official"):
        return "official_paired"
    if name.startswith("p") and "_" in name:
        return "vbdemand"
    return "synthetic"


def load_manifest(path: Path) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def evaluate_pair(
    mix: np.ndarray,
    est: np.ndarray,
    clean: np.ndarray | None,
    sr: int,
    require_pesq: bool = False,
) -> dict:
    row = {
        "mix_dbfs": db_fs(mix),
        "est_dbfs": db_fs(est),
        "clip_rate": clip_rate(est),
        "duration_s": len(est) / sr,
    }
    row.update(proxy_sig_bak(est, mix, sr=sr))
    if clean is not None:
        n = min(len(mix), len(est), len(clean))
        mix, est, clean = mix[:n], est[:n], clean[:n]
        row["si_sdr"] = si_sdr(est, clean)
        row["si_sdri"] = si_sdri(est, clean, mix)
        estoi = try_estoi(est, clean, sr)
        if estoi is not None:
            row["estoi"] = estoi
        pesq_val = try_pesq(est, clean, sr, require=require_pesq)
        if pesq_val is not None:
            row["pesq"] = pesq_val
        elif require_pesq:
            raise RuntimeError("PESQ missing on a clean-reference row")
        row.update(residual_energy(est, clean, mix, sr=sr))
        row["clean_passthrough_sisdr"] = si_sdr(est, clean) if np.allclose(mix, clean, atol=1e-5) else None
        if row["clean_passthrough_sisdr"] is None:
            row.pop("clean_passthrough_sisdr")
    return row


def run_eval(args: argparse.Namespace) -> dict:
    device = args.device
    model = load_ulunas(args.ckpt, device=device)
    items = load_manifest(Path(args.manifest))
    if args.max_clips and args.max_clips > 0:
        items = items[: args.max_clips]
    has_clean = any(item.get("clean") for item in items)
    require_pesq = bool(getattr(args, "require_pesq", False)) and has_clean
    if require_pesq:
        try:
            import pesq  # noqa: F401
        except ImportError:
            raise SystemExit(
                "PESQ is required for this eval (pip install pesq). "
                "Re-run with --no-require-pesq to skip PESQ."
            )
    out_dir = Path(args.output_dir)
    ab_dir = out_dir / "ab_listen"
    enh_dir = out_dir / "enhanced"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_audio:
        ab_dir.mkdir(parents=True, exist_ok=True)
        enh_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    timer = Timer()
    audio_seconds = 0.0
    for item in items:
        name = item["id"]
        layer = item.get("layer") or infer_layer(name)
        mix, sr = load_mono(resolve_audio(item["noisy"]))
        clean = None
        if item.get("clean"):
            clean, _ = load_mono(resolve_audio(item["clean"]))
        chip = None
        if item.get("teacher"):
            chip, _ = load_mono(resolve_audio(item["teacher"]))

        with timer:
            est = enhance_numpy(model, mix, device=model.parameters().__next__().device)
        audio_seconds += len(mix) / sr
        if args.save_audio:
            write_wav(enh_dir / f"{name}_ulunas.wav", est, sr)

        row = evaluate_pair(mix, est, clean, sr, require_pesq=require_pesq)
        row.update({"name": name, "layer": layer, "system": "ulunas"})
        if args.save_audio:
            row["path"] = str(enh_dir / f"{name}_ulunas.wav")
        rows.append(row)

        row_in = evaluate_pair(mix, mix, clean, sr, require_pesq=require_pesq)
        row_in.update({"name": name, "layer": layer, "system": "noisy"})
        rows.append(row_in)

        if args.save_audio:
            write_wav(ab_dir / f"{name}_noisy.wav", mix, sr)
            write_wav(ab_dir / f"{name}_ulunas_loudness_matched.wav", match_rms(est, mix), sr)
        if chip is not None:
            n = min(len(mix), len(chip))
            chip_m = match_rms(chip[:n], mix[:n])
            if args.save_audio:
                write_wav(ab_dir / f"{name}_chip_loudness_matched.wav", chip_m, sr)
            row_chip = evaluate_pair(
                mix[:n],
                chip_m,
                clean[:n] if clean is not None else None,
                sr,
                require_pesq=require_pesq,
            )
            row_chip.update({"name": name, "layer": layer, "system": "chip_loudness_matched"})
            rows.append(row_chip)

    if require_pesq:
        missing = [r for r in rows if "si_sdr" in r and "pesq" not in r]
        if missing:
            raise SystemExit(
                f"PESQ missing on {len(missing)} clean-reference row(s); refusing SI-SDR-only table. "
                "Re-run with --no-require-pesq to skip PESQ."
            )

    elapsed = sum(timer.times)
    report = {
        "ckpt": str(args.ckpt or "checkpoints/model_trained_on_dns3.tar"),
        "device": device,
        "n_clips": len(items),
        "timing": {
            **timer.summary_ms(),
            "elapsed_s": elapsed,
            "audio_s": audio_seconds,
            "rtf": rtf(elapsed, audio_seconds),
            "frame_budget_ms": 1000.0 * 256 / 16000,
        },
        "by_layer_system": {},
        "rows": rows,
    }
    for system in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == system]
        report["by_layer_system"][system] = summarize_rows(sub)

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "rows"}
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(out_dir / "metrics.md", report)
    return report


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# UL-UNAS evaluation",
        "",
        f"- checkpoint: `{report['ckpt']}`",
        f"- RTF: {report['timing']['rtf']:.4f}",
        f"- mean / p99 clip time: {report['timing']['mean_ms']:.1f} / {report['timing']['p99_ms']:.1f} ms",
        "",
        "## Layer averages",
        "",
    ]
    for system, layers in report["by_layer_system"].items():
        lines.append(f"### {system}")
        lines.append("")
        keys = sorted({k for stats in layers.values() for k in stats})
        header = ["layer"] + keys
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for layer, stats in layers.items():
            vals = [layer] + [f"{stats[k]:.3f}" if k in stats else "-" for k in keys]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate UL-UNAS on a layered manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="training/outputs/eval")
    p.add_argument("--max_clips", type=int, default=0, help="0 = all clips")
    p.add_argument("--save_audio", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--require-pesq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if pesq is missing and the manifest has clean references (default: true)",
    )
    return p


if __name__ == "__main__":
    t0 = time.time()
    report = run_eval(build_parser().parse_args())
    print(json.dumps({"timing": report["timing"], "systems": list(report["by_layer_system"])}, indent=2))
    print(f"wrote report in {time.time() - t0:.1f}s")
