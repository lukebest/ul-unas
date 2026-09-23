"""Align chip teacher audio, remove slow AGC, and export mask-distillation clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import correlate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import HOP_LEN, N_FFT, SAMPLE_RATE
from training.audio_io import frame_rms, load_mono, stft_np, write_wav
from training.paths import repo_rel, resolve_audio


def gcc_phat_delay(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, float]:
    n = int(2 ** np.ceil(np.log2(len(x) + len(y))))
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    R = X * np.conj(Y)
    R /= np.abs(R) + 1e-8
    cc = np.fft.irfft(R, n=n)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    i = int(np.argmax(cc))
    return int(lags[i]), float(cc[i])


def envelope_delay(x: np.ndarray, y: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[float, float]:
    hop = sr // 100
    ex = frame_rms(x, frame=hop * 2, hop=hop)
    ey = frame_rms(y, frame=hop * 2, hop=hop)
    ex = (ex - ex.mean()) / (ex.std() + 1e-8)
    ey = (ey - ey.mean()) / (ey.std() + 1e-8)
    cc = correlate(ey, ex, mode="full", method="fft")
    lags = np.arange(-len(ex) + 1, len(ey))
    keep = np.abs(lags) <= 200
    lag = int(lags[keep][np.argmax(cc[keep])])
    peak = float(cc[keep].max() / max(min(len(ex), len(ey)), 1))
    return lag * hop / sr, peak


def window_delays(x: np.ndarray, y: np.ndarray, sr: int, win_s: float = 4.0) -> list[tuple[float, float]]:
    win = int(win_s * sr)
    hop = win // 2
    out = []
    for start in range(0, max(len(x) - win, 1), hop):
        xs = x[start : start + win]
        ys = y[start : start + win]
        if len(xs) < win or len(ys) < win:
            continue
        delay, score = gcc_phat_delay(xs, ys, max_lag=sr // 5)
        out.append((start / sr, delay / sr, score))
    return out


def fit_delay_drift(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    t = np.array([p[0] for p in points], dtype=np.float64)
    d = np.array([p[1] for p in points], dtype=np.float64)
    w = np.clip(np.array([p[2] for p in points], dtype=np.float64), 1e-3, None)
    A = np.stack([np.ones_like(t), t], axis=1)
    Aw = A * w[:, None]
    coeff, *_ = np.linalg.lstsq(Aw, d * w, rcond=None)
    return float(coeff[0]), float(coeff[1])


def apply_delay(audio: np.ndarray, delay_s: float, sr: int) -> np.ndarray:
    shift = int(round(delay_s * sr))
    if shift > 0:
        return np.pad(audio, (shift, 0))[: len(audio)]
    if shift < 0:
        return np.pad(audio, (0, -shift))[-shift : -shift + len(audio)]
    return audio


def remove_slow_gain(teacher: np.ndarray, student_ref: np.ndarray, sr: int) -> np.ndarray:
    hop = sr // 10
    frame = hop * 4
    te = frame_rms(teacher, frame=frame, hop=hop)
    re = frame_rms(student_ref, frame=frame, hop=hop)
    gain = re / np.maximum(te, 1e-6)
    gain = np.clip(gain, 0.05, 20.0)
    sample = np.repeat(gain, hop)
    if len(sample) < len(teacher):
        sample = np.pad(sample, (0, len(teacher) - len(sample)), constant_values=gain[-1])
    return (teacher * sample[: len(teacher)]).astype(np.float32)


def magnitude_mask(mix: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    xs = np.abs(stft_np(mix))
    ys = np.abs(stft_np(teacher))
    t = min(xs.shape[1], ys.shape[1])
    mask = np.clip(ys[:, :t] / np.maximum(xs[:, :t], 1e-6), 0.0, 1.0)
    return mask.astype(np.float32)


def align_one(noisy_path: Path, teacher_path: Path, out_dir: Path, min_corr: float) -> dict:
    mix, _ = load_mono(noisy_path)
    teacher, _ = load_mono(teacher_path)
    n = min(len(mix), len(teacher))
    mix, teacher = mix[:n], teacher[:n]
    env_delay, env_peak = envelope_delay(mix, teacher)
    points = window_delays(mix, teacher, SAMPLE_RATE)
    a, b = fit_delay_drift(points)
    delay = a if points else env_delay
    aligned = apply_delay(teacher, delay, SAMPLE_RATE)
    aligned = remove_slow_gain(aligned, mix, SAMPLE_RATE)
    corr = float(np.corrcoef(np.abs(mix), np.abs(aligned))[0, 1])
    clip = float(np.mean(np.abs(teacher) >= 0.999))
    keep = corr >= min_corr and clip < 0.02 and abs(b) < 2e-4
    rec = {
        "id": noisy_path.stem,
        "noisy": repo_rel(noisy_path),
        "teacher_raw": repo_rel(teacher_path),
        "delay_s": delay,
        "drift": b,
        "env_delay_s": env_delay,
        "env_peak": env_peak,
        "corr_abs": corr,
        "clip_rate": clip,
        "keep": keep,
        "n_windows": len(points),
    }
    if keep:
        out_wav = out_dir / f"{noisy_path.stem}_teacher_aligned.wav"
        write_wav(out_wav, aligned, SAMPLE_RATE)
        mask = magnitude_mask(mix, aligned)
        np.save(out_dir / f"{noisy_path.stem}_teacher_mask.npy", mask)
        rec["teacher_aligned"] = repo_rel(out_wav)
        rec["mask_path"] = repo_rel(out_dir / f"{noisy_path.stem}_teacher_mask.npy")
        rec["noisy_aligned"] = repo_rel(noisy_path)
    return rec


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with Path(args.manifest).open() as f:
        items = [json.loads(line) for line in f if line.strip()]
    for item in items:
        if not item.get("teacher"):
            continue
        rows.append(align_one(resolve_audio(item["noisy"]), resolve_audio(item["teacher"]), out_dir, args.min_corr))
    keep = [r for r in rows if r.get("keep")]
    manifest = []
    for r in keep:
        manifest.append(
            {
                "id": r["id"] + "_teacher",
                "noisy": r["noisy"],
                "teacher": r["teacher_aligned"],
                "mask_path": r["mask_path"],
                "layer": "teacher",
                "use_for_training": False,
                "held_out": True,
                "note": "Demo chip clips stay held-out; use only after replacing with non-test recordings.",
            }
        )
    man_path = Path(args.manifest_out)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with man_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"n": len(rows), "kept": len(keep), "rows": rows}
    (out_dir / "align_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n": len(rows), "kept": len(keep), "mean_delay_s": float(np.mean([r["delay_s"] for r in rows]) if rows else 0)}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="training/data/manifests/eval_real.jsonl")
    p.add_argument("--output_dir", default="training/outputs/teacher")
    p.add_argument("--manifest_out", default="training/data/manifests/teacher.jsonl")
    p.add_argument("--min_corr", type=float, default=0.35)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
