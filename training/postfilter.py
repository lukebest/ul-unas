"""Causal gain-warp and conservative MCRA/OM-LSA post-filters sharing UL-UNAS STFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.audio_io import istft_np, load_mono, write_wav
from training.metrics import proxy_sig_bak
from training.model_utils import enhance_with_mask, load_ulunas


def warp_gain(gain: np.ndarray, strength: float) -> np.ndarray:
    """strength 0.01/0.02/0.04: more attenuation on uncertain bins, slight lift on clean bins."""
    g = np.clip(gain, 1e-4, 1.0)
    alpha = 1.0 + 25.0 * strength
    warped = np.power(g, alpha)
    lift = 1.0 + strength * (g > 0.8)
    return np.clip(warped * lift, 0.0, 1.0).astype(np.float32)


def apply_gain_to_mix(mix: np.ndarray, gain: np.ndarray) -> np.ndarray:
    spec = _stft(mix)
    t = min(spec.shape[1], gain.shape[-1] if gain.ndim == 2 else spec.shape[1])
    if gain.ndim == 3:
        gain = gain[0]
    if gain.shape[0] != spec.shape[0]:
        # gain is (T, F) or (F, T)
        if gain.shape[0] == spec.shape[1]:
            gain = gain.T
    g = gain[:, :t]
    spec = spec[:, :t] * g
    return istft_np(spec, length=len(mix))


def _stft(mix: np.ndarray):
    from training.audio_io import stft_np

    return stft_np(mix)


def mcra_omlsa(mix: np.ndarray, prior_gain: np.ndarray | None, max_atten_db: float, freeze_transient: bool = True) -> np.ndarray:
    spec = _stft(mix)
    mag = np.abs(spec) + 1e-8
    noise = mag[:, : max(4, mag.shape[1] // 20)].mean(axis=1, keepdims=True)
    noise = np.maximum(noise, 1e-6)
    alpha_d = 0.85
    gains = []
    floor = 10.0 ** (-max_atten_db / 20.0)
    prev = mag[:, 0]
    for t in range(mag.shape[1]):
        x = mag[:, t]
        transient = bool(np.mean(x / (prev + 1e-8)) > 3.5)
        if not (freeze_transient and transient):
            speech_prob = np.clip(x / (noise[:, 0] * 1.5 + 1e-8), 0.0, 1.0)
            noise[:, 0] = alpha_d * noise[:, 0] + (1 - alpha_d) * (1 - speech_prob) * x
        xi = np.clip(x ** 2 / (np.maximum(noise[:, 0], 1e-4) ** 2) - 1.0, 0.0, 100.0)
        gain = xi / (xi + 1.0)
        gain = np.maximum(gain, floor)
        if prior_gain is not None:
            pg = prior_gain[:, t] if prior_gain.shape[1] > t else prior_gain[:, -1]
            gain = np.minimum(gain, np.maximum(pg, floor))
        gains.append(gain)
        prev = x
    g = np.stack(gains, axis=1).astype(np.float32)
    return apply_gain_to_mix(mix, g)


def run_ablation(args: argparse.Namespace) -> dict:
    model = load_ulunas(args.ckpt, device=args.device)
    items = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("none", None, None),
        ("warp_0.01", 0.01, None),
        ("warp_0.02", 0.02, None),
        ("warp_0.04", 0.04, None),
        ("omlsa_3db", None, 3.0),
        ("omlsa_6db", None, 6.0),
        ("omlsa_9db", None, 9.0),
    ]
    rows = []
    for item in items:
        mix, sr = load_mono(item["noisy"])
        est, gain, _ = enhance_with_mask(model, mix)
        if gain.ndim == 3:
            gain = gain[0]
        if gain.shape[0] != (512 // 2 + 1) and gain.shape[-1] == (512 // 2 + 1):
            gain = np.transpose(gain, (1, 0)) if gain.ndim == 2 else gain
        for name, warp, omlsa in configs:
            y = est
            if warp is not None:
                y = apply_gain_to_mix(mix, warp_gain(gain, warp))
            if omlsa is not None:
                y = mcra_omlsa(mix, gain, omlsa)
            write_wav(out_dir / f"{item['id']}_{name}.wav", y, sr)
            met = proxy_sig_bak(y, mix, sr)
            met.update({"id": item["id"], "config": name, "layer": item.get("layer")})
            rows.append(met)
    report = {"rows": rows}
    (out_dir / "postfilter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    by = {}
    for r in rows:
        by.setdefault(r["config"], []).append(r["proxy_bak_improve"])
    summary = {k: float(np.mean(v)) for k, v in by.items()}
    print(json.dumps(summary, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="training/data/manifests/eval_real.jsonl")
    p.add_argument("--ckpt", default="checkpoints/model_trained_on_dns3.tar")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="training/outputs/postfilter")
    return p


if __name__ == "__main__":
    run_ablation(build_parser().parse_args())
