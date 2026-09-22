"""Reference and no-reference metrics used by evaluate.py."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from training.audio_io import active_mask, rms


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = np.asarray(est, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    n = min(len(est), len(ref))
    est = est[:n]
    ref = ref[:n]
    s_target = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    e_noise = est - s_target
    return float(10.0 * np.log10((np.dot(s_target, s_target) + eps) / (np.dot(e_noise, e_noise) + eps)))


def si_sdri(est: np.ndarray, ref: np.ndarray, mix: np.ndarray) -> float:
    return si_sdr(est, ref) - si_sdr(mix, ref)


def residual_energy(est: np.ndarray, ref: np.ndarray, mix: np.ndarray, sr: int = 16000) -> dict[str, float]:
    n = min(len(est), len(ref), len(mix))
    est, ref, mix = est[:n], ref[:n], mix[:n]
    speech = active_mask(ref, sr=sr)
    silence = ~speech
    def seg_rms(x, mask):
        if not np.any(mask):
            return 0.0
        return rms(x[mask])
    return {
        "speech_residual_rms": seg_rms(est - ref, speech),
        "silence_residual_rms": seg_rms(est, silence),
        "input_silence_rms": seg_rms(mix, silence),
        "silence_atten_db": 20.0 * np.log10((seg_rms(mix, silence) + 1e-12) / (seg_rms(est, silence) + 1e-12)),
    }


def try_pesq(est: np.ndarray, ref: np.ndarray, sr: int = 16000) -> float | None:
    try:
        from pesq import pesq as pesq_fn
    except ImportError:
        return None
    n = min(len(est), len(ref))
    if n < int(sr * 0.25):
        return None
    est = np.asarray(est[:n], dtype=np.float64)
    ref = np.asarray(ref[:n], dtype=np.float64)
    peak = max(np.max(np.abs(est)), np.max(np.abs(ref)), 1e-8)
    if peak > 1.0:
        est = est / peak
        ref = ref / peak
    try:
        mode = "wb" if sr >= 16000 else "nb"
        return float(pesq_fn(int(sr), ref, est, mode))
    except Exception:
        return None


def try_estoi(est: np.ndarray, ref: np.ndarray, sr: int = 16000) -> float | None:
    try:
        from pystoi import stoi
    except ImportError:
        return None
    n = min(len(est), len(ref))
    return float(stoi(ref[:n], est[:n], sr, extended=True))


def proxy_sig_bak(est: np.ndarray, mix: np.ndarray, sr: int = 16000) -> dict[str, float]:
    """Lightweight no-reference proxies when DNSMOS weights are unavailable."""
    from training.audio_io import active_mask, db_fs

    speech = active_mask(est, sr=sr, rel_db=-25.0)
    silence = ~speech
    sig = db_fs(est[speech]) if np.any(speech) else db_fs(est)
    bak = -db_fs(est[silence]) if np.any(silence) else 0.0
    mix_bak = -db_fs(mix[silence]) if np.any(silence) else 0.0
    return {
        "proxy_sig_dbfs": sig,
        "proxy_bak": bak,
        "proxy_bak_improve": bak - mix_bak,
        "proxy_ovrl": sig + 0.25 * (bak - mix_bak),
    }


def clip_rate(audio: np.ndarray, thr: float = 0.999) -> float:
    return float(np.mean(np.abs(audio) >= thr))


class Timer:
    def __init__(self) -> None:
        self.times: list[float] = []

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.times.append(time.perf_counter() - self._t0)
        return False

    def summary_ms(self) -> dict[str, float]:
        if not self.times:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        arr = np.array(self.times) * 1000.0
        return {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }


def rtf(elapsed_s: float, audio_s: float) -> float:
    return float(elapsed_s / max(audio_s, 1e-8))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(row.get("layer", "all"), []).append(row)
        by_layer.setdefault("all", []).append(row)
    out: dict[str, dict[str, float]] = {}
    skip = {"name", "layer", "path", "system"}
    for layer, items in by_layer.items():
        stats: dict[str, float] = {}
        keys = set()
        for item in items:
            keys.update(k for k, v in item.items() if k not in skip and isinstance(v, (int, float)))
        for key in sorted(keys):
            vals = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
            if vals:
                stats[key] = float(np.mean(vals))
        out[layer] = stats
    return out
