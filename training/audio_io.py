"""Resample, loudness, energy and STFT helpers used by the training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from training import HOP_LEN, N_FFT, SAMPLE_RATE, WIN_LEN


def load_mono(path: str | Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1).astype(np.float32)
    if sr != target_sr:
        audio = resample_to(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)


def resample_to(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)
    g = np.gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def rms(audio: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + eps))


def db_fs(audio: np.ndarray, eps: float = 1e-20) -> float:
    return 20.0 * np.log10(max(rms(audio), eps))


def match_rms(audio: np.ndarray, ref: np.ndarray, max_gain: float = 32.0) -> np.ndarray:
    gain = rms(ref) / max(rms(audio), 1e-12)
    gain = float(np.clip(gain, 1.0 / max_gain, max_gain))
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def frame_rms(audio: np.ndarray, frame: int = 320, hop: int = 160) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < frame:
        return np.array([rms(audio)], dtype=np.float64)
    n = 1 + (len(audio) - frame) // hop
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        sl = audio[i * hop : i * hop + frame]
        out[i] = np.sqrt(np.mean(sl * sl) + 1e-12)
    return out


def active_mask(audio: np.ndarray, sr: int = SAMPLE_RATE, rel_db: float = -30.0) -> np.ndarray:
    hop = sr // 100
    frame = hop * 2
    env = frame_rms(audio, frame=frame, hop=hop)
    thr = env.max() * (10.0 ** (rel_db / 20.0))
    mask = env >= max(thr, 1e-8)
    sample = np.repeat(mask, hop)
    if len(sample) < len(audio):
        sample = np.pad(sample, (0, len(audio) - len(sample)), constant_values=mask[-1] if len(mask) else False)
    return sample[: len(audio)]


def active_rms(audio: np.ndarray, sr: int = SAMPLE_RATE, rel_db: float = -30.0) -> float:
    mask = active_mask(audio, sr=sr, rel_db=rel_db)
    if not np.any(mask):
        return rms(audio)
    return rms(audio[mask])


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(speech), len(noise))
    speech = speech[:n].astype(np.float32)
    noise = noise[:n].astype(np.float32)
    s_rms = active_rms(speech)
    n_rms = rms(noise)
    scale = s_rms / max(n_rms, 1e-12) * (10.0 ** (-snr_db / 20.0))
    noise = noise * scale
    mix = np.clip(speech + noise, -1.0, 1.0).astype(np.float32)
    return mix, noise


def apply_rir(audio: np.ndarray, sr: int = SAMPLE_RATE, rt60: float = 0.2) -> np.ndarray:
    n = max(int(sr * min(rt60, 0.6)), 8)
    t = np.arange(n, dtype=np.float32) / sr
    decay = np.exp(-3.0 * t / max(rt60, 1e-3)).astype(np.float32)
    ir = (np.random.randn(n).astype(np.float32) * decay)
    ir[0] = 1.0
    ir = ir / (np.sqrt(np.sum(ir * ir)) + 1e-8)
    wet = np.convolve(audio, ir, mode="full")[: len(audio)]
    return wet.astype(np.float32)


def soft_clip(audio: np.ndarray, drive: float = 1.0) -> np.ndarray:
    return np.tanh(audio * drive).astype(np.float32)


def random_eq(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    tilt = (np.maximum(freqs, 1.0) / max(sr / 2, 1.0)) ** np.random.uniform(-0.4, 0.4)
    spec = spec * tilt.astype(np.complex64)
    out = np.fft.irfft(spec, n=len(audio))
    return out.astype(np.float32)


def stft_np(audio: np.ndarray, n_fft: int = N_FFT, hop: int = HOP_LEN, win: int = WIN_LEN) -> np.ndarray:
    window = np.hanning(win).astype(np.float32)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    n_frames = 1 + (len(audio) - win) // hop
    spec = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        frame = audio[i * hop : i * hop + win] * window
        spec[:, i] = np.fft.rfft(frame, n=n_fft)
    return spec


def istft_np(spec: np.ndarray, hop: int = HOP_LEN, win: int = WIN_LEN, length: int | None = None) -> np.ndarray:
    n_fft = (spec.shape[0] - 1) * 2
    window = np.hanning(win).astype(np.float32)
    n_frames = spec.shape[1]
    out_len = (n_frames - 1) * hop + win
    out = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    for i in range(n_frames):
        frame = np.fft.irfft(spec[:, i], n=n_fft).real[:win] * window
        sl = slice(i * hop, i * hop + win)
        out[sl] += frame
        wsum[sl] += window * window
    nz = wsum > 1e-8
    out[nz] /= wsum[nz]
    audio = out.astype(np.float32)
    if length is not None:
        if len(audio) < length:
            audio = np.pad(audio, (0, length - len(audio)))
        else:
            audio = audio[:length]
    return audio


def file_sha1(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
