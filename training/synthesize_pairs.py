"""Synthesize strictly paired 16 kHz training clips from speech + target noise."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.audio_io import apply_rir, load_mono, mix_at_snr, random_eq, soft_clip, write_wav
from training import SAMPLE_RATE


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def crop_or_tile(audio: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) >= n:
        start = int(rng.integers(0, len(audio) - n + 1))
        return audio[start : start + n].copy()
    reps = int(np.ceil(n / max(len(audio), 1)))
    return np.tile(audio, reps)[:n]


def extract_residual_noise(repo: Path) -> list[np.ndarray]:
    noises = []
    noisy_dir = repo / "audio" / "noisy"
    clean_dir = repo / "audio" / "clean"
    for wav in sorted(noisy_dir.glob("*.wav")):
        clean = clean_dir / wav.name
        if not clean.exists():
            continue
        n, _ = load_mono(wav)
        c, _ = load_mono(clean)
        m = min(len(n), len(c))
        noises.append((n[:m] - c[:m]).astype(np.float32))
    return noises


def synth_music(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    f0 = float(rng.uniform(110, 330))
    wav = np.zeros(n, dtype=np.float32)
    for k in range(1, int(rng.integers(4, 9))):
        wav += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
    beat = (np.sin(2 * np.pi * rng.uniform(1.5, 4.0) * t) > 0).astype(np.float32)
    wav = wav * (0.4 + 0.6 * beat)
    return (wav / (np.max(np.abs(wav)) + 1e-6) * rng.uniform(0.2, 0.6)).astype(np.float32)


def synth_gun(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    wav = np.zeros(n, dtype=np.float32)
    n_shots = int(rng.integers(2, 8))
    for _ in range(n_shots):
        pos = int(rng.integers(0, max(n - 1, 1)))
        length = int(sr * rng.uniform(0.04, 0.25))
        burst = rng.normal(0, 1, length).astype(np.float32)
        burst *= np.exp(-np.linspace(0, 8, length)).astype(np.float32)
        end = min(n, pos + length)
        wav[pos:end] += burst[: end - pos] * rng.uniform(0.4, 1.0)
    return np.clip(wav, -1, 1).astype(np.float32)


def synth_game_ambience(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    pink = rng.normal(0, 1, n).astype(np.float32)
    spec = np.fft.rfft(pink)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec /= np.maximum(freqs, 20.0) ** 0.5
    amb = np.fft.irfft(spec, n=n).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / sr
    tone = 0.15 * np.sin(2 * np.pi * rng.uniform(180, 520) * t)
    return 0.25 * amb / (np.max(np.abs(amb)) + 1e-6) + tone.astype(np.float32)


def maybe_interferer(speech_pool: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray | None:
    if len(speech_pool) < 2 or rng.random() > 0.25:
        return None
    other = speech_pool[int(rng.integers(0, len(speech_pool)))]
    return 0.35 * crop_or_tile(other, n, rng)


def synthesize(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    repo = Path(args.repo)
    out_root = Path(args.output_dir)
    speech_rows = read_jsonl(Path(args.speech_manifest))
    if not speech_rows:
        raise SystemExit("speech manifest is empty; run prepare_manifest.py first")
    speech_audio = []
    speech_meta = []
    for row in speech_rows:
        wav, _ = load_mono(row["path"])
        speech_audio.append(wav)
        speech_meta.append(row)

    residual = extract_residual_noise(repo)
    n_samples = int(args.seconds * SAMPLE_RATE)
    scenes = ["game_music", "gunshot", "mmo_bgm", "clean_only", "noise_only"]
    counts = {"train": args.n_train, "dev": args.n_dev, "test": args.n_test}
    written = []

    idx = 0
    for split, n_clip in counts.items():
        for i in range(n_clip):
            scene = scenes[i % len(scenes)]
            spk_i = 0 if split == "train" else 1 % max(len(speech_audio), 1)
            if len(speech_audio) == 1:
                spk_i = 0
            elif split == "test":
                spk_i = len(speech_audio) - 1
            speech = crop_or_tile(speech_audio[spk_i], n_samples, rng)
            noise = np.zeros(n_samples, dtype=np.float32)
            if scene == "game_music":
                noise += synth_music(n_samples, SAMPLE_RATE, rng)
                if residual:
                    noise += 0.35 * crop_or_tile(residual[0], n_samples, rng)
            elif scene == "gunshot":
                noise += synth_gun(n_samples, SAMPLE_RATE, rng)
                noise += 0.2 * synth_game_ambience(n_samples, SAMPLE_RATE, rng)
            elif scene == "mmo_bgm":
                noise += 0.7 * synth_music(n_samples, SAMPLE_RATE, rng)
                noise += 0.5 * synth_game_ambience(n_samples, SAMPLE_RATE, rng)
                inter = maybe_interferer(speech_audio, n_samples, rng)
                if inter is not None:
                    noise += inter
            elif scene == "noise_only":
                noise += synth_game_ambience(n_samples, SAMPLE_RATE, rng)
                speech = np.zeros_like(speech)
            if residual and scene not in {"clean_only"}:
                noise += 0.15 * crop_or_tile(residual[min(1, len(residual) - 1)], n_samples, rng)

            if scene == "clean_only":
                mix = speech.copy()
                target = speech.copy()
                snr = 99.0
            elif scene == "noise_only":
                mix = np.clip(noise, -1, 1).astype(np.float32)
                target = np.zeros_like(mix)
                snr = -99.0
            else:
                snr = float(rng.choice([-8, -5, 0, 5, 8, 12, 18], p=[0.08, 0.18, 0.22, 0.22, 0.15, 0.10, 0.05]))
                mix, _ = mix_at_snr(speech, noise, snr)
                target = speech

            if rng.random() < 0.35 and scene not in {"clean_only"}:
                mix = apply_rir(mix, SAMPLE_RATE, rt60=float(rng.uniform(0.08, 0.35)))
                mix = random_eq(mix)
            if rng.random() < 0.2:
                mix = soft_clip(mix, drive=float(rng.uniform(1.0, 1.8)))

            stem = f"{split}_{scene}_{i:04d}"
            noisy_path = out_root / split / "noisy" / f"{stem}.wav"
            clean_path = out_root / split / "clean" / f"{stem}.wav"
            write_wav(noisy_path, mix, SAMPLE_RATE)
            write_wav(clean_path, target, SAMPLE_RATE)
            written.append(
                {
                    "id": stem,
                    "noisy": str(noisy_path.resolve()),
                    "clean": str(clean_path.resolve()),
                    "layer": scene if scene != "noise_only" else "synthetic",
                    "split": split,
                    "snr_db": snr,
                    "scene": scene,
                    "source_id": speech_meta[spk_i]["source_id"],
                    "use_for_training": split == "train",
                }
            )
            idx += 1

    write_jsonl(Path(args.manifest_out), written)
    write_jsonl(Path(args.manifest_out).with_name("train_synth.jsonl"), [r for r in written if r["split"] == "train"])
    write_jsonl(Path(args.manifest_out).with_name("dev_synth.jsonl"), [r for r in written if r["split"] == "dev"])
    write_jsonl(Path(args.manifest_out).with_name("eval_synth.jsonl"), [r for r in written if r["split"] == "test"])
    summary = {"n": len(written), "by_split": {k: v for k, v in counts.items()}}
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(ROOT))
    p.add_argument("--speech_manifest", default="training/data/manifests/speech.jsonl")
    p.add_argument("--output_dir", default="training/data/synth")
    p.add_argument("--manifest_out", default="training/data/manifests/synth_all.jsonl")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--n_train", type=int, default=40)
    p.add_argument("--n_dev", type=int, default=10)
    p.add_argument("--n_test", type=int, default=10)
    p.add_argument("--seed", type=int, default=17)
    return p


if __name__ == "__main__":
    synthesize(build_parser().parse_args())
