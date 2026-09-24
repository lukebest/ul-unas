"""Hour-capped MS-SNSD-style pair synthesizer (16 kHz).

Writes `{split}/{clean,noisy}` trees that `ingest_public_pairs` (same-name wav)
and `paired_layout.find_mssnsd_roots` can consume.

Official microsoft/MS-SNSD `noisyspeech_synthesizer.py` names files
`noisyN_SNRdb_*_clnspN.wav` / `clnspN.wav`. This mixer writes same-stem wavs
for ingest; `paired_layout.match_pairs` still understands the official names.

Smoke cap is 0.5–2.0 h (`--hours`, `--max_hours`). Use `--allow_large` to
raise the cap. Does not download corpora.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import SAMPLE_RATE
from training.audio_io import load_mono, mix_at_snr, write_wav
from training.paths import repo_rel
from training.synthesize_pairs import crop_or_tile, extract_residual_noise, synth_game_ambience


SMOKE_MAX_HOURS = 2.0
DEFAULT_HOURS = 0.5


def _iter_wavs(folder: Path) -> list[Path]:
    if not folder or not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".wav")


def discover_mssnsd_sources(root: Path) -> tuple[Path | None, Path | None]:
    """Locate official MS-SNSD clean/noise *source* dirs (not pre-mixed pairs)."""
    clean_names = ("clean_train", "CleanSpeech", "cleanspeech")
    noise_names = ("noise_train", "Noise", "noise")
    clean = next((root / n for n in clean_names if (root / n).is_dir()), None)
    noise = next((root / n for n in noise_names if (root / n).is_dir()), None)
    if clean is None:
        for p in root.rglob("*"):
            if p.is_dir() and p.name.lower() in {n.lower() for n in clean_names}:
                clean = p
                break
    if noise is None:
        for p in root.rglob("*"):
            if p.is_dir() and p.name.lower() in {n.lower() for n in noise_names}:
                # Do not pick NoisySpeech_training (already mixed).
                if "noisyspeech" in p.name.lower() or p.name.lower() == "noisy_train":
                    continue
                noise = p
                break
    return clean, noise


def load_pool(paths: list[Path]) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    for wav in paths:
        audio, _ = load_mono(wav)
        if len(audio) > 0:
            pool.append(audio)
    return pool


def speech_noise_pools(args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray], str]:
    repo = Path(args.repo)
    speech_dir = Path(args.speech_dir) if args.speech_dir else None
    noise_dir = Path(args.noise_dir) if args.noise_dir else None
    if speech_dir is None or noise_dir is None:
        src_root = Path(args.source_root) if args.source_root else repo / "training" / "data" / "raw" / "mssnsd"
        found_s, found_n = discover_mssnsd_sources(src_root)
        speech_dir = speech_dir or found_s
        noise_dir = noise_dir or found_n
    speech = load_pool(_iter_wavs(speech_dir) if speech_dir else [])
    noise = load_pool(_iter_wavs(noise_dir) if noise_dir else [])
    note = "mssnsd_sources"
    if not speech:
        speech = load_pool(_iter_wavs(repo / "audio" / "clean"))
        note = "repo_audio_clean"
    if not noise:
        # Prefer procedural / game-like noise for in-repo smokes. Residual
        # (official noisy−clean) remixed onto the same speech is a poor
        # stand-in for MS-SNSD everyday noise and often yields SI-SDRi < 0.
        rng = np.random.default_rng(args.seed)
        noise = [synth_game_ambience(SAMPLE_RATE * 4, SAMPLE_RATE, rng) for _ in range(4)]
        residual = extract_residual_noise(repo)
        if residual:
            noise.extend(0.35 * r for r in residual)
            note = f"{note}+procedural_noise+light_residual"
        else:
            note = f"{note}+procedural_noise"
    if not speech:
        raise SystemExit("no clean speech found; pass --speech_dir or keep audio/clean in the repo")
    if not noise:
        raise SystemExit("no noise found; pass --noise_dir")
    return speech, noise, note


def synthesize(args: argparse.Namespace) -> dict:
    hours = float(args.hours)
    max_hours = float(args.max_hours)
    if hours <= 0:
        raise SystemExit("--hours must be > 0")
    if hours > max_hours and not args.allow_large:
        raise SystemExit(
            f"--hours {hours} exceeds smoke cap --max_hours {max_hours}; "
            "pass --allow_large only when you intend a bigger local synth"
        )
    rng = np.random.default_rng(args.seed)
    speech, noise, source_note = speech_noise_pools(args)
    clip_s = float(args.seconds)
    n_samples = int(clip_s * SAMPLE_RATE)
    total_s = hours * 3600.0
    n_clips = max(1, int(round(total_s / clip_s)))
    n_train = max(1, int(round(n_clips * args.train_frac)))
    n_dev = max(1, int(round(n_clips * args.dev_frac))) if n_clips >= 3 else 1
    n_test = max(1, n_clips - n_train - n_dev)
    if n_train + n_dev + n_test > n_clips:
        n_train = max(1, n_clips - n_dev - n_test)

    snrs = [float(x) for x in args.snr]
    dest = Path(args.output_dir)
    written: list[dict] = []
    counts = {"train": n_train, "dev": n_dev, "test": n_test}
    idx = 0
    for split, n_clip in counts.items():
        for i in range(n_clip):
            sp = crop_or_tile(speech[idx % len(speech)], n_samples, rng)
            nz = crop_or_tile(noise[idx % len(noise)], n_samples, rng)
            snr = float(snrs[idx % len(snrs)])
            mix, _ = mix_at_snr(sp, nz, snr)
            stem = f"{split}_{i:04d}"
            noisy_path = dest / split / "noisy" / f"{stem}.wav"
            clean_path = dest / split / "clean" / f"{stem}.wav"
            write_wav(noisy_path, mix, SAMPLE_RATE)
            write_wav(clean_path, sp, SAMPLE_RATE)
            written.append(
                {
                    "id": stem,
                    "split": split,
                    "snr_db": snr,
                    "seconds": clip_s,
                    "noisy": repo_rel(noisy_path),
                    "clean": repo_rel(clean_path),
                }
            )
            idx += 1

    dest.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "MS-SNSD-style",
        "sr": SAMPLE_RATE,
        "hours_requested": hours,
        "hours_written": (n_train + n_dev + n_test) * clip_s / 3600.0,
        "seconds_per_clip": clip_s,
        "n": len(written),
        "by_split": counts,
        "snr_db": snrs,
        "source_note": source_note,
        "output_dir": repo_rel(dest),
        "max_hours": max_hours,
        "allow_large": bool(args.allow_large),
        "layout": "{split}/{clean,noisy} same-stem wav (ingestible)",
    }
    (dest / "synth_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Synthesize hour-capped MS-SNSD-style 16 kHz pairs")
    p.add_argument("--repo", default=str(ROOT))
    p.add_argument("--source_root", default="training/data/raw/mssnsd")
    p.add_argument("--speech_dir", default="")
    p.add_argument("--noise_dir", default="")
    p.add_argument("--output_dir", default="training/data/processed/mssnsd/16k")
    p.add_argument("--hours", type=float, default=DEFAULT_HOURS, help="target hours (smoke default 0.5)")
    p.add_argument("--max_hours", type=float, default=SMOKE_MAX_HOURS, help="refuse above this unless --allow_large")
    p.add_argument("--allow_large", action="store_true")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--dev_frac", type=float, default=0.1)
    p.add_argument("--snr", nargs="+", type=float, default=[0, 5, 10, 15, 20])
    p.add_argument("--seed", type=int, default=23)
    return p


if __name__ == "__main__":
    synthesize(build_parser().parse_args())
