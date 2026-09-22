"""Scan local and optional public dataset roots, then write JSONL manifests.

VB-DMD JSONLs: `scan_vbdemand` vs `synthesize_pairs.ingest_public_pairs` —
both kept; see training/experiments/vbdemand_finetune.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.audio_io import db_fs, file_sha1, load_mono
from training.paths import repo_rel


AUDIO_EXTS = {".wav", ".flac", ".ogg"}

# Speakers held out from the 28-spk Valentini train set for validation.
# Official test speakers p232 / p257 stay in eval_vbdemand.jsonl.
DEV_SPEAKERS = {"p226", "p287"}


def iter_audio(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def record(path: Path, kind: str, source: str, extra: dict | None = None, repo: Path | None = None) -> dict:
    audio, sr = load_mono(path)
    rec = {
        "id": path.stem,
        "path": repo_rel(path, repo),
        "kind": kind,
        "source": source,
        "sr": sr,
        "duration_s": len(audio) / sr,
        "rms_dbfs": db_fs(audio),
        "sha1": file_sha1(path),
        "source_id": path.parent.name + "/" + path.stem,
    }
    if extra:
        rec.update(extra)
    return rec


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def write_jsonl_skip_empty(path: Path, rows: list[dict]) -> int:
    """Write JSONL unless `rows` is empty and `path` already has rows. Returns on-disk count."""
    if not rows:
        existing = jsonl_row_count(path)
        if existing:
            print(f"skip empty write; keep existing {path} ({existing} rows)")
            return existing
    write_jsonl(path, rows)
    return len(rows)


def speaker_id(stem: str) -> str:
    return stem.split("_")[0] if "_" in stem else stem


def paired_row(
    utt: str,
    noisy: Path,
    clean: Path,
    split: str,
    repo: Path,
    layer: str = "vbdemand",
    held_out: bool = False,
    extra: dict | None = None,
) -> dict:
    rec = {
        "id": utt,
        "noisy": repo_rel(noisy, repo),
        "clean": repo_rel(clean, repo),
        "layer": layer,
        "split": split,
        "scene": "speech_enhancement",
        "source": "VoiceBank+DEMAND",
        "speaker": speaker_id(utt),
        "held_out": held_out,
        "use_for_training": split == "train" and not held_out,
    }
    if extra:
        rec.update(extra)
    return rec


def find_vbdemand_roots(public_root: Path) -> list[tuple[Path, Path, str]]:
    """Return (clean_dir, noisy_dir, split_hint) pairs for VoiceBank+DEMAND layouts."""
    found: list[tuple[Path, Path, str]] = []
    if not public_root.exists():
        return found
    named = [
        (public_root / "voicebank_demand" / "16k" / "train" / "clean",
         public_root / "voicebank_demand" / "16k" / "train" / "noisy", "train"),
        (public_root / "voicebank_demand" / "16k" / "test" / "clean",
         public_root / "voicebank_demand" / "16k" / "test" / "noisy", "test"),
        (public_root / "voicebank_demand" / "16k" / "dev" / "clean",
         public_root / "voicebank_demand" / "16k" / "dev" / "noisy", "dev"),
    ]
    for clean_dir, noisy_dir, split in named:
        if clean_dir.is_dir() and noisy_dir.is_dir():
            found.append((clean_dir, noisy_dir, split))
    # Official unzip layout (possibly still 48 kHz; load_mono resamples).
    aliases = [
        ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav", "train"),
        ("clean_testset_wav", "noisy_testset_wav", "test"),
        ("clean_trainset_28spk_wav_16k", "noisy_trainset_28spk_wav_16k", "train"),
        ("clean_testset_wav_16k", "noisy_testset_wav_16k", "test"),
    ]
    for clean_name, noisy_name, split in aliases:
        for clean_dir in public_root.rglob(clean_name):
            if not clean_dir.is_dir():
                continue
            noisy_dir = clean_dir.parent / noisy_name
            if noisy_dir.is_dir() and (clean_dir, noisy_dir, split) not in found:
                found.append((clean_dir, noisy_dir, split))
    return found


def scan_vbdemand(public_root: Path, repo: Path) -> dict[str, list[dict]]:
    """Index VoiceBank+DEMAND pairs already on disk; do not copy or resample them.

    Use this writer (via `python training/prepare_manifest.py`) to build
    `{train,dev,eval}_vbdemand.jsonl` as part of the full catalog (speech/noise/
    demo/official manifests included). Matches clean/noisy by filename under
    layouts from `find_vbdemand_roots`, including wav/flac/ogg, with an rglob
    fallback when noisy is nested. Does not write `paired_all.jsonl`.

    The sibling writer is `synthesize_pairs.ingest_public_pairs`: same three
    VB-DMD JSONLs, plus `paired_all.jsonl` and optional `--copy_paired` 16 kHz
    copies. Keep both; see `training/experiments/vbdemand_finetune.md`.
    """
    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "eval": []}
    seen: set[str] = set()
    for clean_dir, noisy_dir, split_hint in find_vbdemand_roots(public_root):
        for wav in iter_audio(clean_dir):
            noisy = noisy_dir / wav.name
            if not noisy.exists():
                matches = list(noisy_dir.rglob(wav.name))
                noisy = matches[0] if matches else None
            if noisy is None or not noisy.exists():
                continue
            utt = wav.stem
            key = f"{split_hint}:{utt}"
            if key in seen:
                continue
            seen.add(key)
            spk = speaker_id(utt)
            if split_hint == "test":
                split = "eval"
                held_out = True
            elif split_hint == "dev" or spk in DEV_SPEAKERS:
                split = "dev"
                held_out = False
            else:
                split = "train"
                held_out = False
            buckets[split].append(
                paired_row(utt, noisy, wav, split if split != "eval" else "test", repo, held_out=held_out)
            )
    for split in buckets:
        buckets[split].sort(key=lambda r: r["id"])
    return buckets


def scan_public(data_root: Path, repo: Path) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {
        "speech": [],
        "noise_music": [],
        "noise_gun": [],
        "noise_general": [],
    }
    mapping = {
        "aishell": ("speech", "AISHELL-1"),
        "librispeech": ("speech", "LibriSpeech"),
        "vctk": ("speech", "VCTK"),
        "musan": ("noise_music", "MUSAN"),
        "slakh2100": ("noise_music", "Slakh2100"),
        "slakh": ("noise_music", "Slakh2100"),
        "fsd50k": ("noise_gun", "FSD50K"),
        "dns": ("noise_general", "DNS-Challenge"),
        "demand": ("noise_general", "DEMAND"),
    }
    if not data_root.exists():
        return found
    for child in data_root.iterdir():
        key = child.name.lower()
        if "voicebank" in key or "valentini" in key or "vbdemand" in key:
            continue
        for prefix, (kind, source) in mapping.items():
            if prefix in key:
                for wav in iter_audio(child)[:4000]:
                    found[kind].append(record(wav, kind.split("_")[0], source, {"split_hint": child.name}, repo=repo))
                break
    return found


def build(args: argparse.Namespace) -> dict:
    repo = Path(args.repo)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    speech = []
    noise = []
    for wav in iter_audio(repo / "audio" / "clean"):
        speech.append(record(wav, "speech", "repo_audio_clean", {"split_hint": "official"}, repo=repo))
    official_noise = []
    for wav in iter_audio(repo / "audio" / "noisy"):
        clean = repo / "audio" / "clean" / wav.name
        official_noise.append(
            {
                "id": f"official_{wav.stem}",
                "noisy": repo_rel(wav, repo),
                "clean": repo_rel(clean, repo) if clean.exists() else None,
                "layer": "official_paired",
                "held_out": True,
                "use_for_training": False,
            }
        )

    public = scan_public(Path(args.public_root), repo)
    speech.extend(public["speech"])
    noise.extend(public["noise_music"] + public["noise_gun"] + public["noise_general"])

    vb = scan_vbdemand(Path(args.public_root), repo)

    demo_eval = []
    for wav in sorted((repo / "demo").glob("*降噪前.wav")):
        teacher = repo / "demo" / wav.name.replace("降噪前", "降噪后")
        layer = "game_music"
        if "85db_噪声85" in wav.name:
            layer = "gunshot"
        elif "90" in wav.name:
            layer = "mmo_bgm"
        demo_eval.append(
            {
                "id": wav.stem,
                "noisy": repo_rel(wav, repo),
                "clean": None,
                "teacher": repo_rel(teacher, repo) if teacher.exists() else None,
                "layer": layer,
                "held_out": True,
                "use_for_training": False,
            }
        )

    clean_eval = []
    for wav in iter_audio(repo / "audio" / "clean"):
        clean_eval.append(
            {
                "id": f"clean_only_{wav.stem}",
                "noisy": repo_rel(wav, repo),
                "clean": repo_rel(wav, repo),
                "layer": "clean_only",
                "held_out": True,
                "use_for_training": False,
            }
        )

    write_jsonl(out / "speech.jsonl", speech)
    write_jsonl(out / "noise.jsonl", noise)
    write_jsonl(out / "eval_real.jsonl", demo_eval)
    write_jsonl(out / "eval_official.jsonl", official_noise)
    write_jsonl(out / "eval_clean.jsonl", clean_eval)
    write_jsonl(out / "eval_all.jsonl", demo_eval + official_noise + clean_eval)
    n_train = write_jsonl_skip_empty(out / "train_vbdemand.jsonl", vb["train"])
    n_dev = write_jsonl_skip_empty(out / "dev_vbdemand.jsonl", vb["dev"])
    n_eval = write_jsonl_skip_empty(out / "eval_vbdemand.jsonl", vb["eval"])

    summary = {
        "speech": len(speech),
        "noise": len(noise),
        "eval_real": len(demo_eval),
        "eval_official": len(official_noise),
        "eval_clean": len(clean_eval),
        "train_vbdemand": n_train,
        "dev_vbdemand": n_dev,
        "eval_vbdemand": n_eval,
        "public_root": repo_rel(Path(args.public_root), repo) if Path(args.public_root).exists() else str(args.public_root),
        "notes": (
            "Demo 降噪前/后 clips are held-out real tests and must not be used for training. "
            "In-repo official_paired clips are also held-out. VoiceBank+DEMAND eval is the official test speakers."
        ),
    }
    (out / "manifest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(ROOT))
    p.add_argument("--public_root", default="training/data/raw")
    p.add_argument("--output_dir", default="training/data/manifests")
    return p


if __name__ == "__main__":
    summary = build(build_parser().parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
