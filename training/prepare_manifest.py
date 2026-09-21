"""Scan local and optional public dataset roots, then write JSONL manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.audio_io import db_fs, file_sha1, load_mono


AUDIO_EXTS = {".wav", ".flac", ".ogg"}


def iter_audio(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def record(path: Path, kind: str, source: str, extra: dict | None = None) -> dict:
    audio, sr = load_mono(path)
    rec = {
        "id": path.stem,
        "path": str(path.resolve()),
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


def scan_public(data_root: Path) -> dict[str, list[dict]]:
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
        for prefix, (kind, source) in mapping.items():
            if prefix in key:
                for wav in iter_audio(child)[:4000]:
                    found[kind].append(record(wav, kind.split("_")[0], source, {"split_hint": child.name}))
                break
    return found


def build(args: argparse.Namespace) -> dict:
    repo = Path(args.repo)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    speech = []
    noise = []
    for wav in iter_audio(repo / "audio" / "clean"):
        speech.append(record(wav, "speech", "repo_audio_clean", {"split_hint": "official"}))
    official_noise = []
    for wav in iter_audio(repo / "audio" / "noisy"):
        clean = repo / "audio" / "clean" / wav.name
        official_noise.append(
            {
                "id": f"official_{wav.stem}",
                "noisy": str(wav.resolve()),
                "clean": str(clean.resolve()) if clean.exists() else None,
                "layer": "official_paired",
                "held_out": False,
            }
        )

    public = scan_public(Path(args.public_root))
    speech.extend(public["speech"])
    noise.extend(public["noise_music"] + public["noise_gun"] + public["noise_general"])

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
                "noisy": str(wav.resolve()),
                "clean": None,
                "teacher": str(teacher.resolve()) if teacher.exists() else None,
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
                "noisy": str(wav.resolve()),
                "clean": str(wav.resolve()),
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

    summary = {
        "speech": len(speech),
        "noise": len(noise),
        "eval_real": len(demo_eval),
        "eval_official": len(official_noise),
        "eval_clean": len(clean_eval),
        "public_root": str(Path(args.public_root)),
        "notes": "Demo 降噪前/后 clips are held-out real tests and must not be used for training selection.",
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
