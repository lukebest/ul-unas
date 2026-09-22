"""Download VoiceBank+DEMAND (Valentini) and write 16 kHz paired wavs.

Primary source: Edinburgh DataShare (48 kHz official zips), resampled to 16 kHz.
Fallback: Hugging Face `JacobLinCool/VoiceBank-DEMAND-16k` (already 16 kHz).

Audio is stored under training/data/raw/voicebank_demand/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import SAMPLE_RATE
from training.audio_io import load_mono, write_wav
from training.paths import repo_rel

EDINBURGH_PREFIX = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/"
OFFICIAL_ZIPS = [
    "clean_testset_wav.zip",
    "noisy_testset_wav.zip",
    "clean_trainset_28spk_wav.zip",
    "noisy_trainset_28spk_wav.zip",
]
PAIR_DIRS = [
    ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav", "train"),
    ("clean_testset_wav", "noisy_testset_wav", "test"),
]
HF_DATASET = "JacobLinCool/VoiceBank-DEMAND-16k"


def _download(url: str, dest: Path, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"exists {dest.name} ({dest.stat().st_size} bytes)")
        return
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ul-unas-finetune/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)
    print(f"saved {dest} ({dest.stat().st_size} bytes)")


def _safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract zip members, rejecting path traversal (Zip Slip)."""
    print(f"extracting {zip_path.name}")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name or name.startswith("/") or name.startswith("\\"):
                raise ValueError(f"unsafe zip member {name!r} in {zip_path.name}")
            dest = (out_dir / name).resolve()
            if dest != out_dir and not str(dest).startswith(str(out_dir) + os.sep):
                raise ValueError(f"Zip Slip blocked member {name!r} in {zip_path.name}")
            zf.extract(info, out_dir)


def _extract(zip_path: Path, out_dir: Path) -> None:
    _safe_extract(zip_path, out_dir)


def _resample_one(src: str, dst: str, sr: int) -> str:
    audio, _ = load_mono(src, target_sr=sr)
    write_wav(dst, audio, sr)
    return dst


def resample_tree(src_dir: Path, dst_dir: Path, sr: int, workers: int) -> int:
    wavs = sorted(src_dir.rglob("*.wav"))
    if not wavs:
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for wav in wavs:
        dest = dst_dir / wav.name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        jobs.append((str(wav), str(dest), sr))
    if not jobs:
        return len(wavs)
    n_ok = len(wavs) - len(jobs)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_resample_one, *job) for job in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            n_ok += 1
            if i % 200 == 0 or i == len(futs):
                print(f"  resampled {i}/{len(futs)} from {src_dir.name}")
    return n_ok


def write_license(out_root: Path, source: str) -> None:
    text = f"""# VoiceBank+DEMAND (Valentini) — local copy

- **What**: 28-speaker train / 2-speaker test paired noisy-clean speech for 16 kHz speech enhancement.
- **Official 48 kHz source**: Cassia Valentini-Botinhao, University of Edinburgh CSTR.
  https://doi.org/10.7488/ds/2117
  https://datashare.ed.ac.uk/handle/10283/2791
- **This copy**: resampled to 16 kHz for UL-UNAS. Source used: `{source}`.
- **Speech**: CSTR VCTK Corpus — CC BY 4.0 (https://doi.org/10.7488/ds/1994).
- **Noise**: DEMAND — CC BY-SA 3.0 (https://zenodo.org/records/1227121).
- **Edinburgh DataShare item licence**: End-user Licence (see the handle page).
- **16 kHz Hugging Face mirror** (opt-in `--allow_hf_fallback` only): `JacobLinCool/VoiceBank-DEMAND-16k`, declared CC BY 4.0. Prefer the official DataShare zips.
- **Attribution**: Valentini-Botinhao, C. (2017). Noisy speech database for training speech enhancement algorithms and TTS models, 2016 [sound]. University of Edinburgh.

This checkout uses the data for a **research fine-tune** of UL-UNAS. Do not treat the resulting checkpoint as a commercially cleared product model.
"""
    (out_root / "LICENSE.md").write_text(text, encoding="utf-8")


def download_official(raw_root: Path, timeout: int) -> Path:
    zip_dir = raw_root / "zips"
    extract_dir = raw_root / "48k"
    zip_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_ZIPS:
        url = EDINBURGH_PREFIX + name
        dest = zip_dir / name
        try:
            _download(url, dest, timeout=timeout)
        except Exception as exc:
            alt = url + "?sequence=1&isAllowed=y"
            print(f"retry {name} via {alt}: {exc}")
            _download(alt, dest, timeout=timeout)
        marker = extract_dir / name.replace(".zip", "")
        if not marker.exists() or not any(marker.rglob("*.wav")):
            _extract(dest, extract_dir)
    return extract_dir


def layout_from_48k(extract_dir: Path, out16: Path, workers: int) -> dict:
    counts = {}
    for clean_name, noisy_name, split in PAIR_DIRS:
        clean_src = _find_dir(extract_dir, clean_name)
        noisy_src = _find_dir(extract_dir, noisy_name)
        if clean_src is None or noisy_src is None:
            raise FileNotFoundError(f"missing {clean_name} or {noisy_name} under {extract_dir}")
        n_c = resample_tree(clean_src, out16 / split / "clean", SAMPLE_RATE, workers)
        n_n = resample_tree(noisy_src, out16 / split / "noisy", SAMPLE_RATE, workers)
        counts[split] = {"clean": n_c, "noisy": n_n}
    return counts


def _find_dir(root: Path, name: str) -> Path | None:
    if (root / name).is_dir():
        return root / name
    matches = [p for p in root.rglob("*") if p.is_dir() and p.name == name]
    return matches[0] if matches else None


def download_hf(out16: Path) -> dict:
    from huggingface_hub import snapshot_download

    print(f"falling back to Hugging Face {HF_DATASET}")
    cache = snapshot_download(repo_id=HF_DATASET, repo_type="dataset")
    try:
        from datasets import load_dataset
    except ImportError:
        # datasets is optional; parquet-to-wav via huggingface_hub + soundfile if parquet exists
        return _export_hf_snapshot(Path(cache), out16)
    ds = load_dataset(HF_DATASET)
    counts = {}
    for split, table in ds.items():
        mapped = "train" if split.startswith("train") else "test" if "test" in split else split
        clean_dir = out16 / mapped / "clean"
        noisy_dir = out16 / mapped / "noisy"
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for row in table:
            utt = row["id"]
            write_wav(clean_dir / f"{utt}.wav", row["clean"]["array"], SAMPLE_RATE)
            write_wav(noisy_dir / f"{utt}.wav", row["noisy"]["array"], SAMPLE_RATE)
            n += 1
            if n % 500 == 0:
                print(f"  wrote {n} {mapped} pairs from HF")
        counts[mapped] = {"pairs": n}
    return counts


def _export_hf_snapshot(cache: Path, out16: Path) -> dict:
    """Load HF parquet audio without the datasets package if needed."""
    import numpy as np
    import soundfile as sf

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("Install `datasets` or `pyarrow` to export the HF fallback")

    counts: dict[str, dict[str, int]] = {}
    for parquet in sorted(cache.rglob("*.parquet")):
        split = "train" if "train" in parquet.name else "test"
        table = pq.read_table(parquet).to_pydict()
        ids = table.get("id") or table.get("ID")
        n = 0
        for i, utt in enumerate(ids):
            for key, sub in (("clean", "clean"), ("noisy", "noisy")):
                rec = table[key][i]
                arr = rec["array"] if isinstance(rec, dict) else rec
                sr = rec.get("sampling_rate", SAMPLE_RATE) if isinstance(rec, dict) else SAMPLE_RATE
                dest = out16 / split / sub / f"{utt}.wav"
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_wav(dest, np.asarray(arr, dtype=np.float32), int(sr) or SAMPLE_RATE)
            n += 1
        counts.setdefault(split, {"pairs": 0})
        counts[split]["pairs"] += n
    return counts


def run(args: argparse.Namespace) -> dict:
    raw_root = Path(args.output_dir)
    out16 = raw_root / "16k"
    out16.mkdir(parents=True, exist_ok=True)
    source = "edinburgh_datashare_48k_resampled_16k"
    counts: dict
    try:
        extract_dir = download_official(raw_root, timeout=args.timeout)
        counts = layout_from_48k(extract_dir, out16, workers=args.workers)
    except Exception as exc:
        print(f"official download failed: {exc}")
        if not args.allow_hf_fallback:
            raise
        source = f"huggingface:{HF_DATASET}"
        counts = download_hf(out16)
    write_license(raw_root, source)
    report = {
        "source": source,
        "output_dir": repo_rel(raw_root),
        "sr": SAMPLE_RATE,
        "counts": counts,
        "doi": "10.7488/ds/2117",
    }
    (raw_root / "download_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    ncpu = os.cpu_count() or 2
    p = argparse.ArgumentParser(description="Download VoiceBank+DEMAND and resample to 16 kHz")
    p.add_argument("--output_dir", default="training/data/raw/voicebank_demand")
    p.add_argument("--workers", type=int, default=max(1, ncpu - 1))
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument(
        "--allow_hf_fallback",
        action="store_true",
        default=False,
        help="Opt-in Hugging Face 16 kHz mirror if Edinburgh DataShare fails. Official DataShare End-user Licence still applies to the content.",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
