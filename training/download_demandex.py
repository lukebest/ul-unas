"""VB-DemandEx layout adapter (+ optional HF zip download).

HF card: `NikolaiKyhne/VB-DemandEx` (pretty_name VB-DemandEx), license MIT on
the card; underlying speech/noise still VoiceBank+DEMAND (VCTK CC BY 4.0 +
DEMAND BY-SA 3.0 + DataShare End-user). ~1.94 GB usedStorage.

Zips: clean_{train,valid,test}.zip + noisy_{train,valid,test}.zip
Mapped to `16k/{train,dev,test}/{clean,noisy}` so generic ingest works.

Does **not** register these dirs with `find_vbdemand_roots` (Valentini-only).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import SAMPLE_RATE
from training.audio_io import load_mono, write_wav
from training.paired_layout import find_demandex_roots, match_pairs, normalize_split
from training.paths import repo_rel

HF_REPO = "NikolaiKyhne/VB-DemandEx"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
ZIPS = (
    "clean_train.zip",
    "noisy_train.zip",
    "clean_valid.zip",
    "noisy_valid.zip",
    "clean_test.zip",
    "noisy_test.zip",
)
STEPS = """
# VB-DemandEx — local steps

1. Optional download (~1.94 GB; skip on smoke boxes):
   python training/download_demandex.py --download

2. Or place the six HF zips / extracted clean_{train,valid,test} +
   noisy_{train,valid,test} dirs under training/data/raw/vbdemandex/

3. Layout into 16 kHz split trees (valid → dev):
   python training/download_demandex.py --layout
   # or: python training/download_demandex.py --layout_from /path/to/extract

4. Index (separate JSONLs; VB-DEMAND writers untouched):
   python training/prepare_manifest.py

HF: https://huggingface.co/datasets/NikolaiKyhne/VB-DemandEx
AAU: https://vbn.aau.dk/en/datasets/vb-demandex/
Paper: xLSTM-SENet, arXiv:2501.06146
"""


def _download(url: str, dest: Path, timeout: int) -> None:
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


def write_license(out_root: Path) -> None:
    text = f"""# VB-DemandEx — local copy

- **What**: Pre-mixed VoiceBank-style pairs with extended SNR (about −10 to 20 dB).
- **HF card**: `{HF_REPO}` — pretty_name VB-DemandEx; card license **MIT**.
- **DOI / paper**: https://arxiv.org/abs/2501.06146 (xLSTM-SENet).
- **Underlying audio**: still VoiceBank+DEMAND — VCTK CC BY 4.0, DEMAND CC BY-SA 3.0,
  Edinburgh DataShare End-user Licence (https://doi.org/10.7488/ds/2117).
- **This checkout**: resampled to 16 kHz under `16k/{{train,dev,test}}/{{clean,noisy}}`.
  `valid` maps to `dev`. These folders are **not** Valentini aliases and are
  invisible to `find_vbdemand_roots`.

Research use for UL-UNAS domain-expand experiments. Not a commercially cleared
product model. Attribute both the DemandEx card and Valentini/VCTK/DEMAND.
"""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "LICENSE.md").write_text(text, encoding="utf-8")


def download_zips(raw_root: Path, timeout: int) -> Path:
    zip_dir = raw_root / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    for name in ZIPS:
        _download(HF_BASE + name, zip_dir / name, timeout=timeout)
    return zip_dir


def extract_zips(raw_root: Path) -> Path:
    extract_dir = raw_root / "extract"
    zip_dir = raw_root / "zips"
    extract_dir.mkdir(parents=True, exist_ok=True)
    for name in ZIPS:
        zp = zip_dir / name
        if not zp.is_file():
            continue
        marker = extract_dir / name.replace(".zip", "")
        if marker.is_dir() and any(marker.rglob("*.wav")):
            continue
        _safe_extract(zp, extract_dir)
    return extract_dir


def layout_from(src: Path, dest16: Path, max_pairs: int, workers_unused: int = 0) -> dict:
    del workers_unused
    roots = find_demandex_roots(src)
    if not roots:
        # Also accept a folder that *is* the extract root with clean_train at top.
        roots = find_demandex_roots(src.parent) if src.name == "extract" else []
    counts: dict[str, dict[str, int]] = {}
    dest16.mkdir(parents=True, exist_ok=True)
    for clean_dir, noisy_dir, split in roots:
        split = normalize_split(split)
        pairs = match_pairs(clean_dir, noisy_dir)
        if max_pairs and max_pairs > 0:
            pairs = pairs[:max_pairs]
        n = 0
        for utt, noisy, clean in pairs:
            mix, _ = load_mono(noisy)
            tgt, _ = load_mono(clean)
            m = min(len(mix), len(tgt))
            write_wav(dest16 / split / "noisy" / f"{utt}.wav", mix[:m], SAMPLE_RATE)
            write_wav(dest16 / split / "clean" / f"{utt}.wav", tgt[:m], SAMPLE_RATE)
            n += 1
        counts[split] = {"pairs": n, "src_clean": repo_rel(clean_dir), "src_noisy": repo_rel(noisy_dir)}
    return counts


def run(args: argparse.Namespace) -> dict:
    raw_root = Path(args.output_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    write_license(raw_root)
    report: dict = {
        "dataset": "VB-DemandEx",
        "hf": HF_REPO,
        "output_dir": repo_rel(raw_root),
        "sr": SAMPLE_RATE,
        "downloaded": False,
        "find_vbdemand_roots": "unchanged (Valentini-only)",
    }
    src = Path(args.layout_from) if args.layout_from else raw_root
    if args.download:
        download_zips(raw_root, timeout=args.timeout)
        extract_zips(raw_root)
        src = raw_root / "extract"
        report["downloaded"] = True
    elif (raw_root / "zips").exists() and any((raw_root / "zips").glob("*.zip")):
        extract_zips(raw_root)
        src = raw_root / "extract"
    do_layout = args.layout or args.layout_from or args.download
    if do_layout:
        counts = layout_from(src, raw_root / "16k", max_pairs=args.max_pairs)
        report["layout_counts"] = counts
    report["steps"] = STEPS.strip()
    (raw_root / "download_report.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2),
        encoding="utf-8",
    )
    print(STEPS)
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VB-DemandEx layout adapter (optional HF download)")
    p.add_argument("--output_dir", default="training/data/raw/vbdemandex")
    p.add_argument("--layout_from", default="", help="Existing extract / zip-name tree")
    p.add_argument("--layout", action="store_true", help="Layout from --output_dir extract or zip-name dirs")
    p.add_argument("--download", action="store_true", help="Fetch the six HF zips (~1.94 GB). Skip on smoke.")
    p.add_argument("--max_pairs", type=int, default=0, help="Per-split cap after pairing (0 = all)")
    p.add_argument("--timeout", type=int, default=1200)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
