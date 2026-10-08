"""Optional MS-SNSD helper: scripts, layout, hour-capped synth. No full-corpus grab.

Official source: https://github.com/microsoft/MS-SNSD
- Scripts (MIT): noisyspeech_synthesizer.py + cfg
- Clean speech: Edinburgh / PTDB (ODbL) — see their README
- Noise: CC0 + DEMAND BY-SA 3.0
- Native 16 kHz. Official synth writes NoisySpeech_training / CleanSpeech_training
  (`noisyN_SNRdb_*_clnspN.wav` ↔ `clnspN.wav`). Test is pre-mixed clean_test/noisy_test.

This helper never downloads the multi-GB CleanSpeech/Noise dumps. Point it at a
local checkout, or run the in-repo mixer (`synthesize_mssnsd.py`) from
`audio/clean` + residual noise for a smoke (≤0.5–2 h).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.paired_layout import find_mssnsd_roots, layout_pairs_to_16k, match_pairs, normalize_split
from training.paths import repo_rel

MSSNSD_GIT = "https://github.com/microsoft/MS-SNSD.git"
STEPS = """
# MS-SNSD — local steps (do not pull DNS / URGENT)

1. Scripts only (small):
   python training/download_mssnsd.py --fetch_scripts

2. Place official 16 kHz sources under training/data/raw/mssnsd/:
     clean_train/ or CleanSpeech/   (clean speech)
     noise_train/ or Noise/         (noise, not pre-mixed)
     clean_test/ + noisy_test/      (optional official test pairs)

   Official download notes: https://github.com/microsoft/MS-SNSD
   Community mirrors exist (Kaggle/DagsHub); prefer the Microsoft README.

3. Either run the official synthesizer (total_hours in noisyspeech_synthesizer.cfg)
   or the in-repo mixer (recommended for smoke, same-stem layout):

   python training/synthesize_mssnsd.py --hours 0.5 --max_hours 2 \\
     --source_root training/data/raw/mssnsd \\
     --output_dir training/data/processed/mssnsd/16k

4. If you already have official synth dirs, flatten them:

   python training/download_mssnsd.py --layout_from /path/to/MS-SNSD

5. Index (does not change --mode=auto):
   python training/prepare_manifest.py
"""


def write_license(out_root: Path) -> None:
    text = """# MS-SNSD — local copy / derived pairs

- **What**: Microsoft Scalable Noisy Speech Dataset. Native 16 kHz. Train pairs
  are synthesized; `clean_test` / `noisy_test` are pre-mixed.
- **Code**: MIT — https://github.com/microsoft/MS-SNSD
- **Paper**: Reddy et al., Interspeech 2019.
- **Clean speech**: Edinburgh / PTDB-TUG (ODbL). Confirm before redistribution.
- **Noise**: CC0 plus DEMAND (CC BY-SA 3.0).
- **This repo**: research fine-tune only; not a commercially cleared product model.

In-repo `synthesize_mssnsd.py` smokes can use `audio/clean` + residual noise and
are **not** the official MS-SNSD corpus. Label those manifests `source_note` in
`synth_report.json`.
"""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "LICENSE.md").write_text(text, encoding="utf-8")


def fetch_scripts(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "noisyspeech_synthesizer.py"
    if marker.is_file():
        print(f"scripts already present under {dest}")
        return dest
    clone = dest / "_ms_snsd_git"
    if clone.exists():
        shutil.rmtree(clone)
    print(f"cloning {MSSNSD_GIT} (scripts only, depth=1)")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", MSSNSD_GIT, str(clone)],
        check=True,
    )
    for name in (
        "noisyspeech_synthesizer.py",
        "noisyspeech_synthesizer.cfg",
        "audiolib.py",
        "README.md",
        "LICENSE",
    ):
        src = clone / name
        if src.exists():
            shutil.copy2(src, dest / name)
    return dest


def layout_from(src: Path, dest: Path, copy_audio: bool) -> dict:
    roots = find_mssnsd_roots(src)
    counts: dict[str, int] = {}
    for clean_dir, noisy_dir, split in roots:
        pairs = match_pairs(clean_dir, noisy_dir)
        layout_pairs_to_16k(pairs, dest / "16k" if dest.name != "16k" else dest, split, copy_audio=copy_audio)
        counts[normalize_split(split)] = counts.get(normalize_split(split), 0) + len(pairs)
    return counts


def run(args: argparse.Namespace) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_license(out)
    report: dict = {
        "dataset": "MS-SNSD",
        "output_dir": repo_rel(out),
        "steps": STEPS.strip(),
        "downloaded_full_corpus": False,
    }
    if args.fetch_scripts:
        scripts = fetch_scripts(out / "scripts")
        report["scripts"] = repo_rel(scripts)
    if args.layout_from:
        counts = layout_from(Path(args.layout_from), out, copy_audio=not args.symlink)
        report["layout_counts"] = counts
    if args.synth:
        from training.synthesize_mssnsd import build_parser as synth_parser
        from training.synthesize_mssnsd import synthesize

        synth_args = synth_parser().parse_args(
            [
                "--repo",
                args.repo,
                "--source_root",
                args.layout_from or str(out),
                "--output_dir",
                str(out / "16k"),
                "--hours",
                str(args.hours),
                "--max_hours",
                str(args.max_hours),
            ]
            + (["--allow_large"] if args.allow_large else [])
        )
        report["synth"] = synthesize(synth_args)
    (out / "download_report.json").write_text(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2), encoding="utf-8")
    print(STEPS)
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MS-SNSD layout / scripts helper (no full-corpus download)")
    p.add_argument("--repo", default=str(ROOT))
    p.add_argument("--output_dir", default="training/data/raw/mssnsd")
    p.add_argument("--fetch_scripts", action="store_true", help="Clone microsoft/MS-SNSD scripts only")
    p.add_argument("--layout_from", default="", help="Existing MS-SNSD checkout to flatten into 16k/{split}/")
    p.add_argument("--symlink", action="store_true", help="Symlink instead of resample/copy when laying out")
    p.add_argument("--synth", action="store_true", help="Run synthesize_mssnsd.py after layout/sources")
    p.add_argument("--hours", type=float, default=0.5)
    p.add_argument("--max_hours", type=float, default=2.0)
    p.add_argument("--allow_large", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
