"""Generic paired clean/noisy layout discovery for public SE sets.

Extends the dual-writer catalog without replacing VoiceBank+DEMAND:

- `prepare_manifest.scan_vbdemand` / `find_vbdemand_roots` stay Valentini-only.
- `synthesize_pairs.ingest_public_pairs` stays the VB-DMD wav-same-name writer.
- Three extra finders (keep all three):
  1. `find_mssnsd_roots` — `DATASET_SPECS` id `mssnsd`
  2. `find_demandex_roots` — `DATASET_SPECS` id `vbdemandex`
  3. `find_generic_paired_roots` — explicit public API for other
     `{split}/{clean,noisy}` trees. **Not** wired into `DATASET_SPECS`;
     callers use it directly (see `test_generic_split_tree`).

`--mode=auto` is unchanged: it still keys only on `find_vbdemand_roots`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".ogg"}

SPLIT_ALIAS = {
    "train": "train",
    "training": "train",
    "dev": "dev",
    "valid": "dev",
    "validation": "dev",
    "val": "dev",
    "test": "test",
    "eval": "test",
}

# Official Valentini folder names — never treat these as DemandEx / generic extras.
VALENTINI_DIR_NAMES = {
    "clean_trainset_28spk_wav",
    "noisy_trainset_28spk_wav",
    "clean_testset_wav",
    "noisy_testset_wav",
    "clean_trainset_28spk_wav_16k",
    "noisy_trainset_28spk_wav_16k",
    "clean_testset_wav_16k",
    "noisy_testset_wav_16k",
}

VBDEMAND_PATH_MARKERS = ("voicebank_demand", "valentini")

# HF NikolaiKyhne/VB-DemandEx zip member names (valid → dev).
DEMANDEX_NAMED = (
    ("clean_train", "noisy_train", "train"),
    ("clean_valid", "noisy_valid", "dev"),
    ("clean_validation", "noisy_validation", "dev"),
    ("clean_test", "noisy_test", "test"),
)

# Official MS-SNSD synthesizer output + pre-mixed test.
MSSNSD_NAMED = (
    ("CleanSpeech_training", "NoisySpeech_training", "train"),
    ("cleanspeech_training", "noisyspeech_training", "train"),
)

MSSNSD_PATH_MARKERS = ("mssnsd", "ms-snsd", "ms_snsd", "snsd")
DEMANDEX_PATH_MARKERS = ("demandex", "vbdemandex", "vb-demandex", "vb_demandex")

CLNSP_IN_NOISY = re.compile(r"(clnsp\d+)", re.IGNORECASE)
NOISY_INDEX = re.compile(r"^noisy(\d+)(?:_|$)", re.IGNORECASE)

# Auto-ingest ids only. `find_generic_paired_roots` is intentionally omitted
# (third public API; not an mssnsd/demandex finder).
DATASET_SPECS = (
    {
        "id": "mssnsd",
        "layer": "mssnsd",
        "source": "MS-SNSD",
        "dir_names": ("mssnsd", "MS-SNSD", "ms_snsd", "ms-snsd"),
        "finder": "mssnsd",
    },
    {
        "id": "vbdemandex",
        "layer": "vbdemandex",
        "source": "VB-DemandEx",
        "dir_names": ("vbdemandex", "VB-DemandEx", "vb_demandex", "vb-demandex"),
        "finder": "demandex",
    },
)


def iter_audio(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


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
    if not rows:
        existing = jsonl_row_count(path)
        if existing:
            print(f"skip empty write; keep existing {path} ({existing} rows)")
            return existing
    write_jsonl(path, rows)
    return len(rows)


def normalize_split(hint: str) -> str:
    key = hint.lower().strip()
    if key not in SPLIT_ALIAS:
        raise ValueError(f"unknown split hint {hint!r}")
    return SPLIT_ALIAS[key]


def _path_has_marker(path: Path, markers: tuple[str, ...]) -> bool:
    parts = [p.lower() for p in path.parts]
    return any(any(m in part for m in markers) for part in parts)


def is_vbdemand_path(path: Path) -> bool:
    if _path_has_marker(path, VBDEMAND_PATH_MARKERS):
        return True
    return path.name in VALENTINI_DIR_NAMES


def _unique_roots(rows: list[tuple[Path, Path, str]]) -> list[tuple[Path, Path, str]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[Path, Path, str]] = []
    for clean_dir, noisy_dir, split in rows:
        key = (str(clean_dir.resolve()), str(noisy_dir.resolve()), split)
        if key in seen:
            continue
        seen.add(key)
        out.append((clean_dir, noisy_dir, split))
    return out


def pair_clean_stem(noisy_stem: str, clean_stems: set[str]) -> str | None:
    """Match a noisy file to a clean stem.

    Handles:
    - identical names (DemandEx / flattened ingest)
    - official MS-SNSD synth: noisyN_SNRdb_*_clnspN.wav ↔ clnspN.wav
    """
    if noisy_stem in clean_stems:
        return noisy_stem
    m = CLNSP_IN_NOISY.search(noisy_stem)
    if m:
        hit = m.group(1)
        if hit in clean_stems:
            return hit
        lower = {c.lower(): c for c in clean_stems}
        if hit.lower() in lower:
            return lower[hit.lower()]
    m = NOISY_INDEX.match(noisy_stem)
    if m:
        idx = m.group(1)
        for cand in (f"clnsp{idx}", f"clean{idx}", f"clnsp{idx}.wav"):
            if cand in clean_stems:
                return cand
            for stem in clean_stems:
                if stem.lower() == cand.lower():
                    return stem
    if noisy_stem.lower().startswith("noisy_"):
        stripped = noisy_stem[6:]
        if stripped in clean_stems:
            return stripped
        prefixed = f"clean_{stripped}"
        if prefixed in clean_stems:
            return prefixed
    return None


def match_pairs(clean_dir: Path, noisy_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return (utt, noisy, clean) for files that can be paired."""
    clean_files = {p.stem: p for p in iter_audio(clean_dir)}
    if not clean_files:
        return []
    stems = set(clean_files)
    pairs: list[tuple[str, Path, Path]] = []
    seen: set[str] = set()
    for noisy in iter_audio(noisy_dir):
        stem = pair_clean_stem(noisy.stem, stems)
        if stem is None:
            matches = [p for p in clean_dir.rglob(noisy.name) if p.is_file()]
            if matches:
                stem = matches[0].stem
                clean_files[stem] = matches[0]
            else:
                continue
        clean = clean_files[stem]
        utt = noisy.stem
        if utt in seen:
            continue
        seen.add(utt)
        pairs.append((utt, noisy, clean))
    pairs.sort(key=lambda r: r[0])
    return pairs


def _find_dir_named(root: Path, name: str) -> list[Path]:
    """Find directories named `name` (case-insensitive). Dir-only walk, no file rglob."""
    if not root.is_dir():
        return []
    found: list[Path] = []
    target = name.lower()
    # Known shallow joins first (dataset root and 16k layout).
    for cand in (root / name, root / "16k" / name):
        if cand.is_dir() and cand not in found:
            found.append(cand)
    for dirpath, dirnames, _files in os.walk(root, followlinks=False):
        for d in dirnames:
            if d.lower() != target:
                continue
            p = Path(dirpath) / d
            if p not in found:
                found.append(p)
    return found


def find_named_pair_dirs(root: Path, clean_name: str, noisy_name: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for clean_dir in _find_dir_named(root, clean_name):
        if is_vbdemand_path(clean_dir):
            continue
        sibling = clean_dir.parent / noisy_name
        if sibling.is_dir():
            pairs.append((clean_dir, sibling))
            continue
        for noisy_dir in _find_dir_named(clean_dir.parent, noisy_name):
            if not is_vbdemand_path(noisy_dir):
                pairs.append((clean_dir, noisy_dir))
                break
    return pairs


def find_split_tree_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    """Find `{split}/{clean,noisy}` and `{root}/16k/{split}/{clean,noisy}`."""
    found: list[tuple[Path, Path, str]] = []
    if not root.exists():
        return found
    candidates = [root]
    sixteen = root / "16k"
    if sixteen.is_dir():
        candidates.append(sixteen)
    for base in candidates:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            try:
                split = normalize_split(child.name)
            except ValueError:
                continue
            clean_dir = child / "clean"
            noisy_dir = child / "noisy"
            if clean_dir.is_dir() and noisy_dir.is_dir() and not is_vbdemand_path(clean_dir):
                found.append((clean_dir, noisy_dir, split))
    return found


def _skip_other_dataset(path: Path, own_markers: tuple[str, ...], other_markers: tuple[str, ...]) -> bool:
    """Drop dirs that live under a sibling dataset namespace."""
    if _path_has_marker(path, own_markers):
        return False
    return _path_has_marker(path, other_markers)


def find_mssnsd_roots(root: Path) -> list[tuple[Path, Path, str]]:
    """MS-SNSD native synth dirs, official test, or flattened 16 kHz tree."""
    found = find_split_tree_pairs(root)
    for clean_name, noisy_name, split in MSSNSD_NAMED:
        for clean_dir, noisy_dir in find_named_pair_dirs(root, clean_name, noisy_name):
            if _skip_other_dataset(clean_dir, MSSNSD_PATH_MARKERS, DEMANDEX_PATH_MARKERS):
                continue
            found.append((clean_dir, noisy_dir, split))
    # Official pre-mixed test. Accept at this root when this is an MS-SNSD tree
    # or when CleanSpeech_training / noise_train sit alongside.
    mssnsd_ctx = _path_has_marker(root, MSSNSD_PATH_MARKERS) or any(
        (root / name).is_dir()
        for name in ("CleanSpeech_training", "clean_train", "noise_train", "Noise", "CleanSpeech")
    )
    if mssnsd_ctx:
        for clean_dir, noisy_dir in find_named_pair_dirs(root, "clean_test", "noisy_test"):
            if is_vbdemand_path(clean_dir):
                continue
            if _skip_other_dataset(clean_dir, MSSNSD_PATH_MARKERS, DEMANDEX_PATH_MARKERS):
                continue
            found.append((clean_dir, noisy_dir, "test"))
    found = [
        (c, n, s)
        for c, n, s in found
        if not _skip_other_dataset(c, MSSNSD_PATH_MARKERS, DEMANDEX_PATH_MARKERS)
    ]
    return _unique_roots(found)


def find_demandex_roots(root: Path) -> list[tuple[Path, Path, str]]:
    """VB-DemandEx zip names (`clean_train` …) or flattened 16 kHz tree.

    Does not match Valentini `clean_trainset_28spk_wav` aliases.
    """
    found = find_split_tree_pairs(root)
    for clean_name, noisy_name, split in DEMANDEX_NAMED:
        if clean_name in VALENTINI_DIR_NAMES or noisy_name in VALENTINI_DIR_NAMES:
            continue
        for clean_dir, noisy_dir in find_named_pair_dirs(root, clean_name, noisy_name):
            if is_vbdemand_path(clean_dir) or is_vbdemand_path(noisy_dir):
                continue
            if _skip_other_dataset(clean_dir, DEMANDEX_PATH_MARKERS, MSSNSD_PATH_MARKERS):
                continue
            found.append((clean_dir, noisy_dir, split))
    found = [
        (c, n, s)
        for c, n, s in found
        if not is_vbdemand_path(c) and not _skip_other_dataset(c, DEMANDEX_PATH_MARKERS, MSSNSD_PATH_MARKERS)
    ]
    return _unique_roots(found)


def find_generic_paired_roots(root: Path) -> list[tuple[Path, Path, str]]:
    """Third corpus entry: any `{split}/{clean,noisy}` tree that is not Valentini.

    Public API for other paired layouts (Libri+DEMAND self-mix, flattened
    challenge exports, …). `DATASET_SPECS` does **not** call this — mssnsd
    and demandex use `find_mssnsd_roots` / `find_demandex_roots` only.
    Keep this function and `test_generic_split_tree`.
    """
    return _unique_roots(find_split_tree_pairs(root))


def _finder_for(kind: str):
    if kind == "mssnsd":
        return find_mssnsd_roots
    if kind == "demandex":
        return find_demandex_roots
    return find_generic_paired_roots


def extra_dataset_roots(
    public_root: Path,
    repo: Path,
    include_smoke: bool = False,
) -> dict[str, list[Path]]:
    """Named dataset dirs under raw/ and processed/, plus public_root itself.

    `training/data/smoke/` is **not** always appended. Include it only when
    `public_root` *is* the smoke tree, or when `include_smoke=True`. Default
    ingest (`public_root=training/data/raw`) must not auto-ingest leftover
    smoke into canonical `{train,dev,eval}_{mssnsd,vbdemandex}.jsonl`.
    """
    smoke = repo / "training" / "data" / "smoke"
    bases = [
        public_root,
        repo / "training" / "data" / "raw",
        repo / "training" / "data" / "processed",
    ]
    if include_smoke:
        bases.append(smoke)
    found: dict[str, list[Path]] = {spec["id"]: [] for spec in DATASET_SPECS}
    for spec in DATASET_SPECS:
        ds = spec["id"]
        for base in bases:
            if not base.exists():
                continue
            if spec["finder"] == "mssnsd" and _path_has_marker(base, MSSNSD_PATH_MARKERS):
                found[ds].append(base)
            if spec["finder"] == "demandex" and _path_has_marker(base, DEMANDEX_PATH_MARKERS):
                found[ds].append(base)
            for name in spec["dir_names"]:
                cand = base / name
                if cand.is_dir():
                    found[ds].append(cand)
    return {k: _dedupe_paths(v) for k, v in found.items()}


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def scan_paired_roots(
    roots: list[tuple[Path, Path, str]],
    repo: Path,
    layer: str,
    source: str,
) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "eval": []}
    seen: set[str] = set()
    for clean_dir, noisy_dir, split_hint in roots:
        split_norm = normalize_split(split_hint)
        for utt, noisy, clean in match_pairs(clean_dir, noisy_dir):
            key = f"{split_norm}:{utt}"
            if key in seen:
                continue
            seen.add(key)
            if split_norm == "test":
                split, jsonl_split, held_out = "test", "eval", True
            elif split_norm == "dev":
                split, jsonl_split, held_out = "dev", "dev", False
            else:
                split, jsonl_split, held_out = "train", "train", False
            # Lazy import avoids a cycle with prepare_manifest.
            from training.prepare_manifest import paired_row

            buckets[jsonl_split].append(
                paired_row(
                    utt,
                    noisy,
                    clean,
                    split,
                    repo,
                    layer=layer,
                    held_out=held_out,
                    source=source,
                )
            )
    for split in buckets:
        buckets[split].sort(key=lambda r: r["id"])
    return buckets


def scan_extra_paired(public_root: Path, repo: Path) -> dict[str, dict[str, list[dict]]]:
    """Index MS-SNSD / VB-DemandEx (and generic split trees under those names)."""
    out: dict[str, dict[str, list[dict]]] = {}
    named = extra_dataset_roots(public_root, repo)
    for spec in DATASET_SPECS:
        finder = _finder_for(spec["finder"])
        roots: list[tuple[Path, Path, str]] = []
        for base in named[spec["id"]]:
            roots.extend(finder(base))
        # Probe public_root only when it itself looks like this dataset
        # (namespaced dirs are already in `named`).
        if public_root.exists() and public_root not in named[spec["id"]]:
            if spec["finder"] == "mssnsd" and (
                _path_has_marker(public_root, MSSNSD_PATH_MARKERS)
                or any(
                    (public_root / n).is_dir()
                    for n in ("CleanSpeech_training", "NoisySpeech_training", "noise_train")
                )
            ):
                roots.extend(finder(public_root))
            elif spec["finder"] == "demandex" and (
                _path_has_marker(public_root, DEMANDEX_PATH_MARKERS)
                or any((public_root / n).is_dir() for n in ("clean_train", "clean_valid", "noisy_valid"))
            ):
                roots.extend(finder(public_root))
        roots = _unique_roots([r for r in roots if not is_vbdemand_path(r[0])])
        if not roots:
            continue
        buckets = scan_paired_roots(roots, repo, layer=spec["layer"], source=spec["source"])
        if any(buckets[s] for s in buckets):
            out[spec["id"]] = buckets
    return out


def write_extra_manifests(
    extras: dict[str, dict[str, list[dict]]],
    out_dir: Path,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ds, buckets in extras.items():
        counts[f"train_{ds}"] = write_jsonl_skip_empty(out_dir / f"train_{ds}.jsonl", buckets["train"])
        counts[f"dev_{ds}"] = write_jsonl_skip_empty(out_dir / f"dev_{ds}.jsonl", buckets["dev"])
        counts[f"eval_{ds}"] = write_jsonl_skip_empty(out_dir / f"eval_{ds}.jsonl", buckets["eval"])
        all_rows = buckets["train"] + buckets["dev"] + buckets["eval"]
        if all_rows:
            write_jsonl(out_dir / f"paired_{ds}.jsonl", all_rows)
    return counts


def layout_pairs_to_16k(
    pairs: list[tuple[str, Path, Path]],
    dest_root: Path,
    split: str,
    copy_audio: bool = True,
) -> Path:
    """Write (or symlink) pairs into `dest_root/{split}/{clean,noisy}/`."""
    from training import SAMPLE_RATE
    from training.audio_io import load_mono, write_wav

    split = normalize_split(split)
    clean_out = dest_root / split / "clean"
    noisy_out = dest_root / split / "noisy"
    clean_out.mkdir(parents=True, exist_ok=True)
    noisy_out.mkdir(parents=True, exist_ok=True)
    for utt, noisy, clean in pairs:
        if copy_audio:
            mix, _ = load_mono(noisy)
            tgt, _ = load_mono(clean)
            n = min(len(mix), len(tgt))
            write_wav(noisy_out / f"{utt}.wav", mix[:n], SAMPLE_RATE)
            write_wav(clean_out / f"{utt}.wav", tgt[:n], SAMPLE_RATE)
        else:
            for src, dest_dir in ((noisy, noisy_out), (clean, clean_out)):
                dest = dest_dir / src.name
                if dest.exists():
                    continue
                dest.symlink_to(src.resolve())
    return dest_root


def unresolved_clean(rows: list[dict], repo: Path) -> list[str]:
    missing: list[str] = []
    for row in rows:
        clean = row.get("clean")
        if not clean:
            missing.append(str(row.get("id")))
            continue
        p = Path(clean)
        if not p.is_file():
            cand = repo / clean
            if not cand.is_file():
                missing.append(str(row.get("id")))
    return missing
