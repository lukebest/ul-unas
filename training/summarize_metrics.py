"""Summarize SI-SDR / PESQ from one or more evaluate.py metrics.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize(path: Path, system: str = "ulunas") -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in report.get("rows", []) if r.get("system") == system]
    out = {
        "file": str(path),
        "ckpt": report.get("ckpt"),
        "system": system,
        "n": len(rows),
        "si_sdr": _mean(rows, "si_sdr"),
        "si_sdri": _mean(rows, "si_sdri"),
        "pesq": _mean(rows, "pesq"),
        "estoi": _mean(rows, "estoi"),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+", help="metrics.json from evaluate.py")
    p.add_argument("--system", default="ulunas")
    p.add_argument("--out", default="")
    args = p.parse_args()
    rows = [summarize(Path(m), args.system) for m in args.metrics]
    noisy = [summarize(Path(m), "noisy") for m in args.metrics]
    payload = {"systems": rows, "noisy": noisy}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
