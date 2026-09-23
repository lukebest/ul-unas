"""Summarize SI-SDR / PESQ from evaluate.py metrics.json or aggregate metrics_summary.json."""

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
    if rows:
        return {
            "file": str(path),
            "ckpt": report.get("ckpt"),
            "system": system,
            "n": len(rows),
            "si_sdr": _mean(rows, "si_sdr"),
            "si_sdri": _mean(rows, "si_sdri"),
            "pesq": _mean(rows, "pesq"),
            "estoi": _mean(rows, "estoi"),
        }
    stats = (report.get("by_layer_system") or {}).get(system, {}).get("all") or {}
    return {
        "file": str(path),
        "ckpt": report.get("ckpt"),
        "system": system,
        "n": report.get("n_clips"),
        "si_sdr": stats.get("si_sdr"),
        "si_sdri": stats.get("si_sdri"),
        "pesq": stats.get("pesq"),
        "estoi": stats.get("estoi"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("metrics", nargs="+", help="metrics.json or metrics_summary.json from evaluate.py")
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
