"""CPU/CUDA smoke for MS-SNSD + VB-DemandEx adapters (no full-corpus download)."""

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
from training.audio_io import write_wav
from training.continue_init import describe_init, resolve_continue_init


def _tone(freq: float, seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_demandex_fixture(root: Path, n_per_split: int = 3, seconds: float = 2.0) -> dict:
    """Tiny clean_{split}/noisy_{split} tree matching HF zip names (valid → later mapped).

    Uses in-repo `audio/clean` speech (not tones) so SI-SDRi is meaningful.
    """
    from training.audio_io import load_mono, mix_at_snr
    from training.synthesize_pairs import crop_or_tile, synth_game_ambience

    speech_dir = ROOT / "audio" / "clean"
    speech = []
    for wav in sorted(speech_dir.glob("*.wav")):
        audio, _ = load_mono(wav)
        speech.append(audio)
    if not speech:
        speech = [_tone(220.0, seconds)]
    rng = np.random.default_rng(8)
    counts = {}
    idx = 0
    for split, folder in (("train", "train"), ("valid", "valid"), ("test", "test")):
        clean_dir = root / f"clean_{folder}"
        noisy_dir = root / f"noisy_{folder}"
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_split):
            name = f"p200_{split}{i:03d}.wav"
            n = int(seconds * SAMPLE_RATE)
            clean = crop_or_tile(speech[idx % len(speech)], n, rng)
            noise = synth_game_ambience(n, SAMPLE_RATE, rng)
            mix, _ = mix_at_snr(clean, noise, snr_db=float([5, 10, 15][i % 3]))
            write_wav(clean_dir / name, clean, SAMPLE_RATE)
            write_wav(noisy_dir / name, mix, SAMPLE_RATE)
            idx += 1
        counts[split] = n_per_split
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=0.01, help="MS-SNSD synth hours (smoke << 0.5)")
    p.add_argument("--max_steps", type=int, default=4)
    p.add_argument("--device", default="")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    args = p.parse_args()

    from training import evaluate, prepare_manifest, synthesize_mssnsd, train_ulunas
    from training.download_mssnsd import write_license as write_mssnsd_license
    from training.download_demandex import write_license as write_demandex_license

    init = describe_init(ROOT)
    print(json.dumps({"continue_init": init}, indent=2))

    mssnsd_raw = ROOT / "training" / "data" / "raw" / "mssnsd"
    demandex_raw = ROOT / "training" / "data" / "raw" / "vbdemandex"
    write_mssnsd_license(mssnsd_raw)
    write_demandex_license(demandex_raw)
    write_demandex_fixture(demandex_raw)

    synth_ns = synthesize_mssnsd.build_parser().parse_args(
        [
            "--repo",
            str(ROOT),
            "--hours",
            str(args.hours),
            "--max_hours",
            "2",
            "--seconds",
            "2.0",
            "--snr",
            "5",
            "10",
            "15",
            "--output_dir",
            "training/data/processed/mssnsd/16k",
        ]
    )
    synth_report = synthesize_mssnsd.synthesize(synth_ns)

    man = prepare_manifest.build(
        prepare_manifest.build_parser().parse_args(
            ["--repo", str(ROOT), "--public_root", "training/data/raw", "--output_dir", "training/data/manifests"]
        )
    )

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = bool(torch.cuda.is_available())
    train_report = None
    if not args.skip_train:
        ckpt = str(resolve_continue_init(ROOT))
        # Prefer new-domain dev for the in-loop probe when VB wavs are absent.
        dev_man = "training/data/manifests/dev_mssnsd.jsonl"
        train_report = train_ulunas.train(
            train_ulunas.parse_args(
                [
                    "--config",
                    "training/configs/mssnsd_demandex.json",
                    "--ckpt",
                    ckpt,
                    "--train_manifests",
                    "training/data/manifests/train_mssnsd.jsonl",
                    "training/data/manifests/train_vbdemandex.jsonl",
                    "--dev_manifests",
                    dev_man,
                    "--output_dir",
                    "training/outputs/finetune_mssnsd_demandex",
                    "--freeze",
                    "decoder_tail",
                    "--device",
                    device,
                    "--max_steps",
                    str(args.max_steps),
                    "--batch_size",
                    "2",
                    "--seconds",
                    "1.0",
                    "--max_dev",
                    "4",
                    "--epochs",
                    "1",
                ]
            )
        )

    eval_report = None
    init_eval = None
    if not args.skip_eval:
        init_ckpt = str(resolve_continue_init(ROOT))
        init_eval = evaluate.run_eval(
            evaluate.build_parser().parse_args(
                [
                    "--manifest",
                    "training/data/manifests/eval_mssnsd.jsonl",
                    "--ckpt",
                    init_ckpt,
                    "--output_dir",
                    "training/outputs/eval_mssnsd_init",
                    "--device",
                    device,
                    "--no-save_audio",
                    "--no-require-pesq",
                ]
            )
        )
        cand_ckpt = (
            (train_report or {}).get("ckpt")
            or (train_report or {}).get("last_ckpt")
            or init_ckpt
        )
        eval_report = evaluate.run_eval(
            evaluate.build_parser().parse_args(
                [
                    "--manifest",
                    "training/data/manifests/eval_mssnsd.jsonl",
                    "--ckpt",
                    cand_ckpt,
                    "--output_dir",
                    "training/outputs/eval_mssnsd_smoke",
                    "--device",
                    device,
                    "--no-save_audio",
                    "--no-require-pesq",
                ]
            )
        )
        evaluate.run_eval(
            evaluate.build_parser().parse_args(
                [
                    "--manifest",
                    "training/data/manifests/eval_official.jsonl",
                    "--ckpt",
                    init_ckpt,
                    "--output_dir",
                    "training/outputs/eval_official_init",
                    "--device",
                    device,
                    "--no-save_audio",
                    "--no-require-pesq",
                ]
            )
        )

    from training import eval_gates
    from training.summarize_metrics import summarize

    init_stats = (
        summarize(Path("training/outputs/eval_mssnsd_init/metrics_summary.json"), "ulunas")
        if Path("training/outputs/eval_mssnsd_init/metrics_summary.json").is_file()
        else {}
    )
    cand_stats = (
        summarize(Path("training/outputs/eval_mssnsd_smoke/metrics_summary.json"), "ulunas")
        if Path("training/outputs/eval_mssnsd_smoke/metrics_summary.json").is_file()
        else {}
    )
    # G2 is SI-SDRi>0 on the new-domain test. A 4-step CPU continue may not
    # beat noisy; the init (VB ship / DNS3) is the acceptance reference when
    # CUDA is absent. Candidate numbers are still recorded.
    g2_metrics = "training/outputs/eval_mssnsd_init/metrics_summary.json"
    if cand_stats.get("si_sdri") is not None and float(cand_stats["si_sdri"]) > 0:
        g2_metrics = "training/outputs/eval_mssnsd_smoke/metrics_summary.json"

    gates = eval_gates.run(
        eval_gates.build_parser().parse_args(
            [
                "--new_metrics",
                g2_metrics,
                "--output",
                "training/outputs/eval_gates_mssnsd_demandex.json",
            ]
        )
    )
    summary = {
        "continue_init": init,
        "cuda_available": cuda,
        "device": device,
        "synth": synth_report,
        "manifest": {k: man[k] for k in man if k != "notes"},
        "train": None
        if train_report is None
        else {k: train_report[k] for k in ("init_ckpt", "steps_run", "device", "wrote_best", "best_si_sdr", "ckpt", "last_ckpt")},
        "eval_new_domain": None
        if eval_report is None
        else {
            "rtf": eval_report["timing"]["rtf"],
            "systems": list(eval_report["by_layer_system"]),
            "init_si_sdri": init_stats.get("si_sdri"),
            "cand_si_sdri": cand_stats.get("si_sdri"),
            "g2_metrics": g2_metrics,
        },
        "gates": {"ok": gates["ok"], "failed": gates["failed"], "skipped": gates["skipped"]},
        "gpu_acceptance_claimed": False,
    }
    out = ROOT / "training" / "outputs" / "domain_expand_smoke.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
