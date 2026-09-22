"""Run the target-domain pipeline in plan order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--export_onnx", action="store_true")
    args = parser.parse_args()

    from training import align_teacher, deploy_stream, evaluate, postfilter, prepare_manifest, remixit, scene_head, synthesize_pairs, train_ulunas

    summary = {}
    summary["manifest"] = prepare_manifest.build(prepare_manifest.build_parser().parse_args([]))
    summary["synth"] = synthesize_pairs.synthesize(synthesize_pairs.build_parser().parse_args([]))
    summary["eval_baseline"] = evaluate.run_eval(
        evaluate.build_parser().parse_args([
            "--manifest", "training/data/manifests/eval_all.jsonl",
            "--output_dir", "training/outputs/eval_baseline",
            "--device", args.device,
        ])
    )["timing"]
    if not args.skip_train:
        summary["train"] = train_ulunas.train(
            train_ulunas.parse_args([
                "--config", "training/configs/default.json",
                "--device", args.device,
                "--output_dir", "training/outputs/finetune",
                "--train_manifests", "training/data/manifests/train_synth.jsonl",
                "--dev_manifests", "training/data/manifests/dev_synth.jsonl",
                "--max_steps", "16",
                "--epochs", "2",
                "--seconds", "2.0",
            ])
        )
        ckpt = summary["train"]["ckpt"]
    else:
        ckpt = "checkpoints/model_trained_on_dns3.tar"
    summary["teacher"] = {
        "kept": align_teacher.run(align_teacher.build_parser().parse_args([])).get("kept")
    }
    summary["postfilter"] = "training/outputs/postfilter/postfilter_report.json"
    postfilter.run_ablation(
        postfilter.build_parser().parse_args([
            "--ckpt", ckpt,
            "--device", args.device,
        ])
    )
    summary["remixit"] = remixit.run(
        remixit.build_parser().parse_args(["--device", args.device, "--method", "remixit"])
    )
    summary["re2re"] = remixit.run(
        remixit.build_parser().parse_args([
            "--device", args.device,
            "--method", "re2re",
            "--output_dir", "training/outputs/re2re",
        ])
    )
    summary["scene"] = scene_head.run(scene_head.build_parser().parse_args([]))
    deploy_args = ["--ckpt", ckpt, "--device", args.device]
    if args.export_onnx:
        deploy_args.append("--export_onnx")
    summary["deploy"] = deploy_stream.run(deploy_stream.build_parser().parse_args(deploy_args))
    Path("training/outputs/pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in list(v)[:6]}) for k, v in summary.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
