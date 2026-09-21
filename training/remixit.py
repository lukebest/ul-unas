"""Noisy-only domain adaptation: RemixIT and Re2Re helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.audio_io import load_mono
from training.losses import hybrid_loss
from training.model_utils import apply_freeze, load_ulunas


def remix_batch(speech_hat: torch.Tensor, noise_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    noise_hat = speech_hat.new_tensor(noise_hat) if not torch.is_tensor(noise_hat) else noise_hat
    perm = torch.randperm(noise_hat.shape[0], device=noise_hat.device)
    remixed_noise = noise_hat[perm]
    mix = speech_hat + remixed_noise
    return mix, speech_hat


def remixit_step(student, teacher, mix: torch.Tensor, ema: bool = False) -> torch.Tensor:
    with torch.no_grad():
        s_hat = teacher(mix)
        n_hat = mix - s_hat
        remix, target = remix_batch(s_hat, n_hat)
    pred = student(remix)
    return hybrid_loss(pred, target)["loss"]


def re2re_step(student, teacher, mix: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        s_hat = teacher(mix)
        n_hat = mix - s_hat
        remix_a, _ = remix_batch(s_hat, n_hat)
        remix_b, _ = remix_batch(s_hat, n_hat)
    pred_a = student(remix_a)
    pred_b = student(remix_b)
    return hybrid_loss(pred_a, pred_b.detach())["loss"] + hybrid_loss(pred_b, pred_a.detach())["loss"]


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    teacher = load_ulunas(args.ckpt, device=device)
    student = load_ulunas(args.ckpt, device=device)
    student.train()
    apply_freeze(student, "decoder_tail")
    opt = torch.optim.Adam((p for p in student.parameters() if p.requires_grad), lr=args.lr)
    items = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [it for it in items if not it.get("held_out")]
    if not items:
        items = [json.loads(l) for l in Path("training/data/manifests/train_synth.jsonl").read_text().splitlines() if l.strip()]
    losses = []
    n = min(args.max_steps, len(items))
    for i in range(n):
        wav, _ = load_mono(items[i % len(items)]["noisy"])
        wav = wav[: int(args.seconds * 16000)]
        mix = torch.from_numpy(wav).unsqueeze(0).to(device)
        if args.method == "re2re":
            loss = re2re_step(student, teacher, mix)
        else:
            loss = remixit_step(student, teacher, mix)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"{args.method}_student.pt"
    torch.save({"model": student.state_dict(), "method": args.method, "losses": losses}, ckpt)
    report = {"method": args.method, "mean_loss": float(np.mean(losses) if losses else 0), "ckpt": str(ckpt)}
    (out / "remixit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="remixit", choices=["remixit", "re2re"])
    p.add_argument("--ckpt", default="checkpoints/model_trained_on_dns3.tar")
    p.add_argument("--manifest", default="training/data/manifests/eval_official.jsonl")
    p.add_argument("--output_dir", default="training/outputs/remixit")
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--max_steps", type=int, default=4)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
