"""Optional tiny scene/transient head used as a soft gain conditioner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.audio_io import load_mono, stft_np


SCENES = ["game_music", "gunshot", "mmo_bgm", "other"]


class SceneHead(nn.Module):
    def __init__(self, n_freq: int = 257, hidden: int = 32, n_class: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_freq, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_class),
        )

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        # mag: (B, F, T)
        pooled = mag.mean(dim=-1)
        return self.net(pooled)


def soft_gain_scale(logits: torch.Tensor) -> torch.Tensor:
    """Return per-scene attenuation bias in [0.7, 1.0]."""
    p = logits.softmax(dim=-1)
    scales = logits.new_tensor([0.82, 0.75, 0.88, 1.0])
    return (p * scales).sum(dim=-1, keepdim=True)


def infer_label(item: dict) -> int:
    layer = item.get("layer") or item.get("scene") or "other"
    return SCENES.index(layer) if layer in SCENES else 3


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    head = SceneHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    items = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    losses = []
    for _ in range(args.epochs):
        for item in items:
            wav, _ = load_mono(item["noisy"])
            mag = np.abs(stft_np(wav[: 16000 * 2]))
            x = torch.from_numpy(mag).unsqueeze(0).to(device)
            y = torch.tensor([infer_label(item)], device=device)
            logits = head(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "scene_head.pt"
    torch.save({"model": head.state_dict(), "scenes": SCENES}, ckpt)
    report = {
        "ckpt": str(ckpt),
        "mean_loss": float(np.mean(losses) if losses else 0),
        "note": "Soft conditioner only; do not hard-switch models. Use when game/gun/MMO policies conflict.",
    }
    (out / "scene_head_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="training/data/manifests/eval_synth.jsonl")
    p.add_argument("--output_dir", default="training/outputs/scene")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=2)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
