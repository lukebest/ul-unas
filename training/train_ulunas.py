"""Fine-tune UL-UNAS with generic replay, speech protection and optional teacher masks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset import PairDataset, balanced_sampler, read_jsonl
from training.losses import total_loss
from training.metrics import si_sdr
from training.model_utils import apply_freeze, load_ulunas, trainable_param_count


def evaluate_loader(model, loader, device) -> dict[str, float]:
    model.eval()
    scores = []
    with torch.inference_mode():
        for batch in loader:
            mix = batch["mix"].to(device)
            clean = batch["clean"].to(device)
            est = model(mix)
            for i in range(mix.shape[0]):
                if float(clean[i].abs().mean()) < 1e-6:
                    continue
                scores.append(si_sdr(est[i].cpu().numpy(), clean[i].cpu().numpy()))
    return {"si_sdr": float(sum(scores) / max(len(scores), 1)), "n": len(scores)}


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    model = load_ulunas(args.ckpt, device=device)
    model.train()
    apply_freeze(model, args.freeze)
    n_train, n_all = trainable_param_count(model)
    print(f"trainable {n_train}/{n_all} freeze={args.freeze}")

    train_set = PairDataset(args.train_manifests, seconds=args.seconds)
    dev_set = PairDataset(args.dev_manifests, seconds=args.seconds) if args.dev_manifests else None
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_set),
        num_workers=0,
        drop_last=False,
    )
    dev_loader = (
        DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0) if dev_set is not None else None
    )
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best = -1e9
    best_path = out_dir / "ulunas_finetuned.pt"
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        apply_freeze(model, args.freeze)
        for batch in loader:
            mix = batch["mix"].to(device)
            clean = batch["clean"].to(device)
            kinds = batch["kind"]
            est = model(mix)
            loss = 0.0
            logs = []
            for i in range(mix.shape[0]):
                teacher = batch["teacher"][i : i + 1].to(device) if "teacher" in batch else None
                # keep demo/chip teacher unused unless explicitly allowed
                if not args.use_teacher:
                    teacher = None
                part = total_loss(
                    est[i : i + 1],
                    clean[i : i + 1],
                    mix[i : i + 1],
                    kind=kinds[i],
                    teacher=teacher,
                    w_teacher=args.w_teacher,
                )
                loss = loss + part["loss"]
                logs.append({k: float(v.detach().cpu()) for k, v in part.items()})
            loss = loss / mix.shape[0]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                print(f"step {step} loss={float(loss.detach().cpu()):.4f} kind={kinds[0]}")
            if step >= args.max_steps:
                break
        metrics = {"epoch": epoch, "step": step, "train_loss": float(loss)}
        if dev_loader is not None:
            metrics.update(evaluate_loader(model, dev_loader, device))
            print(f"epoch {epoch} dev SI-SDR={metrics['si_sdr']:.3f}")
            if metrics["si_sdr"] > best:
                best = metrics["si_sdr"]
                torch.save({"model": model.state_dict(), "metrics": metrics, "freeze": args.freeze}, best_path)
        else:
            torch.save({"model": model.state_dict(), "metrics": metrics, "freeze": args.freeze}, best_path)
        history.append(metrics)
        if step >= args.max_steps:
            break

    last_path = out_dir / "ulunas_last.pt"
    torch.save({"model": model.state_dict(), "history": history, "freeze": args.freeze}, last_path)
    report = {
        "trainable": n_train,
        "total": n_all,
        "best_si_sdr": best if best > -1e8 else None,
        "seconds": time.time() - t0,
        "ckpt": str(best_path),
        "history": history,
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("trainable", "best_si_sdr", "seconds", "ckpt")}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/model_trained_on_dns3.tar")
    p.add_argument("--train_manifests", nargs="+", default=["training/data/manifests/train_synth.jsonl"])
    p.add_argument("--dev_manifests", nargs="+", default=["training/data/manifests/dev_synth.jsonl"])
    p.add_argument("--output_dir", default="training/outputs/finetune")
    p.add_argument("--device", default="cpu")
    p.add_argument("--freeze", default="decoder_tail", choices=["decoder_tail", "decoder_all", "full"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--use_teacher", action="store_true")
    p.add_argument("--w_teacher", type=float, default=0.2)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
