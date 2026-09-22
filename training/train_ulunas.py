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

from training.audio_io import load_mono
from training.dataset import PairDataset, balanced_sampler, read_jsonl
from training.losses import total_loss
from training.metrics import si_sdr
from training.model_utils import apply_freeze, enhance_numpy, load_ulunas, trainable_param_count
from training.paths import resolve_audio


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_manifest(model, manifest: str | Path, device, max_clips: int = 24) -> dict[str, float]:
    items = [r for r in read_jsonl(manifest) if r.get("clean") and r.get("noisy")]
    if max_clips:
        items = items[:max_clips]
    scores = []
    model.eval()
    for item in items:
        mix, _ = load_mono(resolve_audio(item["noisy"]))
        clean, _ = load_mono(resolve_audio(item["clean"]))
        est = enhance_numpy(model, mix, device=device)
        n = min(len(est), len(clean), len(mix))
        scores.append(si_sdr(est[:n], clean[:n]))
    return {"si_sdr": float(sum(scores) / max(len(scores), 1)), "n": len(scores)}


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
    if args.dev_manifests:
        init_full = evaluate_manifest(model, args.dev_manifests[0], device, max_clips=args.max_dev or 24)
        print(f"init full-file SI-SDR={init_full['si_sdr']:.3f} n={init_full['n']}")
    else:
        init_full = None

    train_set = PairDataset(args.train_manifests, seconds=args.seconds)
    dev_set = PairDataset(args.dev_manifests, seconds=args.seconds) if args.dev_manifests else None
    if dev_set is not None and args.max_dev and len(dev_set) > args.max_dev:
        from torch.utils.data import Subset

        stride = max(1, len(dev_set) // args.max_dev)
        idx = list(range(0, len(dev_set), stride))[: args.max_dev]
        dev_set = Subset(dev_set, idx)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_set),
        num_workers=args.num_workers,
        drop_last=len(train_set) >= args.batch_size,
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
                    w_mag=args.w_mag,
                    w_ri=args.w_ri,
                    w_sisnr=args.w_sisnr,
                    w_prot=args.w_prot,
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
            if step % max(args.log_every * 5, 100) == 0:
                torch.save(
                    {"model": model.state_dict(), "step": step, "freeze": args.freeze, "init_ckpt": str(args.ckpt)},
                    out_dir / "ulunas_last.pt",
                )
            if args.max_steps and step >= args.max_steps:
                break
        metrics = {"epoch": epoch, "step": step, "train_loss": float(loss.detach())}
        if args.dev_manifests:
            full = evaluate_manifest(model, args.dev_manifests[0], device, max_clips=args.max_dev or 24)
            metrics["full_si_sdr"] = full["si_sdr"]
            metrics["full_n"] = full["n"]
            print(f"epoch {epoch} full-file SI-SDR={full['si_sdr']:.3f} n={full['n']}")
            if full["si_sdr"] > best:
                best = full["si_sdr"]
                torch.save(
                    {"model": model.state_dict(), "metrics": metrics, "freeze": args.freeze, "init_ckpt": str(args.ckpt)},
                    best_path,
                )
        elif dev_loader is not None:
            metrics.update(evaluate_loader(model, dev_loader, device))
            print(f"epoch {epoch} crop SI-SDR={metrics['si_sdr']:.3f}")
            if metrics["si_sdr"] > best:
                best = metrics["si_sdr"]
                torch.save(
                    {"model": model.state_dict(), "metrics": metrics, "freeze": args.freeze, "init_ckpt": str(args.ckpt)},
                    best_path,
                )
        else:
            torch.save(
                {"model": model.state_dict(), "metrics": metrics, "freeze": args.freeze, "init_ckpt": str(args.ckpt)},
                best_path,
            )
        history.append(metrics)
        if args.max_steps and step >= args.max_steps:
            break

    last_path = out_dir / "ulunas_last.pt"
    torch.save({"model": model.state_dict(), "history": history, "freeze": args.freeze, "init_ckpt": str(args.ckpt)}, last_path)
    report = {
        "init_ckpt": str(args.ckpt),
        "trainable": n_train,
        "total": n_all,
        "freeze": args.freeze,
        "lr": args.lr,
        "w_mag": args.w_mag,
        "w_ri": args.w_ri,
        "w_sisnr": args.w_sisnr,
        "w_prot": args.w_prot,
        "init_full_si_sdr": None if init_full is None else init_full["si_sdr"],
        "batch_size": args.batch_size,
        "seconds_per_clip": args.seconds,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "steps_run": step,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "best_si_sdr": best if best > -1e8 else None,
        "seconds": time.time() - t0,
        "ckpt": str(best_path),
        "last_ckpt": str(last_path),
        "train_manifests": list(args.train_manifests),
        "dev_manifests": list(args.dev_manifests or []),
        "history": history,
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("init_ckpt", "trainable", "best_si_sdr", "steps_run", "device", "seconds", "ckpt")}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/model_trained_on_dns3.tar")
    p.add_argument("--train_manifests", nargs="+", default=["training/data/manifests/train_synth.jsonl"])
    p.add_argument("--dev_manifests", nargs="+", default=["training/data/manifests/dev_synth.jsonl"])
    p.add_argument("--output_dir", default="training/outputs/finetune")
    p.add_argument("--device", default=default_device())
    p.add_argument("--freeze", default="decoder_tail", choices=["decoder_tail", "decoder_all", "full"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=0, help="0 = run all epochs with no step cap")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_dev", type=int, default=64, help="cap in-loop dev clips; 0 = all")
    p.add_argument("--use_teacher", action="store_true")
    p.add_argument("--w_teacher", type=float, default=0.2)
    p.add_argument("--w_mag", type=float, default=70.0)
    p.add_argument("--w_ri", type=float, default=30.0)
    p.add_argument("--w_sisnr", type=float, default=1.0)
    p.add_argument("--w_prot", type=float, default=8.0)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
