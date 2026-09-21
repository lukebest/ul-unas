"""Load fine-tuned weights into StreamULUNAS and measure offline/stream error plus RTF."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.audio_io import load_mono, write_wav
from training.model_utils import load_ulunas
from ulunas_onnx.stream.ulunas_stream import StreamULUNAS


def stream_enhance(stream_model: StreamULUNAS, audio: np.ndarray, device) -> tuple[np.ndarray, dict]:
    x = torch.from_numpy(audio).unsqueeze(0).to(device)
    window = torch.hann_window(512, device=device)
    spec_c = torch.stft(x, n_fft=512, hop_length=256, win_length=512, window=window, return_complex=True)
    spec = torch.view_as_real(spec_c)
    conv, tfa, inter = StreamULUNAS.init_caches(1, device=device)
    frames = []
    times = []
    with torch.inference_mode():
        for i in range(spec.shape[2]):
            t0 = time.perf_counter()
            yi, conv, tfa, inter = stream_model(spec[:, :, i : i + 1, :], conv, tfa, inter)
            times.append((time.perf_counter() - t0) * 1000)
            frames.append(yi)
    ys = torch.cat(frames, dim=2)
    ys_c = torch.complex(ys[..., 0], ys[..., 1])
    wav = torch.istft(ys_c[0], n_fft=512, hop_length=256, win_length=512, window=window, onesided=True, length=x.shape[1])
    arr = np.array(times)
    stats = {
        "n_frames": int(len(times)),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "frame_budget_ms": 16.0,
        "rtf": float(arr.mean() / 16.0),
        "within_budget_p99": bool(np.percentile(arr, 99) < 16.0),
    }
    return wav.detach().cpu().numpy().astype(np.float32), stats


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    ckpt = args.ckpt
    model = load_ulunas(ckpt, device=device)
    stream = StreamULUNAS().to(device).eval()
    stream.load_state_dict(model.state_dict(), strict=True)

    audio, sr = load_mono(args.wav)
    assert sr == 16000
    with torch.inference_mode():
        offline = model(torch.from_numpy(audio).unsqueeze(0).to(device)).squeeze().cpu().numpy()
    streamed, timing = stream_enhance(stream, audio, device)
    err = float(np.max(np.abs(offline - streamed)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wav(out_dir / "offline.wav", offline, 16000)
    write_wav(out_dir / "stream.wav", streamed, 16000)

    export = None
    if args.export_onnx:
        try:
            import onnx
            import onnxruntime
            from onnxsim import simplify

            dummy_mix = torch.randn(1, 257, 1, 2, device=device)
            conv, tfa, inter = StreamULUNAS.init_caches(1, device=device)
            onnx_path = out_dir / "ulunas_finetuned_stream.onnx"
            try:
                torch.onnx.export(
                    stream,
                    (dummy_mix, conv, tfa, inter),
                    str(onnx_path),
                    input_names=["mix", "conv_cache", "tfa_cache", "inter_cache"],
                    output_names=["enh", "conv_cache_out", "tfa_cache_out", "inter_cache_out"],
                    opset_version=11,
                    do_constant_folding=False,
                    dynamo=False,
                )
            except TypeError:
                torch.onnx.export(
                    stream,
                    (dummy_mix, conv, tfa, inter),
                    str(onnx_path),
                    input_names=["mix", "conv_cache", "tfa_cache", "inter_cache"],
                    output_names=["enh", "conv_cache_out", "tfa_cache_out", "inter_cache_out"],
                    opset_version=11,
                    do_constant_folding=False,
                )
            model_onnx = onnx.load(str(onnx_path))
            onnx.checker.check_model(model_onnx)
            simple, ok = simplify(model_onnx)
            simple_path = out_dir / "ulunas_finetuned_stream_simple.onnx"
            if ok:
                onnx.save(simple, str(simple_path))
            export = {"onnx": str(onnx_path), "simple": str(simple_path) if ok else None, "simplified": bool(ok)}
        except Exception as exc:
            export = {"error": str(exc)}

    report = {
        "ckpt": str(ckpt),
        "wav": str(args.wav),
        "offline_vs_stream_maxabs": err,
        "match_ok": err < 5e-3,
        "timing": timing,
        "export": export,
        "algo_delay_ms": 32.0,
        "hop_ms": 16.0,
    }
    (out_dir / "deploy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("offline_vs_stream_maxabs", "match_ok", "timing", "export")}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/model_trained_on_dns3.tar")
    p.add_argument("--wav", default="audio/noisy/0174.wav")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="training/outputs/deploy")
    p.add_argument("--export_onnx", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
