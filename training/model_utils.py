"""Load UL-UNAS, extract masks, and control freeze groups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training import HOP_LEN, N_FFT, WIN_LEN
from ulunas import ULUNAS, cTFA


CKPT_DEFAULT = Path("checkpoints/model_trained_on_dns3.tar")


def load_ulunas(ckpt: str | Path | None = None, device: str | torch.device = "cpu") -> ULUNAS:
    device = torch.device(device)
    model = ULUNAS().to(device)
    path = Path(ckpt) if ckpt else CKPT_DEFAULT
    state = torch.load(str(path), map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def enhance_numpy(model: nn.Module, audio: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.inference_mode():
        y = model(x)
    return y.squeeze().detach().cpu().numpy().astype(np.float32)


def enhance_with_mask(model: ULUNAS, audio: np.ndarray, device: torch.device | None = None):
    if device is None:
        device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(device)
    n_samples = x.shape[1]
    window = torch.hann_window(model.win_len, device=device)
    stft_kwargs = {
        "n_fft": model.n_fft,
        "hop_length": model.hop_len,
        "win_length": model.win_len,
        "window": window,
        "onesided": True,
    }
    with torch.inference_mode():
        spec_c = torch.stft(x, **stft_kwargs, return_complex=True)
        spec = torch.view_as_real(spec_c).permute(0, 3, 2, 1)
        feat = torch.log10(torch.norm(spec, dim=1, keepdim=True).clamp(1e-12))
        feat = model.erb.bm(feat)
        feat, en_outs = model.encoder(feat)
        feat = model.dpgrnn(feat)
        m_feat = model.decoder(feat, en_outs)
        mask = model.erb.bs(m_feat)
        spec_enh = spec * mask
        spec_enh = spec_enh.permute(0, 3, 2, 1)
        spec_enh_c = torch.complex(spec_enh[..., 0], spec_enh[..., 1])
        output = torch.istft(spec_enh_c, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples - output.shape[1]))
    wav = output.squeeze().detach().cpu().numpy().astype(np.float32)
    gain = mask.squeeze().detach().cpu().numpy().astype(np.float32)
    return wav, gain, spec.squeeze().detach().cpu().numpy()


def freeze_batchnorm(model: nn.Module, freeze: bool = True) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()
            if freeze:
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)


def apply_freeze(model: ULUNAS, mode: str = "decoder_tail") -> None:
    """mode: decoder_tail | decoder_all | full"""
    for p in model.parameters():
        p.requires_grad = False

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        freeze_batchnorm(model, freeze=True)
        return

    for module in model.modules():
        if isinstance(module, cTFA):
            for p in module.parameters():
                p.requires_grad = True

    if mode == "decoder_tail":
        for block in model.decoder.de_convs[-2:]:
            for p in block.parameters():
                p.requires_grad = True
    elif mode == "decoder_all":
        for p in model.decoder.parameters():
            p.requires_grad = True
        for p in model.dpgrnn.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown freeze mode: {mode}")
    freeze_batchnorm(model, freeze=True)


def trainable_param_count(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, total


def stft_kwargs(device) -> dict:
    return {
        "n_fft": N_FFT,
        "hop_length": HOP_LEN,
        "win_length": WIN_LEN,
        "window": torch.hann_window(WIN_LEN, device=device),
        "onesided": True,
    }
