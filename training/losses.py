"""Paper hybrid loss plus identity, noise-only and speech-protection terms."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from training.model_utils import stft_kwargs


def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = torch.sum(est * ref, dim=-1, keepdim=True)
    s_energy = torch.sum(ref * ref, dim=-1, keepdim=True)
    s_target = dot / (s_energy + eps) * ref
    e_noise = est - s_target
    return 10 * torch.log10((torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps))


def compressed_specs(wav: torch.Tensor, power: float = 0.3):
    spec = torch.stft(wav, **stft_kwargs(wav.device), return_complex=True)
    mag = spec.abs().clamp_min(1e-8)
    mag_c = mag ** power
    phase = spec / mag
    spec_c = mag_c * phase
    return mag_c, torch.view_as_real(spec_c)


def hybrid_loss(est: torch.Tensor, ref: torch.Tensor, w_mag: float = 70.0, w_ri: float = 30.0, w_sisnr: float = 1.0) -> dict[str, torch.Tensor]:
    mag_e, ri_e = compressed_specs(est)
    mag_r, ri_r = compressed_specs(ref)
    t = min(mag_e.shape[-1], mag_r.shape[-1])
    l_mag = F.mse_loss(mag_e[..., :t], mag_r[..., :t])
    l_ri = F.mse_loss(ri_e[..., :t, :], ri_r[..., :t, :])
    l_sisnr = -si_snr(est, ref).mean()
    total = w_mag * l_mag + w_ri * l_ri + w_sisnr * l_sisnr
    return {"loss": total, "l_mag": l_mag, "l_ri": l_ri, "l_sisnr": l_sisnr}


def identity_loss(est: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(est, mix)


def noise_only_loss(est: torch.Tensor) -> torch.Tensor:
    return est.pow(2).mean()


def speech_protection_loss(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    mag_e, _ = compressed_specs(est)
    mag_r, _ = compressed_specs(ref)
    t = min(mag_e.shape[-1], mag_r.shape[-1])
    under = F.relu(mag_r[..., :t] - mag_e[..., :t])
    return under.mean()


def mask_distill_loss(est: torch.Tensor, mix: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    mag_e, _ = compressed_specs(est)
    mag_m, _ = compressed_specs(mix)
    mag_t, _ = compressed_specs(teacher)
    t = min(mag_e.shape[-1], mag_m.shape[-1], mag_t.shape[-1])
    pred_mask = (mag_e[..., :t] / mag_m[..., :t].clamp_min(1e-6)).clamp(0, 1)
    teacher_mask = (mag_t[..., :t] / mag_m[..., :t].clamp_min(1e-6)).clamp(0, 1)
    return F.mse_loss(pred_mask, teacher_mask)


def total_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    mix: torch.Tensor,
    kind: str,
    teacher: torch.Tensor | None = None,
    w_id: float = 5.0,
    w_noise: float = 5.0,
    w_prot: float = 8.0,
    w_teacher: float = 0.2,
    w_mag: float = 70.0,
    w_ri: float = 30.0,
    w_sisnr: float = 1.0,
) -> dict[str, torch.Tensor]:
    if kind == "noise_only":
        l = noise_only_loss(est) * w_noise
        return {"loss": l, "l_noise": l}
    if kind == "clean_only":
        l_h = hybrid_loss(est, ref, w_mag=w_mag, w_ri=w_ri, w_sisnr=w_sisnr)
        l_id = identity_loss(est, mix) * w_id
        out = {**l_h, "l_id": l_id}
        out["loss"] = l_h["loss"] + l_id
        return out
    out = hybrid_loss(est, ref, w_mag=w_mag, w_ri=w_ri, w_sisnr=w_sisnr)
    out["l_prot"] = speech_protection_loss(est, ref) * w_prot
    out["loss"] = out["loss"] + out["l_prot"]
    if teacher is not None:
        out["l_teacher"] = mask_distill_loss(est, mix, teacher) * w_teacher
        out["loss"] = out["loss"] + out["l_teacher"]
    return out
