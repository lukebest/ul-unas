# T-ulunas-posttrain-001 — VoiceBank+DEMAND fine-tune

Research fine-tune of UL-UNAS from the official DNS3 checkpoint. **Not a production or commercially cleared model.**

## Checkpoint

- **Init (required):** `checkpoints/model_trained_on_dns3.tar`
- **Do not use:** `training/outputs/finetune/ulunas_finetuned.pt` (16-step SI-SDR −50 smoke)
- **Output:** `training/outputs/finetune_vbdemand/ulunas_finetuned.pt` (`init_ckpt` recorded in the file)

This VM had **no CUDA device** (`torch.cuda.is_available() == False`). The run used CPU; the trainer still selects `cuda` automatically when a GPU is present.

## Data

VoiceBank+DEMAND (Valentini 28-spk train / 2-spk test), downloaded from Edinburgh DataShare and resampled to 16 kHz.

- DOI: https://doi.org/10.7488/ds/2117
- Handle: https://datashare.ed.ac.uk/handle/10283/2791
- Speech: CSTR VCTK, CC BY 4.0
- Noise: DEMAND, CC BY-SA 3.0
- Edinburgh DataShare item licence: End-user Licence
- Local copy: `training/data/raw/voicebank_demand/` (gitignored)
- Download: `python training/download_vbdemand.py`
- License notes: `training/data/licenses.md`, `training/data/voicebank_demand_LICENSE.md`

Counts: 11,572 train pairs + 824 official test pairs at 16 kHz.

| Split | Manifest | n | Speakers |
|---|---|---|---|
| train | `training/data/manifests/train_vbdemand.jsonl` | 10802 | 28-spk minus `p226`, `p287` |
| dev | `training/data/manifests/dev_vbdemand.jsonl` | 770 | `p226`, `p287` |
| eval | `training/data/manifests/eval_vbdemand.jsonl` | 824 | official test `p232`, `p257` (held-out) |

In-repo `demo/*降噪前.wav` remains held-out (`eval_real.jsonl`). In-repo `audio/{clean,noisy}` official pairs are also held-out.

## Train command (shipped run)

```bash
python training/train_ulunas.py \
  --ckpt checkpoints/model_trained_on_dns3.tar \
  --train_manifests training/data/manifests/train_vbdemand.jsonl \
  --dev_manifests training/data/manifests/dev_vbdemand.jsonl \
  --output_dir training/outputs/finetune_vbdemand \
  --freeze decoder_tail \
  --lr 1e-5 \
  --batch_size 8 \
  --seconds 2.0 \
  --epochs 1 \
  --max_steps 800 \
  --log_every 50 \
  --num_workers 2 \
  --max_dev 24 \
  --w_mag 10 --w_ri 5 --w_sisnr 1 --w_prot 2 \
  --device cpu
```

- **Steps actually run:** 800 (batch 8, 2.0 s crops, freeze `decoder_tail`)
- **Wall time:** 546 s
- **Init full-file SI-SDR on 24 dev clips:** 12.068 dB
- **After 800 steps, same 24-clip probe:** 13.967 dB

A first attempt at `--lr 1e-4`, paper-scale mag weights (`70/30`), 2 epochs / 2700 steps **collapsed** quality (full-file SI-SDR dropped vs DNS3 on a 8-clip probe). That run is logged in `train_report_lr1e-4_collapsed.json` and was **not** used for the table below.

## Metrics — held-out VoiceBank+DEMAND test (n = 824)

Measured by `training/evaluate.py` with wideband PESQ. No fabricated numbers.

| System | SI-SDR (dB) | SI-SDRi (dB) | PESQ (WB) | ESTOI | n |
|---|---|---|---|---|---|
| noisy | 8.449 | 0.000 | 1.968 | 0.787 | 824 |
| official DNS3 | 15.985 | 7.536 | 2.524 | 0.819 | 824 |
| finetuned VB-DMD | 17.215 | 8.766 | 2.554 | 0.820 | 824 |

Delta vs official DNS3: **+1.231 dB SI-SDR**, **+0.030 PESQ**.

Sources: `training/outputs/eval_vbdemand_baseline/metrics.json`, `training/outputs/eval_vbdemand_finetuned/metrics.json`, `training/outputs/finetune_vbdemand/eval_compare.json`.

## In-repo demo (held-out, no paired clean)

`eval_real.jsonl` was not used for training. Proxy-only (no SI-SDR / PESQ):

| System | proxy BAK improve | proxy OVRL |
|---|---|---|
| official DNS3 | 13.393 | −38.925 |
| finetuned | 14.770 | −38.681 |

## Eval commands

```bash
python training/evaluate.py \
  --manifest training/data/manifests/eval_vbdemand.jsonl \
  --ckpt checkpoints/model_trained_on_dns3.tar \
  --output_dir training/outputs/eval_vbdemand_baseline \
  --no-save_audio

python training/evaluate.py \
  --manifest training/data/manifests/eval_vbdemand.jsonl \
  --ckpt training/outputs/finetune_vbdemand/ulunas_finetuned.pt \
  --output_dir training/outputs/eval_vbdemand_finetuned \
  --no-save_audio
```
