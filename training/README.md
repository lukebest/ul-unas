# UL-UNAS target-domain training

16 kHz causal fine-tune from the **official DNS3 checkpoint** (`checkpoints/model_trained_on_dns3.tar`). Demo recordings are evaluation-only.

**Disclaimer:** the VoiceBank+DEMAND run is a research fine-tune. It is not a production or commercially cleared model.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data (VoiceBank+DEMAND)

Official 48 kHz pairs from Edinburgh DataShare (Valentini), resampled to 16 kHz:

```bash
python training/download_vbdemand.py --output_dir training/data/raw/voicebank_demand
# Official DataShare only by default. Opt-in HF 16 kHz mirror:
# python training/download_vbdemand.py --allow_hf_fallback
python training/prepare_manifest.py
python training/synthesize_pairs.py --mode ingest
```

Two writers can emit `{train,dev,eval}_vbdemand.jsonl`; keep both:

- `prepare_manifest.scan_vbdemand` (`python training/prepare_manifest.py`) — catalog indexer: points at files already under `training/data/raw` (no copy), wav/flac/ogg + nested-noisy search, and also writes speech/noise/demo/official manifests.
- `synthesize_pairs.ingest_public_pairs` (`python training/synthesize_pairs.py --mode ingest`) — ingest: wav-only, noisy must match by filename, also writes `paired_all.jsonl`, optional `--copy_paired` 16 kHz copies. Overwrites the three VB-DMD JSONLs if `prepare_manifest` already ran.

`--mode synth` is the tiny bootstrap mixer, not a VB-DMD writer. Default `--mode auto` selects ingest when `find_vbdemand_roots` is non-empty and skips synth (use `--mode synth` or `--mode both` for the mixer). See `training/experiments/vbdemand_finetune.md`.

- Train / dev / eval JSONL: `training/data/manifests/{train,dev,eval}_vbdemand.jsonl` (repo-relative paths).
- Dev speakers `p226` and `p287` are held out of the 28-spk train set.
- Eval is the official Valentini test speakers (`p232`, `p257`).
- In-repo `demo/*降噪前.wav` clips stay in `eval_real.jsonl` with `held_out: true`.
- Licenses: `training/data/licenses.md` and `training/data/raw/voicebank_demand/LICENSE.md`.

If `training/data/raw/` is empty, `synthesize_pairs.py --mode synth` still builds a tiny bootstrap set from `audio/clean` (pipeline bring-up only).

## VoiceBank+DEMAND fine-tune (not the 16-step smoke)

Init **only** from `checkpoints/model_trained_on_dns3.tar`. Do not resume `training/outputs/finetune/ulunas_finetuned.pt`.

Entrypoint is `training/configs/vbdemand.json` (safe defaults: `lr=1e-5`, `w_mag=10`, `w_ri=5`, `w_prot=2`, `max_steps=800`). CLI flags override the JSON. `ulunas_finetuned.pt` (best/ship ckpt) is never written when the probe SI-SDR is strictly below the init SI-SDR. `ulunas_last.pt` is not gated by that check.

```bash
python training/train_ulunas.py --config training/configs/vbdemand.json
```

Equivalent explicit flags:

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
  --w_mag 10 --w_ri 5 --w_sisnr 1 --w_prot 2
```

`--device` defaults to `cuda` when available, otherwise `cpu`. `--max_steps 0` means no step cap (not recommended). `--max_dev 0` uses the full dev set.

## Eval (SI-SDR + PESQ)

Baseline (official DNS3) vs fine-tuned, on the held-out VB-DMD test split:

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

Keep the in-repo demo held-out:

```bash
python training/evaluate.py \
  --manifest training/data/manifests/eval_real.jsonl \
  --ckpt training/outputs/finetune_vbdemand/ulunas_finetuned.pt \
  --output_dir training/outputs/eval_demo_heldout
```

See `training/experiments/vbdemand_finetune.md` for the measured table. Eval with clean references requires `pesq` (`--no-require-pesq` to skip).

## Older smoke / chip-teacher order

```bash
python training/prepare_manifest.py
python training/synthesize_pairs.py --mode synth
python training/evaluate.py --manifest training/data/manifests/eval_all.jsonl --output_dir training/outputs/eval_baseline
python training/train_ulunas.py --config training/configs/default.json --output_dir training/outputs/finetune --max_steps 16
python training/align_teacher.py
python training/postfilter.py --ckpt checkpoints/model_trained_on_dns3.tar
python training/remixit.py --method remixit
python training/scene_head.py
python training/deploy_stream.py --ckpt checkpoints/model_trained_on_dns3.tar
```

`python training/run_pipeline.py` still runs that smoke path (`--max_steps 16`) and is not the VB-DMD fine-tune.

## Notes

- Chip outputs need delay + slow-gain correction. Do not use waveform L1 / SI-SNR on unaligned teacher audio.
- Current demo chip clips remain held-out. `--use_teacher` is off unless you add non-test teacher recordings.
- Post-filter only keeps a setting if BAK/residual improves and SIG / clean-passthrough do not collapse.
- Streaming hop is 16 ms; STFT still costs 32 ms algorithmic delay.
