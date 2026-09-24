# T-ulunas-posttrain-002 — MS-SNSD + VB-DemandEx adapters

Narrow domain-expand ship: layout/ingest adapters, hour-capped MS-SNSD synth,
optional download helpers, short continue-train recipe, G0–G5 gates.

**Does not** change frozen rules A/B/C or `--mode=auto`. **Does not** download
DNS or URGENT corpora. **Does not** touch NoiseZero. Dual VB-DMD writers
(`scan_vbdemand` + `ingest_public_pairs`) stay.

Research use only. Not a production or commercially cleared model.

## Checkpoints (verified on main@5f1acaf)

| Role | Path | Present on this checkout |
|---|---|---|
| Continue-train init (preferred) | `training/outputs/finetune_vbdemand/ulunas_finetuned.pt` | yes (PR #1 ship) |
| Fallback init | `checkpoints/model_trained_on_dns3.tar` | yes |
| Do not resume | `training/outputs/finetune/ulunas_finetuned.pt` | 16-step SI-SDR −50 smoke |
| New run output | `training/outputs/finetune_mssnsd_demandex/` | do **not** overwrite the VB best |

`python -c "from training.continue_init import describe_init; print(describe_init())"`

Freeze remains `decoder_tail`. Rule **A**: never write `ulunas_finetuned.pt` if
the probe SI-SDR is strictly below init. Rule **B**: existing-best lock.

## Data

### MS-SNSD (microsoft/MS-SNSD)

- Native **16 kHz**. Train pairs are synthesized; `clean_test`/`noisy_test` are pre-mixed.
- Official synth names: `noisyN_SNRdb_*_clnspN.wav` ↔ `clnspN.wav`.
- This repo writes same-stem `{split}/{clean,noisy}` for ingest, and still
  matches official names via `training/paired_layout.py`.
- Hour cap for smoke: **0.5–2.0 h** (`--hours` / `--max_hours`). `--allow_large` to raise.

```bash
# Scripts only (optional). Does not fetch the multi-GB CleanSpeech/Noise dumps.
python training/download_mssnsd.py --fetch_scripts

# Smoke mixer from in-repo audio/clean + residual (or local MS-SNSD sources):
python training/synthesize_mssnsd.py --hours 0.5 --max_hours 2 \
  --output_dir training/data/processed/mssnsd/16k

# If you already ran the official synthesizer:
python training/download_mssnsd.py --layout_from /path/to/MS-SNSD
```

### VB-DemandEx (HF `NikolaiKyhne/VB-DemandEx`)

- Pre-paired zips: `clean_{train,valid,test}.zip` + `noisy_{train,valid,test}.zip` (~1.94 GB).
- Adapter maps `valid` → `dev` and writes `16k/{train,dev,test}/{clean,noisy}`.
- **Not** registered with `find_vbdemand_roots` (Valentini aliases unchanged).

```bash
# Optional full download. Skip on smoke boxes.
python training/download_demandex.py --download --layout

# Layout-only if zips/dirs are already on disk:
python training/download_demandex.py --layout --output_dir training/data/raw/vbdemandex
```

### Index (dual writers preserved)

```bash
python training/prepare_manifest.py
python training/synthesize_pairs.py --mode ingest   # still auto-keys only on VB roots
```

`--mode=auto` still means: ingest if `find_vbdemand_roots` is non-empty, else
synth. Extra datasets write **separate** JSONLs
`{train,dev,eval}_{mssnsd,vbdemandex}.jsonl` and never replace the VB-DMD files.

## Continue-train

```bash
python training/train_ulunas.py --config training/configs/mssnsd_demandex.json
```

CLI equivalent (init resolved by you; config default is the verified VB ship):

```bash
python training/train_ulunas.py \
  --ckpt training/outputs/finetune_vbdemand/ulunas_finetuned.pt \
  --train_manifests training/data/manifests/train_mssnsd.jsonl \
                    training/data/manifests/train_vbdemandex.jsonl \
  --dev_manifests training/data/manifests/dev_vbdemand.jsonl \
  --output_dir training/outputs/finetune_mssnsd_demandex \
  --freeze decoder_tail \
  --lr 1e-5 --batch_size 8 --seconds 2.0 --epochs 1 --max_steps 80 \
  --max_dev 24 --w_mag 10 --w_ri 5 --w_sisnr 1 --w_prot 2
```

If `dev_vbdemand.jsonl` audio is missing (gitignored `training/data/raw/`),
point `--dev_manifests` at `dev_mssnsd.jsonl` so the A/B probe is non-empty.

## Eval + gates

```bash
python training/evaluate.py \
  --manifest training/data/manifests/eval_vbdemand.jsonl \
  --ckpt training/outputs/finetune_mssnsd_demandex/ulunas_finetuned.pt \
  --output_dir training/outputs/eval_vbdemand_domain_expand --no-save_audio

python training/evaluate.py \
  --manifest training/data/manifests/eval_mssnsd.jsonl \
  --ckpt training/outputs/finetune_mssnsd_demandex/ulunas_finetuned.pt \
  --output_dir training/outputs/eval_mssnsd --no-save_audio

python training/eval_gates.py \
  --vb_metrics training/outputs/eval_vbdemand_domain_expand/metrics_summary.json \
  --new_metrics training/outputs/eval_mssnsd/metrics_summary.json
```

| Gate | Threshold |
|---|---|
| G0 | new-domain JSONLs non-empty; no unresolved clean |
| G1 | on official `eval_vbdemand` (n=824): SI-SDR drop ≤0.2 dB vs `finetune_vbdemand`; PESQ drop ≤0.02 |
| G2 | new-domain test SI-SDRi > 0 |
| G3 | 24-clip existing-best lock + rule A still block a collapse write |
| G4 | RTF / 16 ms frame budget recorded (CPU smoke OK) |
| G5 | no A/B/C or `--mode=auto` change; no DNS/URGENT downloader; no NoiseZero |

## Smoke (this box)

`python training/run_domain_expand_smoke.py` synthesizes a few minutes of
MS-SNSD-style pairs from `audio/clean`, writes a tiny DemandEx zip-name
fixture, ingests both, continue-trains a handful of CPU steps from
`finetune_vbdemand/ulunas_finetuned.pt`, evals the new-domain test, and runs
G0–G5.

**CUDA:** see the smoke JSON `cuda_available`. If false, this is a **CPU
pipeline + smoke** landing. Do **not** treat it as GPU acceptance. G1 on the
full Valentini test is **skipped** when `training/data/raw/voicebank_demand/`
wavs are absent (gitignored).

## Out of scope

DNS subset synth, URGENT2024 eval import, long multi-epoch GPU FT, changing
`--mode=auto`, merging, opening the coordinator PR.
