# UL-UNAS target-domain training

16 kHz causal fine-tuning, chip-teacher alignment, post-filters and streaming checks. Demo recordings are evaluation-only.

## Order

```bash
python training/prepare_manifest.py
python training/synthesize_pairs.py
python training/evaluate.py --manifest training/data/manifests/eval_all.jsonl --output_dir training/outputs/eval_baseline
python training/train_ulunas.py --freeze decoder_tail --lr 1e-4
python training/align_teacher.py
python training/postfilter.py --ckpt training/outputs/finetune/ulunas_finetuned.pt
python training/remixit.py --method remixit
python training/scene_head.py
python training/deploy_stream.py --ckpt training/outputs/finetune/ulunas_finetuned.pt
```

Or run `python training/run_pipeline.py`.

## Notes

- Place AISHELL / LibriSpeech / VCTK / Slakh / MUSAN / FSD50K under `training/data/raw/<name>/`; see `data/licenses.md`.
- Chip outputs need delay + slow-gain correction. Do not use waveform L1 / SI-SNR on unaligned teacher audio.
- Current demo chip clips remain held-out. `--use_teacher` is off unless you add non-test teacher recordings.
- Post-filter only keeps a setting if BAK/residual improves and SIG / clean-passthrough do not collapse.
- Streaming hop is 16 ms; STFT still costs 32 ms algorithmic delay.
