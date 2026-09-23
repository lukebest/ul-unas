# UL-UNAS evaluation

- checkpoint: `training/outputs/finetune_vbdemand/ulunas_finetuned.pt`
- RTF: 0.0433
- mean / p99 clip time: 109.0 / 159.9 ms

## Layer averages

### noisy

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vbdemand | 0.000 | 2.515 | -22.889 | 0.787 | 0.030 | -22.889 | 1.968 | 29.304 | 0.000 | -22.322 | -22.322 | 8.449 | 0.000 | 0.000 | 0.030 | 0.030 |
| all | 0.000 | 2.515 | -22.889 | 0.787 | 0.030 | -22.889 | 1.968 | 29.304 | 0.000 | -22.322 | -22.322 | 8.449 | 0.000 | 0.000 | 0.030 | 0.030 |

### ulunas

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vbdemand | 0.000 | 2.515 | -24.071 | 0.820 | 0.030 | -22.889 | 2.554 | 48.378 | 16.375 | -17.067 | -21.161 | 17.215 | 8.766 | 17.145 | 0.005 | 0.011 |
| all | 0.000 | 2.515 | -24.071 | 0.820 | 0.030 | -22.889 | 2.554 | 48.378 | 16.375 | -17.067 | -21.161 | 17.215 | 8.766 | 17.145 | 0.005 | 0.011 |
