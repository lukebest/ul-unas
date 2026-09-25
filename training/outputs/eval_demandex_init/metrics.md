# UL-UNAS evaluation

- checkpoint: `training/outputs/finetune_vbdemand/ulunas_finetuned.pt`
- RTF: 0.0434
- mean / p99 clip time: 86.7 / 90.4 ms

## Layer averages

### noisy

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vbdemandex | 0.000 | 2.000 | -26.254 | 0.766 | 0.018 | -26.254 | 1.394 | 0.000 | 0.000 | -26.254 | -26.254 | 9.721 | 0.000 | 0.000 | 0.018 | 0.017 |
| all | 0.000 | 2.000 | -26.254 | 0.766 | 0.018 | -26.254 | 1.394 | 0.000 | 0.000 | -26.254 | -26.254 | 9.721 | 0.000 | 0.000 | 0.018 | 0.017 |

### ulunas

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vbdemandex | 0.000 | 2.000 | -27.320 | 0.865 | 0.018 | -26.254 | 2.706 | 51.591 | 15.616 | -22.911 | -26.815 | 14.034 | 4.313 | 11.148 | 0.007 | 0.010 |
| all | 0.000 | 2.000 | -27.320 | 0.865 | 0.018 | -26.254 | 2.706 | 51.591 | 15.616 | -22.911 | -26.815 | 14.034 | 4.313 | 11.148 | 0.007 | 0.010 |
