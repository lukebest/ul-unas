# UL-UNAS evaluation

- checkpoint: `/workspace/training/outputs/finetune_vbdemand/ulunas_finetuned.pt`
- RTF: 0.0154
- mean / p99 clip time: 154.5 / 168.8 ms

## Layer averages

### noisy

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_paired | 0.000 | 10.000 | -24.917 | 0.517 | 0.016 | -24.917 | 1.067 | 42.418 | 0.000 | -24.163 | -24.163 | 1.414 | 0.000 | 0.000 | 0.016 | 0.040 |
| all | 0.000 | 10.000 | -24.917 | 0.517 | 0.016 | -24.917 | 1.067 | 42.418 | 0.000 | -24.163 | -24.163 | 1.414 | 0.000 | 0.000 | 0.016 | 0.040 |

### ulunas

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_paired | 0.000 | 10.000 | -28.092 | 0.651 | 0.016 | -24.917 | 1.884 | 52.218 | 21.590 | -21.393 | -26.790 | 6.750 | 5.336 | 22.943 | 0.001 | 0.019 |
| all | 0.000 | 10.000 | -28.092 | 0.651 | 0.016 | -24.917 | 1.884 | 52.218 | 21.590 | -21.393 | -26.790 | 6.750 | 5.336 | 22.943 | 0.001 | 0.019 |
