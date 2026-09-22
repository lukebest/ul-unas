# UL-UNAS evaluation

- checkpoint: `checkpoints/model_trained_on_dns3.tar`
- RTF: 0.0440
- mean / p99 clip time: 110.6 / 161.4 ms

## Layer averages

### noisy

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vbdemand | 0.000 | 2.515 | -22.889 | 0.787 | 0.030 | -22.889 | 1.968 | 29.304 | 0.000 | -22.322 | -22.322 | 8.449 | 0.000 | 0.000 | 0.030 | 0.030 |
| all | 0.000 | 2.515 | -22.889 | 0.787 | 0.030 | -22.889 | 1.968 | 29.304 | 0.000 | -22.322 | -22.322 | 8.449 | 0.000 | 0.000 | 0.030 | 0.030 |

### ulunas

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vbdemand | 0.000 | 2.515 | -24.005 | 0.819 | 0.030 | -22.889 | 2.524 | 46.975 | 14.617 | -17.695 | -21.349 | 15.985 | 7.536 | 13.427 | 0.008 | 0.012 |
| all | 0.000 | 2.515 | -24.005 | 0.819 | 0.030 | -22.889 | 2.524 | 46.975 | 14.617 | -17.695 | -21.349 | 15.985 | 7.536 | 13.427 | 0.008 | 0.012 |
