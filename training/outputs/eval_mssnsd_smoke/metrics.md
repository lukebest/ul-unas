# UL-UNAS evaluation

- checkpoint: `training/outputs/finetune_mssnsd_demandex/ulunas_last.pt`
- RTF: 0.0330
- mean / p99 clip time: 66.0 / 81.8 ms

## Layer averages

### noisy

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mssnsd | 0.000 | 2.000 | -27.577 | 0.697 | 0.007 | -27.577 | 1.417 | 0.000 | 0.000 | -27.577 | -27.577 | 12.377 | 0.000 | 0.000 | 0.007 | 0.010 |
| all | 0.000 | 2.000 | -27.577 | 0.697 | 0.007 | -27.577 | 1.417 | 0.000 | 0.000 | -27.577 | -27.577 | 12.377 | 0.000 | 0.000 | 0.007 | 0.010 |

### ulunas

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mssnsd | 0.000 | 2.000 | -28.168 | 0.804 | 0.007 | -27.577 | 2.509 | 50.796 | 12.343 | -24.637 | -27.722 | 16.010 | 3.633 | 12.540 | 0.000 | 0.007 |
| all | 0.000 | 2.000 | -28.168 | 0.804 | 0.007 | -27.577 | 2.509 | 50.796 | 12.343 | -24.637 | -27.722 | 16.010 | 3.633 | 12.540 | 0.000 | 0.007 |
