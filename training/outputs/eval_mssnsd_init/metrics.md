# UL-UNAS evaluation

- checkpoint: `/workspace/training/outputs/finetune_vbdemand/ulunas_finetuned.pt`
- RTF: 0.0465
- mean / p99 clip time: 93.1 / 109.0 ms

## Layer averages

### noisy

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mssnsd | 0.000 | 2.000 | -27.577 | 0.697 | 0.007 | -27.577 | 1.417 | 0.000 | 0.000 | -27.577 | -27.577 | 12.377 | 0.000 | 0.000 | 0.007 | 0.010 |
| all | 0.000 | 2.000 | -27.577 | 0.697 | 0.007 | -27.577 | 1.417 | 0.000 | 0.000 | -27.577 | -27.577 | 12.377 | 0.000 | 0.000 | 0.007 | 0.010 |

### ulunas

| layer | clip_rate | duration_s | est_dbfs | estoi | input_silence_rms | mix_dbfs | pesq | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs | si_sdr | si_sdri | silence_atten_db | silence_residual_rms | speech_residual_rms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mssnsd | 0.000 | 2.000 | -28.169 | 0.804 | 0.007 | -27.577 | 2.507 | 50.824 | 12.371 | -24.630 | -27.723 | 16.011 | 3.634 | 12.588 | 0.000 | 0.007 |
| all | 0.000 | 2.000 | -28.169 | 0.804 | 0.007 | -27.577 | 2.507 | 50.824 | 12.371 | -24.630 | -27.723 | 16.011 | 3.634 | 12.588 | 0.000 | 0.007 |
