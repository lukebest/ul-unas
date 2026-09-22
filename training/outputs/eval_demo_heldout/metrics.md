# UL-UNAS evaluation

- checkpoint: `training/outputs/finetune_vbdemand/ulunas_finetuned.pt`
- RTF: 0.0162
- mean / p99 clip time: 988.5 / 1110.5 ms

## Layer averages

### chip_loudness_matched

| layer | clip_rate | duration_s | est_dbfs | mix_dbfs | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| game_music | 0.000 | 60.960 | -45.382 | -45.382 | 66.532 | 16.627 | -37.283 | -41.440 |
| all | 0.000 | 61.070 | -44.414 | -44.414 | 64.982 | 17.147 | -36.026 | -40.313 |
| gunshot | 0.000 | 61.390 | -44.855 | -44.855 | 66.407 | 17.672 | -36.342 | -40.760 |
| mmo_bgm | 0.000 | 60.859 | -43.006 | -43.006 | 62.008 | 17.143 | -34.452 | -38.738 |

### noisy

| layer | clip_rate | duration_s | est_dbfs | mix_dbfs | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| game_music | 0.000 | 60.960 | -45.382 | -45.382 | 62.695 | 0.000 | -41.405 | -41.405 |
| all | 0.000 | 61.070 | -44.414 | -44.414 | 61.521 | 0.000 | -42.023 | -42.023 |
| gunshot | 0.000 | 61.390 | -44.855 | -44.855 | 61.215 | 0.000 | -42.064 | -42.064 |
| mmo_bgm | 0.000 | 60.859 | -43.006 | -43.006 | 60.652 | 0.000 | -42.599 | -42.599 |

### ulunas

| layer | clip_rate | duration_s | est_dbfs | mix_dbfs | proxy_bak | proxy_bak_improve | proxy_ovrl | proxy_sig_dbfs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| game_music | 0.000 | 60.960 | -46.189 | -45.382 | 65.260 | 8.437 | -39.421 | -41.530 |
| all | 0.000 | 61.070 | -47.829 | -44.414 | 66.077 | 14.770 | -38.681 | -42.374 |
| gunshot | 0.000 | 61.390 | -46.081 | -44.855 | 65.835 | 13.313 | -38.001 | -41.330 |
| mmo_bgm | 0.000 | 60.859 | -51.216 | -43.006 | 67.136 | 22.561 | -38.621 | -44.262 |
