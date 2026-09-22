# UL-UNAS evaluation

- checkpoint: `checkpoints/model_trained_on_dns3.tar`
- RTF: 0.0164
- mean / p99 clip time: 1003.4 / 1073.5 ms

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
| game_music | 0.000 | 60.960 | -45.646 | -45.382 | 65.291 | 5.556 | -39.960 | -41.349 |
| all | 0.000 | 61.070 | -47.409 | -44.414 | 65.866 | 13.393 | -38.925 | -42.273 |
| gunshot | 0.000 | 61.390 | -45.829 | -44.855 | 65.731 | 12.634 | -38.125 | -41.284 |
| mmo_bgm | 0.000 | 60.859 | -50.754 | -43.006 | 66.578 | 21.989 | -38.689 | -44.186 |
