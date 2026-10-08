# Dataset licenses for UL-UNAS target-domain training

Record source URL, creator, asset license, dataset license, commercial_allowed, attribution and SHA-1 before adding any file under `training/data/raw/`.

## Clean speech (preferred)

| Dataset | URL | License note |
|---|---|---|
| AISHELL-1 | https://openslr.org/33/ | Apache 2.0 on OpenSLR; confirm commercial use in writing |
| LibriSpeech | https://openslr.org/12/ | CC BY 4.0 |
| VCTK 0.92 | https://datashare.ed.ac.uk/handle/10283/3443 | CC BY 4.0 in package `license_text.txt` |
| DNS Challenge 5 speech | https://github.com/microsoft/DNS-Challenge | per-source licenses; keep an inventory |
| Common Voice | https://commonvoice.mozilla.org/en/terms | CC0; filter quality before use |

## Target noise (preferred)

| Dataset | URL | License note |
|---|---|---|
| Slakh2100 | https://zenodo.org/records/4599666 | CC BY 4.0; dedupe MIDI |
| MUSAN | https://openslr.org/17/ | CC BY 4.0 |
| FSD50K | https://zenodo.org/records/4060432 | keep only CC0 / CC BY for commercial work |
| Freesound API | https://freesound.org/docs/api/resources_apiv2.html | search `license:"Creative Commons 0"` |
| OpenGameArt CC0 | https://opengameart.org/content/cc0-sound-effects | check each asset |
| DNS noise | https://github.com/microsoft/DNS-Challenge | AudioSet / Freesound / DEMAND mix |
| DEMAND | https://zenodo.org/records/1227121 | CC BY-SA 3.0 |

## Do not use as commercial main-set without clearance

WHAM! (CC BY-NC), TAU Urban Acoustic Scenes (non-commercial), PUBG Gun Sound Dataset (research only), Primewords (CC BY-NC-ND), MTG-Jamendo (non-commercial academic).

## VoiceBank+DEMAND (used by T-ulunas-posttrain-001)

| Dataset | URL | License note |
|---|---|---|
| Valentini noisy-clean parallel set | https://doi.org/10.7488/ds/2117 | Edinburgh DataShare item; End-user Licence on the handle page. Citation: Valentini-Botinhao, C. (2017). |
| CSTR VCTK (clean speech) | https://doi.org/10.7488/ds/1994 | CC BY 4.0 |
| DEMAND (noise) | https://zenodo.org/records/1227121 | CC BY-SA 3.0 |
| 16 kHz HF mirror (opt-in only) | https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k | Declared CC BY 4.0; resample of the same Valentini pairs. **Not used unless** `--allow_hf_fallback`. Prefer official DataShare zips (End-user Licence). |

Download + 16 kHz resample:

```bash
python training/download_vbdemand.py --output_dir training/data/raw/voicebank_demand
# Official DataShare only. Add --allow_hf_fallback to opt in to the HF 16 kHz mirror.
```

WAV files stay under `training/data/raw/` (gitignored). A local `LICENSE.md` is written next to the data. This repo uses the set for a **research fine-tune** of UL-UNAS, not as a commercially cleared product model.

## MS-SNSD (T-ulunas-posttrain-002)

| Dataset | URL | License note |
|---|---|---|
| MS-SNSD scripts | https://github.com/microsoft/MS-SNSD | MIT |
| MS-SNSD clean speech | see Microsoft README | Edinburgh / PTDB-TUG (**ODbL**). Confirm before redistribution. |
| MS-SNSD noise | see Microsoft README | CC0 + DEMAND CC BY-SA 3.0 |
| In-repo smoke mixer | `training/synthesize_mssnsd.py` | Uses `audio/clean` + residual/procedural noise; **not** the official corpus |

Do **not** download the full CleanSpeech/Noise dumps on smoke boxes. Cap synth hours at 0.5–2.0 unless `--allow_large`.

```bash
python training/download_mssnsd.py --fetch_scripts
python training/synthesize_mssnsd.py --hours 0.5 --max_hours 2
```

## VB-DemandEx (T-ulunas-posttrain-002)

| Dataset | URL | License note |
|---|---|---|
| VB-DemandEx (HF card) | https://huggingface.co/datasets/NikolaiKyhne/VB-DemandEx | Card license **MIT**; pretty_name VB-DemandEx; ~1.94 GB |
| Paper / AAU | https://arxiv.org/abs/2501.06146 | xLSTM-SENet |
| Underlying VB+DEMAND | https://doi.org/10.7488/ds/2117 | VCTK CC BY 4.0 + DEMAND BY-SA 3.0 + DataShare End-user |

Pre-paired zips `clean_{train,valid,test}` + `noisy_{train,valid,test}`. Layout adapter writes `16k/{train,dev,test}/{clean,noisy}` (`valid` → `dev`) and is **not** a `find_vbdemand_roots` alias.

```bash
python training/download_demandex.py --layout
# Full HF pull is opt-in: python training/download_demandex.py --download
```

## URGENT / DNS (not ingested this round)

URGENT2024 official is **CC BY-NC-SA 4.0** (NC — research eval only; not a commercial train set). DNS 1–5 full dumps are TB-class and are **not** downloaded here. No `download_dns.py` / `download_urgent.py` ship in this task.

## Local fallback used by this repo

If `training/data/raw/` is empty, `synthesize_pairs.py` builds a runnable bootstrap set from:

- `audio/clean/*.wav` as official clean speech
- residual `audio/noisy - audio/clean` as general noise
- procedural music / gun / ambience textures

These bootstrap clips are only for pipeline bring-up. Replace them with licensed public data or in-house game-loop recordings before any production training.

Demo `降噪前` / `降噪后` files stay in `eval_real.jsonl` and are held-out.
