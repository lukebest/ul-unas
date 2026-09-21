"""Paired and replay datasets with event-balanced sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from training.audio_io import load_mono


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class PairDataset(Dataset):
    def __init__(self, manifests: list[str | Path], seconds: float = 4.0, sr: int = 16000):
        self.rows: list[dict] = []
        for man in manifests:
            for row in read_jsonl(man):
                if row.get("held_out"):
                    continue
                if not row.get("noisy"):
                    continue
                self.rows.append(row)
        if not self.rows:
            raise ValueError("PairDataset is empty")
        self.n = int(seconds * sr)
        self.sr = sr

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        mix, _ = load_mono(row["noisy"])
        if row.get("clean"):
            clean, _ = load_mono(row["clean"])
        else:
            clean = np.zeros_like(mix)
        teacher = None
        if row.get("teacher"):
            teacher, _ = load_mono(row["teacher"])
        mix = self._fix_len(mix)
        clean = self._fix_len(clean)
        item = {
            "mix": torch.from_numpy(mix),
            "clean": torch.from_numpy(clean),
            "kind": row.get("scene") or row.get("layer") or "synthetic",
            "id": row.get("id", str(idx)),
        }
        if teacher is not None:
            item["teacher"] = torch.from_numpy(self._fix_len(teacher))
        return item

    def _fix_len(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) >= self.n:
            start = 0 if len(audio) == self.n else int(np.random.randint(0, len(audio) - self.n + 1))
            return audio[start : start + self.n]
        return np.pad(audio, (0, self.n - len(audio)))


def balanced_sampler(dataset: PairDataset) -> WeightedRandomSampler:
    weights = []
    for row in dataset.rows:
        scene = row.get("scene") or row.get("layer") or "synthetic"
        w = 1.0
        if scene == "gunshot":
            w = 2.0
        elif scene == "mmo_bgm":
            w = 1.6
        elif scene == "game_music":
            w = 1.3
        elif scene in {"clean_only", "noise_only"}:
            w = 0.8
        weights.append(w)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
