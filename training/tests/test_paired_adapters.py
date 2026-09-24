"""Layout / pairing / frozen-rule tests. No GPU, no corpus download."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import SAMPLE_RATE
from training.audio_io import write_wav
from training.paired_layout import (
    extra_dataset_roots,
    find_demandex_roots,
    find_generic_paired_roots,
    find_mssnsd_roots,
    match_pairs,
    pair_clean_stem,
    scan_extra_paired,
)
from training.prepare_manifest import find_vbdemand_roots
from training.synthesize_pairs import build_parser as synth_parser


def _wav(path: Path, freq: float = 220.0, seconds: float = 0.25) -> None:
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    write_wav(path, (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SAMPLE_RATE)


class PairStemTests(unittest.TestCase):
    def test_identical_and_official_mssnsd(self):
        stems = {"clnsp1", "p226_001", "clean_aa"}
        self.assertEqual(pair_clean_stem("p226_001", stems), "p226_001")
        self.assertEqual(pair_clean_stem("noisy1_SNRdb_10.0_clnsp1", stems), "clnsp1")
        self.assertEqual(pair_clean_stem("noisy1_SNRdb_0.0", stems), "clnsp1")
        self.assertIsNone(pair_clean_stem("unrelated", stems))


class LayoutTests(unittest.TestCase):
    def test_mssnsd_official_names_and_16k_tree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "mssnsd"
            clean = root / "CleanSpeech_training"
            noisy = root / "NoisySpeech_training"
            clean.mkdir(parents=True)
            noisy.mkdir(parents=True)
            _wav(clean / "clnsp1.wav", 200)
            _wav(noisy / "noisy1_SNRdb_10.0_clnsp1.wav", 200)
            test_c = root / "clean_test"
            test_n = root / "noisy_test"
            test_c.mkdir()
            test_n.mkdir()
            _wav(test_c / "uttA.wav", 180)
            _wav(test_n / "uttA.wav", 180)
            roots = find_mssnsd_roots(root)
            splits = {s for _, _, s in roots}
            self.assertIn("train", splits)
            self.assertIn("test", splits)
            pairs = match_pairs(clean, noisy)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0], "noisy1_SNRdb_10.0_clnsp1")

    def test_demandex_zip_names_not_valentini(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "vbdemandex"
            for split in ("train", "valid", "test"):
                (root / f"clean_{split}").mkdir(parents=True)
                (root / f"noisy_{split}").mkdir(parents=True)
                _wav(root / f"clean_{split}" / "p200_001.wav", 160)
                _wav(root / f"noisy_{split}" / "p200_001.wav", 160)
            # Valentini dirs sitting beside must not be claimed.
            val_c = root / "clean_trainset_28spk_wav"
            val_n = root / "noisy_trainset_28spk_wav"
            val_c.mkdir()
            val_n.mkdir()
            _wav(val_c / "p226_001.wav", 140)
            _wav(val_n / "p226_001.wav", 140)
            roots = find_demandex_roots(root)
            splits = sorted({s for _, _, s in roots})
            self.assertEqual(splits, ["dev", "test", "train"])
            for clean_dir, _, _ in roots:
                self.assertNotEqual(clean_dir.name, "clean_trainset_28spk_wav")

    def test_find_vbdemand_roots_ignores_demandex_and_mssnsd(self):
        with tempfile.TemporaryDirectory() as td:
            public = Path(td)
            (public / "vbdemandex" / "clean_train").mkdir(parents=True)
            (public / "vbdemandex" / "noisy_train").mkdir(parents=True)
            _wav(public / "vbdemandex" / "clean_train" / "x.wav")
            _wav(public / "vbdemandex" / "noisy_train" / "x.wav")
            (public / "mssnsd" / "16k" / "train" / "clean").mkdir(parents=True)
            (public / "mssnsd" / "16k" / "train" / "noisy").mkdir(parents=True)
            _wav(public / "mssnsd" / "16k" / "train" / "clean" / "y.wav")
            _wav(public / "mssnsd" / "16k" / "train" / "noisy" / "y.wav")
            self.assertEqual(find_vbdemand_roots(public), [])
            extras = scan_extra_paired(public, ROOT)
            self.assertIn("mssnsd", extras)
            self.assertIn("vbdemandex", extras)
            self.assertGreater(len(extras["mssnsd"]["train"]), 0)
            self.assertGreater(len(extras["vbdemandex"]["train"]), 0)

    def test_smoke_roots_not_appended_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            raw = repo / "training" / "data" / "raw"
            smoke_train = repo / "training" / "data" / "smoke" / "mssnsd" / "16k" / "train"
            (smoke_train / "clean").mkdir(parents=True)
            (smoke_train / "noisy").mkdir(parents=True)
            raw.mkdir(parents=True)
            _wav(smoke_train / "clean" / "s.wav")
            _wav(smoke_train / "noisy" / "s.wav")
            self.assertEqual(extra_dataset_roots(raw, repo)["mssnsd"], [])
            flagged = extra_dataset_roots(raw, repo, include_smoke=True)["mssnsd"]
            self.assertTrue(any("smoke" in p.parts for p in flagged))
            via_public = extra_dataset_roots(repo / "training" / "data" / "smoke", repo)["mssnsd"]
            self.assertTrue(via_public)

    def test_generic_split_tree(self):
        """Keep: public API for find_generic_paired_roots (not DATASET_SPECS)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "generic"
            for split in ("train", "dev", "test"):
                (root / split / "clean").mkdir(parents=True)
                (root / split / "noisy").mkdir(parents=True)
                _wav(root / split / "clean" / "z.wav")
                _wav(root / split / "noisy" / "z.wav")
            roots = find_generic_paired_roots(root)
            self.assertEqual({s for _, _, s in roots}, {"train", "dev", "test"})


class FrozenRulesTests(unittest.TestCase):
    def test_mode_auto_default(self):
        args = synth_parser().parse_args([])
        self.assertEqual(args.mode, "auto")

    def test_rule_a_b_still_gate_best(self):
        from training.train_ulunas import should_write_best

        self.assertFalse(should_write_best(10.0, -1e9, 12.0, Path("/tmp/no-best-002.pt"), n=4))
        self.assertTrue(should_write_best(12.0, -1e9, 12.0, Path("/tmp/no-best-002.pt"), n=4))
        self.assertFalse(should_write_best(20.0, -1e9, 1.0, Path("/tmp/no-best-002.pt"), n=0))

    def test_g5_static(self):
        from training.eval_gates import gate_g5

        g = gate_g5()
        self.assertTrue(g["ok"], g)


class HoursCapTests(unittest.TestCase):
    def test_refuse_over_cap(self):
        from training.synthesize_mssnsd import build_parser, synthesize

        with tempfile.TemporaryDirectory() as td:
            args = build_parser().parse_args(
                ["--hours", "3", "--max_hours", "2", "--output_dir", td, "--repo", str(ROOT)]
            )
            with self.assertRaises(SystemExit):
                synthesize(args)


if __name__ == "__main__":
    unittest.main()
