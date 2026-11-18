"""Purpose: Verify leakage and surrogate invariants of offline cross-block decoding.
Inputs: Synthetic anonymous gait-phase summaries; cross_block_decoding.py.
Outputs: unittest results on stdout (no patient data or artifacts written).
Author: Lena
Date: 2026-10-20
Seed: 42 (numpy and random; fixture-only randomness).
"""

import importlib.util
from pathlib import Path
import random
import unittest

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cross_block_decoding", ROOT / "code/analysis/revision/cross_block_decoding.py"
)
DECODING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DECODING)


def fixture(block="pre8walk"):
    rows = []
    rng = np.random.default_rng(SEED)
    for cycle in range(4):
        for phase in DECODING.PHASES:
            for area in DECODING.AREAS:
                row = dict(
                    SubjectID="P4-1005",
                    Cohort="AgeMatched_NFB",
                    Block=block,
                    Modality="eeg",
                    Area=area,
                    Hemisphere="Both",
                    Epoch=cycle,
                    GaitPhase=phase,
                )
                row.update({f"ERD_ERS_{band}": rng.normal() for band in DECODING.BANDS})
                rows.append(row)
    return pd.DataFrame(rows)


class CrossBlockTests(unittest.TestCase):
    def test_centering_cancels_offsets_independently_per_cycle_area_band(self):
        original = fixture()
        shifted = original.copy()
        for index, band in enumerate(DECODING.BANDS):
            shifted[f"ERD_ERS_{band}"] += (
                120 + 10 * shifted.Epoch + 3 * index + (shifted.Area == "motor") * 5
            )
        a = DECODING.build_samples(original, "P4-1005", "pre8walk")
        b = DECODING.build_samples(shifted, "P4-1005", "pre8walk")
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)
        np.testing.assert_allclose(a[0].reshape(-1, 4, 10).mean(axis=1), 0, atol=1e-12)
        self.assertFalse(np.allclose(a[1], b[1]))

    def test_scaler_statistics_are_training_only(self):
        train = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y = np.array(["a", "a", "b", "b"])
        _, model = DECODING.fit_predict(train, y, train + 1000)
        np.testing.assert_allclose(model[0].mean_, train.mean(axis=0))
        self.assertEqual(model[0].n_samples_seen_, len(train))
        _, second = DECODING.fit_predict(train, y, train - 1000)
        np.testing.assert_allclose(model[1].coef_, second[1].coef_)

    def test_same_epoch_numbers_are_distinct_between_blocks(self):
        pre = DECODING.build_samples(fixture(), "P4-1005", "pre8walk")
        post = DECODING.build_samples(fixture("post8walk"), "P4-1005", "post8walk")
        self.assertFalse(set(pre[3]) & set(post[3]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            DECODING.decode_pair(pre, pre, "4class", "phase_centered", SEED, 0)
        result, predictions, _ = DECODING.decode_pair(
            pre, post, "4class", "phase_centered", SEED, 0
        )
        self.assertEqual(result["n_test_cycles"], 4)
        self.assertEqual(len(predictions), 16)

    def test_rotation_preserves_cyclic_order_and_class_counts(self):
        phase = np.tile(DECODING.PHASES, 30)
        cycles = np.repeat(np.arange(30), 4)
        rotated = DECODING.rotate_labels(phase, cycles, np.random.default_rng(SEED))
        shifts = []
        for cycle in np.unique(cycles):
            values = rotated[cycles == cycle]
            possibilities = [np.roll(DECODING.PHASES, k) for k in range(4)]
            matches = [
                k
                for k, candidate in enumerate(possibilities)
                if np.array_equal(values, candidate)
            ]
            self.assertEqual(len(matches), 1)
            shifts.extend(matches)
        self.assertGreater(len(set(shifts)), 1)
        self.assertEqual(set(rotated), set(phase))
        np.testing.assert_array_equal(
            rotated, DECODING.rotate_labels(phase, cycles, np.random.default_rng(SEED))
        )

    def test_incomplete_cycles_drop_and_duplicates_raise(self):
        table = fixture()
        incomplete = table.drop(index=0)
        data = DECODING.build_samples(incomplete, "P4-1005", "pre8walk")
        self.assertEqual(data[4]["n_cycles_dropped"], 1)
        self.assertEqual(data[0].shape, (12, 10))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            DECODING.build_samples(
                pd.concat([table, table.iloc[:1]]), "P4-1005", "pre8walk"
            )

    def test_whole_cycle_comparator_cannot_distinguish_phases(self):
        pre = DECODING.build_samples(fixture(), "P4-1005", "pre8walk")
        post = DECODING.build_samples(fixture("post8walk"), "P4-1005", "post8walk")
        for decoder, chance in [("4class", 0.25), ("DSvsSwing", 0.5)]:
            result, _, _ = DECODING.decode_pair(
                pre, post, decoder, "whole_cycle_only", SEED, 0
            )
            self.assertAlmostEqual(result["balanced_accuracy"], chance)


if __name__ == "__main__":
    unittest.main()
