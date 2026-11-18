"""Purpose: Verify alignment, gaps, purging and training-only stride regression.
Inputs: Synthetic arrays only; code/analysis/revision/next_stride_prediction.py.
Outputs: unittest results (no files).
Author: Lena
Date: 2026-10-20
Seed: 42.
"""

import importlib.util
import random
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code/analysis/revision/next_stride_prediction.py"
)
spec = importlib.util.spec_from_file_location("next_stride_prediction", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(n=80):
    gait = pd.DataFrame(
        dict(
            cycle_index=np.arange(n),
            kept=True,
            t_rhs_s=np.arange(n, dtype=float),
            stride_time_s=1.0,
        )
    )
    for column in module.TIMINGS:
        gait[column] = 0.25
    features = {i: np.arange(40, dtype=float).reshape(4, 10) + i for i in range(n)}
    rows = []
    for i in range(n):
        for phase_i, phase in enumerate(module.PHASES):
            for area_i, area in enumerate(module.AREAS):
                row = dict(
                    SubjectID="synthetic",
                    Block="pre8walk",
                    Modality="eeg",
                    Epoch=i,
                    GaitPhase=phase,
                    Area=area,
                    Hemisphere="Both",
                )
                row.update(
                    {
                        f"ERD_ERS_{band}": features[i][phase_i, area_i * 5 + j]
                        for j, band in enumerate(module.BANDS)
                    }
                )
                rows.append(row)
    return gait, pd.DataFrame(rows), features


class StrideTests(unittest.TestCase):
    def test_exact_alignment_rejects_shifted_epochs(self):
        gait, neural, _ = fixture(6)
        aligned, features, incomplete = module.validate_alignment(
            gait, neural, np.ones(6, dtype=bool), "synthetic"
        )
        self.assertEqual(list(aligned.index), list(range(6)))
        self.assertEqual(len(features), 6)
        self.assertEqual(incomplete, [])
        neural.Epoch += 1
        with self.assertRaisesRegex(ValueError, "epoch set"):
            module.validate_alignment(gait, neural, np.ones(6, dtype=bool), "synthetic")

    def test_rejected_cycle_and_physical_gap_break_windows(self):
        gait, _, features = fixture(12)
        gait.loc[4, "kept"] = False
        gait.loc[8:, "t_rhs_s"] += 2
        rows, counts = module.build_samples(gait.set_index("cycle_index"), features)
        currents = {row["current_cycle"] for row in rows}
        self.assertTrue({3, 4, 5, 7, 8}.isdisjoint(currents))
        self.assertGreater(counts["gap_or_rejected"], 0)
        self.assertIn(2, currents)

    def test_earlier_cycle_sensitivity_checks_intervening_cycles(self):
        gait, _, features = fixture(12)
        gait.loc[4, "kept"] = False
        rows, _ = module.build_samples(gait.set_index("cycle_index"), features, lag=2)
        for row in rows:
            self.assertFalse(row["first_cycle"] <= 4 <= row["last_cycle"])
            self.assertEqual(row["target_cycle"] - row["feature_cycle"], 3)

    def test_purge_covers_all_used_cycles(self):
        gait, _, features = fixture()
        samples, _ = module.build_samples(gait.set_index("cycle_index"), features)
        folds = list(module.blocked_folds(samples))
        self.assertTrue(folds)
        seen = []
        for _, train, test in folds:
            seen.extend(test.tolist())
            for i in train:
                for j in test:
                    separation = max(
                        samples[i]["first_cycle"] - samples[j]["last_cycle"],
                        samples[j]["first_cycle"] - samples[i]["last_cycle"],
                    )
                    self.assertGreater(separation, 5)
        self.assertEqual(len(seen), len(set(seen)))

    def test_scaler_never_fits_test_values(self):
        train = np.array([[0.0], [2.0], [4.0]])
        _, fitted = module.fit_predict(
            train, np.array([0.0, 1.0, 2.0]), np.array([[1000.0]])
        )
        np.testing.assert_array_equal(fitted.named_steps["standardscaler"].mean_, [2.0])

    def test_contrasts_cancel_constant_offsets(self):
        values = np.arange(40, dtype=float).reshape(4, 10)
        mean, contrast = module.neural_features(values)
        shifted_mean, shifted_contrast = module.neural_features(values + 99)
        np.testing.assert_allclose(contrast, shifted_contrast, atol=1e-12)
        np.testing.assert_allclose(shifted_mean - mean, 99)
        self.assertEqual(contrast.size, 30)

    def test_single_participant_summary_has_no_inferential_statistics(self):
        metrics = pd.DataFrame(
            [
                dict(
                    subject="synthetic",
                    cohort="synthetic",
                    lag_cycles=2,
                    model=name,
                    mae_s=0.01 + index * 0.001,
                )
                for index, name in enumerate(module.MODELS)
            ]
        )
        _, summary = module.summarize(metrics, SEED, 100)
        self.assertTrue(summary.ci_low_s.isna().all())
        self.assertTrue(summary.ci_high_s.isna().all())
        self.assertTrue(summary.two_sided_sign_flip_p.isna().all())
        self.assertTrue((summary.inference_status == "descriptive_only_n_lt_3").all())


if __name__ == "__main__":
    unittest.main()
