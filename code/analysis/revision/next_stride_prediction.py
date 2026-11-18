"""Purpose: Retrospective held-out next-stride association with gait-phase EEG.
Inputs: outputs/stats/revision_2026-10/gait_kinematics_cycles_2026-10-13.csv;
        config.yaml; data/derivatives/{gerd,autoreject} pre8walk exports.
Outputs: outputs/stats/revision_2026-10/next_stride_prediction_*_2026-10-20.*.
Author: Lena
Date: 2026-10-20
Seed: 42, overridden by config repro.seed; logged in all sidecars.

Existing zero-phase filtering and centered Morlet summaries use future samples.
All scores are retrospective association, not prospective or causal prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, setup_run, write_sidecar  # noqa: E402

DATE = "2026-10-20"
PHASES = ("IDS", "LLS", "FDS", "RLS")
AREAS = ("motor", "frontal")
BANDS = ("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma")
TIMINGS = (
    "double_support_initial_s",
    "swing_left_s",
    "double_support_terminal_s",
    "swing_right_s",
)
MODELS = ("gait_history", "plus_cycle_mean", "plus_phase_contrasts")
STATUS = "retrospective_association_only"
BLOCKER = (
    "Prospective interpretation blocked: zero-phase FIR and centered continuous "
    "Morlet features include future samples; native cleaned highpass 2 Hz versus "
    "manuscript 4 Hz provenance remains unresolved. Reconstruct causal features "
    "from audited inputs before any prospective claim. Earlier-cycle sensitivity "
    "does not establish causality. Gait-locked movement artefacts and feature "
    "extraction can drive association; without independent movement proxies, "
    "improvement does not demonstrate neural control."
)


def validate_alignment(gait, neural, keep_mask, subject):
    """Demand exact original epoch identity across all three sources."""
    required = {"cycle_index", "kept", "t_rhs_s", "stride_time_s", *TIMINGS}
    if required - set(gait):
        raise ValueError(f"Missing gait columns: {sorted(required - set(gait))}")
    gait = gait.sort_values("cycle_index").copy()
    cycles = gait.cycle_index.to_numpy()
    mask = np.asarray(keep_mask)
    if mask.ndim != 1 or mask.dtype.kind != "b":
        raise ValueError("Keep mask must be a one-dimensional boolean array")
    if not np.array_equal(cycles, np.arange(len(mask))):
        raise ValueError("Gait original-cycle indices do not exactly match mask extent")
    if not gait.kept.isin([True, False]).all():
        raise ValueError("Gait kept values are not boolean")
    if not np.array_equal(gait.kept.to_numpy(bool), mask):
        raise ValueError("Gait kept flags disagree with mask")
    if set(neural.SubjectID.dropna()) != {subject}:
        raise ValueError("Neural SubjectID mismatch")
    if set(neural.Block.dropna()) != {"pre8walk"} or set(neural.Modality.dropna()) != {
        "eeg"
    }:
        raise ValueError("Unexpected neural block or modality")
    epochs = neural.Epoch.to_numpy(float)
    if not np.isfinite(epochs).all() or not np.equal(epochs, np.floor(epochs)).all():
        raise ValueError("Neural Epoch must contain finite original integer indices")
    expected = set(np.flatnonzero(mask))
    if set(epochs.astype(int)) != expected:
        raise ValueError(
            "Neural epoch set differs from mask-kept original cycle indices"
        )
    selected = neural[(neural.Hemisphere == "Both") & neural.Area.isin(AREAS)].copy()
    if selected.duplicated(["Epoch", "GaitPhase", "Area"]).any():
        raise ValueError("Duplicate neural epoch/phase/area rows")
    if not selected.GaitPhase.isin(PHASES).all():
        raise ValueError("Unknown selected gait phase")
    features, incomplete = {}, []
    index = pd.MultiIndex.from_product([PHASES, AREAS], names=["GaitPhase", "Area"])
    columns = [f"ERD_ERS_{band}" for band in BANDS]
    for epoch in sorted(expected):
        rows = selected[selected.Epoch == epoch].set_index(["GaitPhase", "Area"])
        values = rows.reindex(index)[columns].to_numpy(float).reshape(4, 10)
        if np.isfinite(values).all():
            features[epoch] = values
        else:
            incomplete.append(epoch)
    return gait.set_index("cycle_index"), features, incomplete


def neural_features(values):
    """Ten means and thirty full-rank, orthonormal four-phase contrasts."""
    helmert = (
        np.array([[1, -1, 0, 0], [1, 1, -2, 0], [1, 1, 1, -3]], dtype=float)
        / np.sqrt(np.array([2, 6, 12]))[:, None]
    )
    return values.mean(axis=0), (helmert @ values).reshape(-1)


def build_samples(gait, features, lag=0):
    """Target i+1; gait controls i,i-1; neural cycle i-lag, no gap crossing."""
    if lag not in (0, 2):
        raise ValueError("Only primary lag 0 and earlier-cycle lag 2 are specified")
    rows, counts = [], {
        "candidate": 0,
        "gap_or_rejected": 0,
        "missing_neural": 0,
        "invalid_timing": 0,
    }
    earliest = max(1, lag)
    for current in gait.index:
        if current < earliest or current + 1 not in gait.index:
            continue
        counts["candidate"] += 1
        start, end = current - earliest, current + 1
        indices = np.arange(start, end + 1)
        if not np.isin(indices, gait.index).all():
            counts["gap_or_rejected"] += 1
            continue
        window = gait.loc[indices]
        adjacency = (
            window.t_rhs_s.to_numpy()[:-1] + window.stride_time_s.to_numpy()[:-1]
        )
        if not window.kept.all() or not np.allclose(
            adjacency, window.t_rhs_s.to_numpy()[1:], atol=1e-5, rtol=0
        ):
            counts["gap_or_rejected"] += 1
            continue
        neural_cycle = current - lag
        if neural_cycle not in features:
            counts["missing_neural"] += 1
            continue
        now = gait.loc[current]
        timings = now[list(TIMINGS)].to_numpy(float)
        # Three independent phase durations plus total duration are full rank.
        controls = np.r_[
            now.stride_time_s, gait.loc[current - 1, "stride_time_s"], timings[:3]
        ]
        target = float(gait.loc[current + 1, "stride_time_s"])
        if (
            not np.isfinite(np.r_[controls, timings, target]).all()
            or (timings <= 0).any()
            or target <= 0
        ):
            counts["invalid_timing"] += 1
            continue
        if not np.isclose(timings.sum(), now.stride_time_s, atol=1e-5, rtol=0):
            counts["invalid_timing"] += 1
            continue
        means, contrasts = neural_features(features[neural_cycle])
        rows.append(
            dict(
                current_cycle=int(current),
                feature_cycle=int(neural_cycle),
                target_cycle=int(current + 1),
                first_cycle=int(start),
                last_cycle=int(end),
                target_s=target,
                controls=controls,
                means=means,
                contrasts=contrasts,
            )
        )
    return rows, counts


def blocked_folds(samples, n_splits=5, purge=5, min_train=20, min_test=5):
    """Contiguous time test blocks; every train/test used-cycle distance > purge."""
    order = np.argsort([s["current_cycle"] for s in samples])
    for fold, test in enumerate(np.array_split(order, n_splits)):
        if len(test) < min_test:
            continue
        low = min(samples[i]["first_cycle"] for i in test)
        high = max(samples[i]["last_cycle"] for i in test)
        train = np.array(
            [
                i
                for i in order
                if samples[i]["last_cycle"] < low - purge
                or samples[i]["first_cycle"] > high + purge
            ],
            dtype=int,
        )
        if len(train) >= min_train:
            yield fold, train, test


def model_matrix(samples, model):
    parts = [np.stack([s["controls"] for s in samples])]
    if model in MODELS[1:]:
        parts.append(np.stack([s["means"] for s in samples]))
    if model == MODELS[2]:
        parts.append(np.stack([s["contrasts"] for s in samples]))
    return np.concatenate(parts, axis=1)


def fit_predict(x_train, y_train, x_test):
    """Train-only feature scaling and fixed penalty; targets remain in seconds."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x_train, y_train)
    return model.predict(x_test), model


def evaluate(samples, subject, cohort, lag):
    predictions, metrics = [], []
    folds = list(blocked_folds(samples))
    if not folds:
        return predictions, metrics
    y = np.array([s["target_s"] for s in samples])
    for name in MODELS:
        x = model_matrix(samples, name)
        cells = []
        for fold, train, test in folds:
            predicted, _ = fit_predict(x[train], y[train], x[test])
            for i, value in zip(test, predicted):
                row = dict(
                    subject=subject,
                    cohort=cohort,
                    lag_cycles=lag,
                    model=name,
                    fold=fold,
                    current_cycle=samples[i]["current_cycle"],
                    feature_cycle=samples[i]["feature_cycle"],
                    target_cycle=samples[i]["target_cycle"],
                    truth_s=y[i],
                    prediction_s=value,
                    train_mean_s=y[train].mean(),
                    n_train=len(train),
                    status=STATUS,
                )
                predictions.append(row)
                cells.append(row)
        frame = pd.DataFrame(cells)
        squared = np.square(frame.truth_s - frame.prediction_s).sum()
        baseline = np.square(frame.truth_s - frame.train_mean_s).sum()
        metrics.append(
            dict(
                subject=subject,
                cohort=cohort,
                lag_cycles=lag,
                model=name,
                n_heldout=len(frame),
                n_folds=len(folds),
                mae_s=np.abs(frame.truth_s - frame.prediction_s).mean(),
                oof_r2_vs_training_mean=(
                    1 - squared / baseline if baseline > 0 else np.nan
                ),
                status=STATUS,
            )
        )
    return predictions, metrics


def summarize(metrics, seed, n_resamples):
    """Participant bootstrap CIs and two-sided sign-flip tests, unadjusted."""
    rng = np.random.default_rng(seed)
    paired, summary = [], []
    for (subject, cohort, lag), frame in metrics.groupby(
        ["subject", "cohort", "lag_cycles"]
    ):
        errors = frame.set_index("model").mae_s
        for simpler, richer in [
            (MODELS[0], MODELS[1]),
            (MODELS[1], MODELS[2]),
            (MODELS[0], MODELS[2]),
        ]:
            paired.append(
                dict(
                    subject=subject,
                    cohort=cohort,
                    lag_cycles=lag,
                    comparison=f"{richer}_vs_{simpler}",
                    improvement_s=float(errors[simpler] - errors[richer]),
                    status=STATUS,
                )
            )
    paired = pd.DataFrame(paired)
    for (lag, comparison), frame in paired.groupby(["lag_cycles", "comparison"]):
        for cohort, group in [("all", frame), *list(frame.groupby("cohort"))]:
            values = group.improvement_s.to_numpy()
            boot = rng.choice(values, (n_resamples, len(values))).mean(axis=1)
            null = (rng.choice([-1, 1], (n_resamples, len(values))) * values).mean(
                axis=1
            )
            low, high = np.quantile(boot, [0.025, 0.975])
            summary.append(
                dict(
                    lag_cycles=lag,
                    comparison=comparison,
                    cohort=cohort,
                    n_subjects=len(values),
                    improvement_s=values.mean(),
                    ci_low_s=low if len(values) >= 3 else np.nan,
                    ci_high_s=high if len(values) >= 3 else np.nan,
                    two_sided_sign_flip_p=(
                        (1 + (np.abs(null) >= abs(values.mean())).sum())
                        / (n_resamples + 1)
                        if len(values) >= 3
                        else np.nan
                    ),
                    inference_status=(
                        "exploratory" if len(values) >= 3 else "descriptive_only_n_lt_3"
                    ),
                    multiplicity="unadjusted_exploratory",
                    status=STATUS,
                )
            )
    return paired, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.n_resamples < 100:
        parser.error("--n-resamples must be at least 100")
    config = load_config()
    seed = int(config["repro"]["seed"])
    meta = setup_run("next_stride_prediction", config)
    versions = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scikit-learn", "PyYAML"]
    }
    print(json.dumps(dict(seed=seed, versions=versions)))
    derivatives = PROJECT_ROOT / config["paths"]["derivatives_root"]
    gait_path = (
        PROJECT_ROOT
        / "outputs/stats/revision_2026-10/gait_kinematics_cycles_2026-10-13.csv"
    )
    if not gait_path.is_file():
        raise FileNotFoundError(gait_path)
    gait_all = pd.read_csv(gait_path)
    outdir = PROJECT_ROOT / "outputs/stats/revision_2026-10"
    outdir.mkdir(parents=True, exist_ok=True)
    meta.update(
        versions=versions,
        config=config,
        input_sha256={},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status=STATUS,
        prospective_blocker=BLOCKER,
        parameters=dict(
            n_splits=5,
            purge_cycles=5,
            min_train=20,
            min_test=5,
            ridge_alpha=10,
            n_resamples=args.n_resamples,
        ),
        config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
    )

    def record_hash(path):
        meta["input_sha256"][str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    record_hash(gait_path)
    predictions, metrics, exclusions = [], [], []
    for cohort, subjects in config["cohorts"].items():
        for subject in subjects:
            try:
                neural_path = (
                    derivatives / "gerd" / f"{subject}_pre8walk-eeg-basenorm-gerd.csv"
                )
                mask_path = derivatives / "autoreject" / f"{subject}_pre8walk-mask.npz"
                for path in (neural_path, mask_path):
                    if not path.is_file():
                        raise FileNotFoundError(path.relative_to(PROJECT_ROOT))
                    record_hash(path)
                neural = pd.read_csv(neural_path)
                if set(neural.Cohort.dropna()) != {cohort}:
                    raise ValueError("Neural cohort differs from config")
                gait = gait_all[gait_all.subject == subject]
                expected = "PD" if cohort == "Patient_NFB" else "HC"
                if set(gait.cohort.dropna()) != {expected}:
                    raise ValueError("Gait cohort differs from config")
                with np.load(mask_path) as archive:
                    mask = archive["keep_mask"]
                gait, features, incomplete = validate_alignment(
                    gait, neural, mask, subject
                )
                for lag in (0, 2):
                    samples, counts = build_samples(gait, features, lag)
                    pred, result = evaluate(samples, subject, cohort, lag)
                    predictions.extend(pred)
                    metrics.extend(result)
                    exclusions.append(
                        dict(
                            subject=subject,
                            cohort=cohort,
                            lag_cycles=lag,
                            status=(
                                "included" if result else "insufficient_purged_folds"
                            ),
                            incomplete_neural_cycles=len(incomplete),
                            incomplete_original_cycle_indices=json.dumps(incomplete),
                            n_samples=len(samples),
                            n_heldout=result[0]["n_heldout"] if result else 0,
                            **counts,
                        )
                    )
            except (
                ValueError,
                KeyError,
                FileNotFoundError,
                pd.errors.ParserError,
            ) as error:
                exclusions.append(
                    dict(
                        subject=subject,
                        cohort=cohort,
                        status="invalid_or_missing_input",
                        detail=str(error),
                    )
                )
    outputs = {"exclusions": pd.DataFrame(exclusions)}
    if metrics:
        metric_frame = pd.DataFrame(metrics)
        paired, summary = summarize(metric_frame, seed, args.n_resamples)
        outputs.update(
            metrics=metric_frame,
            predictions=pd.DataFrame(predictions),
            participant_improvements=paired,
            summary=summary,
        )
    for suffix, frame in outputs.items():
        path = outdir / f"next_stride_prediction_{suffix}_{DATE}.csv"
        frame.to_csv(path, index=False)
        write_sidecar(path, meta)
    report = outdir / f"next_stride_prediction_report_{DATE}.md"
    report.write_text(
        "# Retrospective next-stride association\n\n"
        + BLOCKER
        + "\n\nFixed models: two completed stride durations and three current-cycle phase durations; add ten four-phase means; then thirty orthonormal phase contrasts. Five contiguous test blocks; train cycles more than five cycles away from every test used-cycle interval. No tuning. Targets in seconds. Primary neural cycle i predicts target i+1 offline; earlier-cycle sensitivity uses i-2 with all intervening cycles retained and physically adjacent.\n\nParticipant mean MAE improvements are positive when the richer model improves. Bootstrap participants, not cycles. Sign-flip p-values are two-sided and unadjusted; cohorts are descriptive exploratory strata. This is within-participant blocked generalization, not unseen-participant validation. Pooled out-of-fold R2 uses each fold's training mean as reference. Minimum 20 training and five test rows; every exclusion is exported. Purging does not repair upstream acausal transforms or unresolved preprocessing provenance.\n\n"
        + (
            f"Computed {len(metrics)} participant/model/lag cells. See companion CSVs.\n"
            if metrics
            else "No participant has eligible purged folds; inspect exclusions.\n"
        )
    )
    write_sidecar(report, meta)
    if not metrics:
        raise RuntimeError("No eligible participant; exclusions and report saved")


if __name__ == "__main__":
    main()
