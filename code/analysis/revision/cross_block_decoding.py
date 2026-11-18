"""Purpose: Offline gait-phase decoding across walking blocks with known boundaries.
Inputs: data/derivatives/gerd/*_{pre8walk,post8walk}-eeg-basenorm-gerd.csv;
        code/analysis/config.yaml (paths, cohort roster, seed).
Outputs: outputs/stats/revision_2026-10/cross_block_decoding_*_2026-10-20.*.
Author: Lena
Date: 2026-10-20
Seed: 42 default; config repro.seed or --seed; logged in every sidecar.

Four phases are centered within each complete cycle, separately for every
area/band. This cancels constant baseline/export offsets without estimating
any block reference. It uses all four phase summaries and is NOT prospective.
The whole-cycle comparator repeats each cycle's uncentered phase mean.
StandardScaler and logistic regression are fitted on the training block only.
Circular label rotations are independent across cycles and across blocks;
each surrogate refits the classifier, preserving cyclic label order.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
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
BANDS = ("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma")
AREAS = ("motor", "frontal")
BLOCKS = ("pre8walk", "post8walk")
DECODERS = ("4class", "DSvsSwing")


def build_samples(df: pd.DataFrame, subject: str, block: str):
    """Build complete-cycle tensors, rejecting duplicates and incomplete cycles.

    Returns centered features, repeated cycle means, phase labels, cycle keys,
    and counts. Each feature is one area/band; rows are in cyclic phase order.
    """
    cols = [f"ERD_ERS_{b}" for b in BANDS]
    required = [
        "SubjectID",
        "Cohort",
        "Block",
        "Modality",
        "Area",
        "Hemisphere",
        "Epoch",
        "GaitPhase",
        *cols,
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for col, expected in [
        ("SubjectID", subject),
        ("Block", block),
        ("Modality", "eeg"),
    ]:
        if set(df[col].dropna()) != {expected}:
            raise ValueError(f"Unexpected {col} values")
    selected = df[(df.Hemisphere == "Both") & df.Area.isin(AREAS)].copy()
    if selected.empty or not selected.GaitPhase.isin(PHASES).all():
        raise ValueError("Missing selected EEG data or unknown gait phase")
    if selected.duplicated(["Epoch", "GaitPhase", "Area"]).any():
        raise ValueError("Duplicate cycle/phase/area rows")
    epochs = sorted(selected.Epoch.unique())
    complete, kept = [], []
    for epoch in epochs:
        rows = selected[selected.Epoch == epoch].set_index(["GaitPhase", "Area"])
        index = pd.MultiIndex.from_product([PHASES, AREAS], names=["GaitPhase", "Area"])
        values = rows.reindex(index)[cols].to_numpy(float)
        if not np.isfinite(values).all():
            continue
        complete.append(values.reshape(4, len(AREAS) * len(BANDS)))
        kept.append(epoch)
    if len(kept) < 2:
        raise ValueError("Fewer than two complete finite cycles")
    tensor = np.stack(complete)
    cycle_mean = tensor.mean(axis=1, keepdims=True)
    centered = (tensor - cycle_mean).reshape(-1, 10)
    whole_cycle = np.repeat(cycle_mean, 4, axis=1).reshape(-1, 10)
    phase = np.tile(np.asarray(PHASES), len(kept))
    # Epoch numbers can recur between blocks: qualify them by block.
    keys = np.repeat([f"{block}:{epoch}" for epoch in kept], 4)
    return (
        centered,
        whole_cycle,
        phase,
        keys,
        {
            "n_cycles_available": len(epochs),
            "n_cycles_kept": len(kept),
            "n_cycles_dropped": len(epochs) - len(kept),
        },
    )


def labels_for(phase: np.ndarray, decoder: str) -> np.ndarray:
    if decoder == "4class":
        return phase.copy()
    if decoder == "DSvsSwing":
        return np.where(np.isin(phase, ("IDS", "FDS")), "DS", "Swing")
    raise ValueError(decoder)


def rotate_labels(phase: np.ndarray, cycles: np.ndarray, rng) -> np.ndarray:
    """Independently rotate each four-phase cycle, allowing zero rotation."""
    rotated = phase.copy()
    for cycle in np.unique(cycles):
        idx = np.flatnonzero(cycles == cycle)
        if len(idx) != 4 or tuple(phase[idx]) != PHASES:
            raise ValueError("Surrogates require complete cycles in cyclic phase order")
        rotated[idx] = np.roll(phase[idx], rng.integers(4))
    return rotated


def fit_predict(x_train, y_train, x_test, seed=SEED):
    """Fit both preprocessing and model exclusively on training samples."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed),
    )
    model.fit(x_train, y_train)
    return model.predict(x_test), model


def decode_pair(train, test, decoder, feature, seed, n_perm):
    """Evaluate one subject/direction/feature, retaining surrogate scores."""
    if set(train[3]) & set(test[3]):
        raise ValueError("Train/test cycle keys overlap")
    index = 0 if feature == "phase_centered" else 1
    xt, xv = train[index], test[index]
    yt, yv = labels_for(train[2], decoder), labels_for(test[2], decoder)
    pred, _ = fit_predict(xt, yt, xv, seed)
    score = balanced_accuracy_score(yv, pred)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for iteration in range(n_perm):
        st = labels_for(rotate_labels(train[2], train[3], rng), decoder)
        sv = labels_for(rotate_labels(test[2], test[3], rng), decoder)
        surrogate, _ = fit_predict(xt, st, xv, seed)
        null[iteration] = balanced_accuracy_score(sv, surrogate)
    result = {
        "balanced_accuracy": score,
        "macro_f1": f1_score(yv, pred, average="macro"),
        "chance": 0.25 if decoder == "4class" else 0.5,
        "n_train_cycles": len(np.unique(train[3])),
        "n_test_cycles": len(np.unique(test[3])),
        "n_test_samples": len(yv),
        "n_features": xt.shape[1],
        "perm_p": (1 + np.sum(null >= score)) / (n_perm + 1) if n_perm else np.nan,
        "perm_mean": null.mean() if n_perm else np.nan,
    }
    predictions = pd.DataFrame(
        {"test_cycle": test[3], "phase": test[2], "truth": yv, "prediction": pred}
    )
    return result, predictions, null


def mean_ci(values, rng, n_boot=5000):
    """Participant bootstrap CI: cycles are never treated as participants."""
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, (n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975])


def summarize(metrics, nulls, seed):
    rng = np.random.default_rng(seed)
    summaries = []
    keys = ["direction", "decoder", "feature", "cohort"]
    for key, frame in metrics.groupby(keys, sort=True):
        mean, low, high = mean_ci(frame.balanced_accuracy, rng)
        group_null = np.array([nulls[int(i)] for i in frame.cell_id])
        count = group_null.shape[1]
        group_null = group_null.mean(axis=0) if count else np.array([])
        summaries.append(
            dict(
                zip(keys, key),
                n_subjects=len(frame),
                mean_balanced_accuracy=mean,
                ci_low=low,
                ci_high=high,
                participant_rotation_p=(
                    (1 + np.sum(group_null >= mean)) / (count + 1) if count else np.nan
                ),
                n_subjects_perm_p_lt_05=int((frame.perm_p < 0.05).sum()),
            )
        )
    comparisons = []
    for key, frame in metrics.groupby(keys[:-1], sort=True):
        pd_values = frame.loc[
            frame.cohort == "Patient_NFB", "balanced_accuracy"
        ].to_numpy()
        hc_values = frame.loc[
            frame.cohort == "AgeMatched_NFB", "balanced_accuracy"
        ].to_numpy()
        if min(len(pd_values), len(hc_values)) < 2:
            continue
        effect = pd_values.mean() - hc_values.mean()
        pooled = np.r_[pd_values, hc_values]
        differences = []
        for _ in range(5000):
            shuffled = rng.permutation(pooled)
            differences.append(
                shuffled[: len(pd_values)].mean() - shuffled[len(pd_values) :].mean()
            )
        comparisons.append(
            dict(
                zip(keys[:-1], key),
                n_PD=len(pd_values),
                n_HC=len(hc_values),
                PD_minus_HC=effect,
                exploratory_two_sided_p=(1 + np.sum(np.abs(differences) >= abs(effect)))
                / 5001,
            )
        )
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.n_perm < 0:
        parser.error("--n-perm must be nonnegative")
    config = load_config()
    seed = int(config["repro"]["seed"] if args.seed is None else args.seed)
    config["repro"]["seed"] = seed
    meta = setup_run("cross_block_decoding", config)
    versions = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scipy", "scikit-learn", "PyYAML"]
    }
    print(json.dumps({"seed": seed, "versions": versions}))
    gerd = PROJECT_ROOT / config["paths"]["derivatives_root"] / "gerd"
    if not gerd.is_dir():
        raise FileNotFoundError(gerd)
    outdir = PROJECT_ROOT / "outputs/stats/revision_2026-10"
    outdir.mkdir(parents=True, exist_ok=True)
    meta.update(
        versions=versions,
        config=config,
        parameters=vars(args),
        config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={},
        normalization="per-cycle four-phase mean subtraction",
        primary="pre8walk_to_post8walk / 4class / phase_centered",
        secondary="DSvsSwing",
        sensitivity="post8walk_to_pre8walk",
        classifier="StandardScaler(training only) + LogisticRegression(C=1,lbfgs)",
    )
    rows, predictions, exclusions, nulls = [], [], [], {}
    for cohort, subjects in config["cohorts"].items():
        for subject in subjects:
            blocks, failed = {}, False
            for block in BLOCKS:
                path = gerd / f"{subject}_{block}-eeg-basenorm-gerd.csv"
                if not path.is_file():
                    exclusions.append(
                        dict(
                            subject=subject,
                            cohort=cohort,
                            block=block,
                            status="missing_input",
                            detail=path.name,
                        )
                    )
                    failed = True
                    continue
                meta["input_sha256"][str(path.relative_to(PROJECT_ROOT))] = (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                )
                try:
                    table = pd.read_csv(path)
                    if set(table["Cohort"].dropna()) != {cohort}:
                        raise ValueError("Cohort disagrees with configured roster")
                    blocks[block] = build_samples(table, subject, block)
                    exclusions.append(
                        dict(
                            subject=subject,
                            cohort=cohort,
                            block=block,
                            status="available",
                            detail="",
                            **blocks[block][4],
                        )
                    )
                except (ValueError, KeyError, pd.errors.ParserError) as error:
                    exclusions.append(
                        dict(
                            subject=subject,
                            cohort=cohort,
                            block=block,
                            status="invalid_input",
                            detail=str(error),
                        )
                    )
                    failed = True
            if failed:
                continue
            for source, target in [BLOCKS, BLOCKS[::-1]]:
                for decoder in DECODERS:
                    for feature in ["phase_centered", "whole_cycle_only"]:
                        cell_id = len(rows)
                        cell = dict(
                            subject=subject,
                            cohort=cohort,
                            direction=f"{source}_to_{target}",
                            decoder=decoder,
                            feature=feature,
                            cell_id=cell_id,
                        )
                        result, prediction, null = decode_pair(
                            blocks[source],
                            blocks[target],
                            decoder,
                            feature,
                            seed + cell_id,
                            args.n_perm,
                        )
                        rows.append({**cell, **result})
                        predictions.append(prediction.assign(**cell))
                        nulls[cell_id] = null
            print(f"Completed {subject}: matched EEG blocks", flush=True)
    if not rows:
        pd.DataFrame(exclusions).to_csv(
            outdir / f"cross_block_decoding_exclusions_{DATE}.csv", index=False
        )
        raise RuntimeError("No usable matched participants; inspect exclusions")
    metrics = pd.DataFrame(rows)
    summary, comparisons = summarize(metrics, nulls, seed)
    null_frame = pd.DataFrame(
        [
            dict(cell_id=cell, iteration=i, balanced_accuracy=value)
            for cell, values in nulls.items()
            for i, value in enumerate(values)
        ]
    )
    outputs = {
        "per_subject": metrics,
        "predictions": pd.concat(predictions),
        "summary": summary,
        "cohort_comparison_exploratory": comparisons,
        "exclusions": pd.DataFrame(exclusions),
        "surrogates": null_frame,
    }
    for suffix, frame in outputs.items():
        output = outdir / f"cross_block_decoding_{suffix}_{DATE}.csv"
        frame.to_csv(output, index=False)
        write_sidecar(output, meta)
    version_path = outdir / f"cross_block_decoding_versions_{DATE}.json"
    version_path.write_text(json.dumps(versions, indent=2))
    write_sidecar(version_path, meta)
    report = outdir / f"cross_block_decoding_report_{DATE}.md"
    report.write_text(
        "# Offline cross-block gait-phase decoding\n\n"
        "Primary: pre8walk → post8walk, four phases, motor+frontal Both EEG, five bands.\n"
        "Secondary: support versus swing. Reverse direction is sensitivity analysis.\n"
        "Participants must have complete finite cycles in both blocks; all their usable cycles "
        "are retained in each block (cycle counts need not equal). Subject identities are paired "
        "across blocks; epoch numbers denote different cycles and are never matched across blocks.\n"
        "Cycle-centered inputs subtract the arithmetic mean of the four phase summaries per "
        "area/band. This is distinct from the existing avggait export, whose reference uses all "
        "cycles in its block. Constant offsets (including the documented HC post8walk +120 dB "
        "export offset) cancel. Frequency/channel dependent preprocessing differences can remain.\n"
        "Scaling is fitted on training samples only. Whole-cycle comparator repeats each "
        "uncentered cycle mean across phases. Surrogates rotate cyclic labels independently "
        "within every train/test cycle and refit. Group rotation evidence averages participants "
        "with equal weight. Confidence intervals bootstrap participants (5000 draws). Cohort "
        "comparisons use 5000 participant-label permutations and are exploratory, uncorrected.\n"
        "The whole-cycle comparator repeats identical features for all equally represented "
        "phases and is at chance by construction. It is a structural sanity check, not "
        "evidence of added behavioural or clinical value. Independent cycle rotations preserve "
        "within-cycle cyclic order, not dependence across cycles; rotation p-values refer to "
        "this surrogate null and cannot establish independent population replication.\n"
        "\nThis uses known gait boundaries and all phases of each cycle: it is an offline "
        "representation test, not prospective prediction or independent replication. Both blocks "
        "are from one session and have different preprocessing exports. No movement covariates "
        "are included, so decoding cannot establish a neural rather than movement-artifact origin. "
        "These analyses were selected after reviewing existing results. All secondary, reverse "
        "direction and comparator results must be reported, including null results.\n\n"
        f"Seed: {seed}; rotations per participant/cell: {args.n_perm}.\n"
        "See summary CSV for participant means, bootstrap CIs, rotation evidence; exclusions CSV "
        "for every configured participant/block. Subject prediction and surrogate files support audit.\n"
    )
    write_sidecar(report, meta)


if __name__ == "__main__":
    main()
