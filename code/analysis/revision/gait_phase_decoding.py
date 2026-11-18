"""Single-trial decoding of gait phase from cortical and STN band power.

Purpose:  The cohort-level models show gait-phase modulation of cortical
          power that is present and similar in PD and HC. This script asks
          the complementary individual-level question: how much gait-phase
          information does each participant's cortical (and STN) activity
          carry on single cycles, and does decodability differ between
          cohorts?
Inputs:   data/derivatives/gerd/<subj>_pre8walk-eeg-{avggait,basenorm}-gerd.csv
          data/derivatives/gerd/<subj>_pre8walk-dbs-{avggait,basenorm}-gerd.csv
              (compute_gerd_table.py; one row per cycle (Epoch) x Area x
              Hemisphere x GaitPhase with five band columns ERD_ERS_<band>, dB)
          code/analysis/config.yaml (repro.seed, derivatives root)
Outputs:  outputs/stats/revision_2026-10/gait_phase_decoding_per_subject_2026-10-14.csv
          outputs/stats/revision_2026-10/gait_phase_decoding_2026-10-14.csv   (group table)
          outputs/stats/revision_2026-10/gait_phase_decoding_confusion_2026-10-14.csv
          outputs/stats/revision_2026-10/gait_phase_decoding_feature_importance_2026-10-14.csv
          outputs/stats/revision_2026-10/gait_phase_decoding_learning_curve_2026-10-14.csv
          outputs/stats/revision_2026-10/gait_phase_decoding_report_2026-10-14.md
          outputs/figures/revision_2026-10/gait_phase_decoding_accuracy.png
          outputs/figures/revision_2026-10/gait_phase_decoding_confusion.png
          outputs/figures/revision_2026-10/gait_phase_decoding_feature_importance.png
          outputs/figures/revision_2026-10/gait_phase_decoding_learning_curve.png
Author:   Lena
Date:     2026-10-14
Seed:     42 (config.yaml repro.seed via _common.setup_run; numpy.random.seed(42);
          every scikit-learn random_state = 42)

Description:
  Samples are (cycle, gait phase) pairs. Feature sets (all five bands,
  Theta, Alpha, Low_Beta, High_Beta, Gamma):
    eeg_both_motor_frontal   Hemisphere == Both, motor + frontal      (10 features)
    eeg_both_motor           Hemisphere == Both, motor                ( 5)
    eeg_both_frontal         Hemisphere == Both, frontal              ( 5)
    eeg_LR_motor             Left + Right, motor                      (10)
    eeg_LR_motor_frontal     Left + Right, motor + frontal            (20)
    stn                      STN, the one recorded hemisphere (PD)    ( 5)
  Decoders:
    4class       IDS / LLS / FDS / RLS                (chance 25 %)
    LLSvsRLS     left-leg swing vs right-leg swing     (chance 50 %)
    DSvsSwing    IDS + FDS vs LLS + RLS                (chance 50 %)
    contra_ipsi  STN only: contralateral vs ipsilateral swing, i.e.
                 LLSvsRLS relabelled by the recorded hemisphere (left STN:
                 RLS contralateral; right STN: LLS contralateral)
  Classifiers: multinomial logistic regression (StandardScaler, C = 1,
  lbfgs) and LDA (StandardScaler, svd solver). Cross-validation:
  StratifiedGroupKFold(5), group = cycle, so all phases of a cycle fall
  into the same fold. Metrics on pooled out-of-fold predictions: balanced
  accuracy, macro-F1. Permutation baseline: 200 label permutations within
  cycle (same folds), p = (#perm >= observed + 1) / (n_perm + 1).
  Normalisations: avggait (primary, phase information only) and basenorm.
  Group comparison per decoder x feature set x normalisation x classifier:
  Welch t, Mann-Whitney U, Hedges g, proportion of subjects with
  permutation p < 0.05. Feature importance: mean |standardised coefficient|
  across classes from a fit on all cycles, averaged per cohort.
  Learning curve: balanced accuracy versus number of training cycles
  (20, 40, 60, all) for the primary decoder, five random subsamples of the
  training fold per outer fold.

Usage:
  python code/analysis/revision/gait_phase_decoding.py [--n-perm 200] [--n-jobs 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, resolve_paths, setup_run, write_sidecar  # noqa: E402

SEED = 42
DATE = "2026-10-14"
PHASES = ("IDS", "LLS", "FDS", "RLS")
BANDS = ("Theta", "Alpha", "Low_Beta", "High_Beta", "Gamma")
PD, HC = "Patient_NFB", "AgeMatched_NFB"
NORMS = ("avggait", "basenorm")
N_FOLDS = 5

FEATURE_SETS = {
    # name: (modality, [(area, hemisphere), ...])
    "eeg_both_motor_frontal": ("eeg", [("motor", "Both"), ("frontal", "Both")]),
    "eeg_both_motor": ("eeg", [("motor", "Both")]),
    "eeg_both_frontal": ("eeg", [("frontal", "Both")]),
    "eeg_LR_motor": ("eeg", [("motor", "Left"), ("motor", "Right")]),
    "eeg_LR_motor_frontal": ("eeg", [("motor", "Left"), ("motor", "Right"), ("frontal", "Left"), ("frontal", "Right")]),
    "stn": ("dbs", [("STN", None)]),  # None = whichever hemisphere was recorded
}
DECODERS = {
    "4class": {p: p for p in PHASES},
    "LLSvsRLS": {"LLS": "LLS", "RLS": "RLS"},
    "DSvsSwing": {"IDS": "DS", "FDS": "DS", "LLS": "Swing", "RLS": "Swing"},
}
CHANCE = {"4class": 0.25, "LLSvsRLS": 0.5, "DSvsSwing": 0.5, "contra_ipsi": 0.5}
PRIMARY = dict(feature_set="eeg_both_motor_frontal", norm="avggait", clf="logreg")


def make_clf(name: str):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
    if name == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd"))
    raise ValueError(name)


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (na + nb) - 9))


# ----------------------------------------------------------------------------- data
def load_subject_tables(gerd_dir: Path) -> dict:
    """Return {(subject, modality, norm): DataFrame} for every gerd CSV."""
    tables = {}
    for f in sorted(gerd_dir.glob("P4-*_pre8walk-*-gerd.csv")):
        parts = f.stem.split("_")[1].split("-")  # pre8walk, eeg|dbs, avggait|basenorm, gerd
        modality, norm = parts[1], parts[2]
        df = pd.read_csv(f)
        if df.empty:
            continue
        tables[(df["SubjectID"].iloc[0], modality, norm)] = df
    return tables


def build_samples(df: pd.DataFrame, blocks: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, str]:
    """Return X (n_samples, n_features), phase labels, cycle ids, feature names, hemisphere used."""
    cols = [f"ERD_ERS_{b}" for b in BANDS]
    pieces, names, hemi_used = [], [], ""
    for area, hemi in blocks:
        sel = df[df["Area"] == area]
        if hemi is None:
            hemis = sorted(sel["Hemisphere"].unique())
            assert len(hemis) == 1, f"expected one STN hemisphere, got {hemis}"
            hemi = hemis[0]
            hemi_used = hemi
        sel = sel[sel["Hemisphere"] == hemi]
        wide = sel.pivot_table(index=["Epoch", "GaitPhase"], values=cols, aggfunc="first")[cols]
        wide.columns = [f"{area}_{hemi}_{b}" for b in BANDS] if hemi != "Both" else [f"{area}_{b}" for b in BANDS]
        pieces.append(wide)
        names += list(wide.columns)
    wide = pd.concat(pieces, axis=1, join="inner").dropna()
    # keep only cycles that have all four phases
    counts = wide.groupby(level="Epoch").size()
    full = counts[counts == len(PHASES)].index
    wide = wide[wide.index.get_level_values("Epoch").isin(full)].sort_index()
    X = wide.to_numpy(float)
    phase = wide.index.get_level_values("GaitPhase").to_numpy()
    cycle = wide.index.get_level_values("Epoch").to_numpy()
    return X, phase, cycle, names, hemi_used


def decoder_labels(phase: np.ndarray, decoder: str, hemi: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Map phases to decoder classes; returns (mask of samples used, labels)."""
    if decoder == "contra_ipsi":
        contra = "RLS" if hemi == "Left" else "LLS"
        mapping = {contra: "contra", ("LLS" if contra == "RLS" else "RLS"): "ipsi"}
    else:
        mapping = DECODERS[decoder]
    mask = np.array([p in mapping for p in phase])
    y = np.array([mapping[p] for p in phase[mask]])
    return mask, y


# ----------------------------------------------------------------------------- CV
def folds_for(y: np.ndarray, groups: np.ndarray):
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y, groups))


def oof_predict(clf_name: str, X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        clf = make_clf(clf_name)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


def scores(y, pred, labels):
    return (balanced_accuracy_score(y, pred), f1_score(y, pred, average="macro", labels=labels))


def permute_within_cycle(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yp = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        yp[idx] = y[idx][rng.permutation(len(idx))]
    return yp


def decode_cell(subject, cohort, feature_set, norm, decoder, clf_name, X, phase, cycle, names, hemi, cell_index, n_perm):
    """One subject x feature set x norm x decoder x classifier cell (cell_index seeds its permutations)."""
    mask, y = decoder_labels(phase, decoder, hemi)
    Xs, groups = X[mask], cycle[mask]
    labels = sorted(np.unique(y), key=lambda s: (list(PHASES).index(s) if s in PHASES else s))
    folds = folds_for(y, groups)
    pred = oof_predict(clf_name, Xs, y, folds)
    bacc, mf1 = scores(y, pred, labels)
    cm = confusion_matrix(y, pred, labels=labels)
    # permutation baseline: labels shuffled within cycle, same folds
    rng = np.random.default_rng([SEED, cell_index])  # deterministic per cell, independent of process hash seed
    perm = np.empty(n_perm)
    for i in range(n_perm):
        yp = permute_within_cycle(y, groups, rng)
        perm[i] = balanced_accuracy_score(yp, oof_predict(clf_name, Xs, yp, folds))
    p_perm = (np.sum(perm >= bacc) + 1) / (n_perm + 1) if n_perm else np.nan
    # feature importance from a fit on all cycles (standardised coefficients)
    full = make_clf(clf_name).fit(Xs, y)
    coef = full[-1].coef_
    importance = np.mean(np.abs(coef), axis=0)
    row = dict(subject=subject, cohort=cohort, feature_set=feature_set, norm=norm, decoder=decoder, clf=clf_name,
               stn_hemisphere=hemi, n_cycles=len(np.unique(groups)), n_samples=len(y), n_features=Xs.shape[1],
               chance=CHANCE[decoder], balanced_accuracy=bacc, macro_f1=mf1,
               perm_mean=perm.mean() if n_perm else np.nan, perm_sd=perm.std(ddof=1) if n_perm > 1 else np.nan,
               perm_p=p_perm, perm_p95=np.quantile(perm, 0.95) if n_perm else np.nan)
    return row, dict(labels=labels, cm=cm), dict(zip(names, importance))


def learning_curve_cell(subject, cohort, X, phase, cycle, sizes=(20, 40, 60, None), n_rep=5):
    """Balanced accuracy (logreg, 4class) versus number of training cycles."""
    mask, y = decoder_labels(phase, "4class")
    Xs, groups = X[mask], cycle[mask]
    folds = folds_for(y, groups)
    rng = np.random.default_rng(SEED)
    out = []
    for size in sizes:
        accs = []
        for tr, te in folds:
            tr_cycles = np.unique(groups[tr])
            if size is not None and size > len(tr_cycles):
                continue
            for _ in range(n_rep if size is not None else 1):
                keep = tr_cycles if size is None else rng.choice(tr_cycles, size, replace=False)
                sub = tr[np.isin(groups[tr], keep)]
                clf = make_clf("logreg").fit(Xs[sub], y[sub])
                accs.append(balanced_accuracy_score(y[te], clf.predict(Xs[te])))
        out.append(dict(subject=subject, cohort=cohort, n_train_cycles=size if size is not None else "all",
                        n_train_cycles_actual=size if size is not None else int(round(len(np.unique(groups)) * (N_FOLDS - 1) / N_FOLDS)),
                        balanced_accuracy=np.mean(accs) if accs else np.nan, n_estimates=len(accs)))
    return out


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    config = load_config(); paths = resolve_paths(config)
    meta = setup_run("gait_phase_decoding", config); np.random.seed(SEED)
    assert int(config["repro"]["seed"]) == SEED
    import sklearn
    meta.update(sklearn=sklearn.__version__, pandas=pd.__version__, n_perm=args.n_perm, n_folds=N_FOLDS)
    out_dir = PROJECT_ROOT / "outputs/stats/revision_2026-10"; fig_dir = PROJECT_ROOT / "outputs/figures/revision_2026-10"
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)

    tables = load_subject_tables(paths.derivatives_root / "gerd")
    subjects = sorted({k[0] for k in tables})
    cohort_of = {k[0]: v["Cohort"].iloc[0] for k, v in tables.items()}

    # ---- build all cells
    jobs = []
    for subj in subjects:
        for fs_name, (modality, blocks) in FEATURE_SETS.items():
            for norm in NORMS:
                key = (subj, modality, norm)
                if key not in tables:
                    continue
                X, phase, cycle, names, hemi = build_samples(tables[key], blocks)
                decoders = ["4class", "LLSvsRLS", "DSvsSwing"] + (["contra_ipsi"] if fs_name == "stn" else [])
                for dec in decoders:
                    for clf_name in ("logreg", "lda"):
                        jobs.append((subj, cohort_of[subj], fs_name, norm, dec, clf_name, X, phase, cycle, names, hemi, len(jobs)))
    print(f"{len(subjects)} subjects, {len(jobs)} decoding cells, {args.n_perm} permutations each", flush=True)
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(decode_cell)(*j, args.n_perm) for j in jobs)
    per_subject = pd.DataFrame([r[0] for r in results])
    conf = {(r[0]["subject"], r[0]["feature_set"], r[0]["norm"], r[0]["decoder"], r[0]["clf"]): r[1] for r in results}
    imp_rows = []
    for r in results:
        for feat, val in r[2].items():
            imp_rows.append(dict(**{k: r[0][k] for k in ("subject", "cohort", "feature_set", "norm", "decoder", "clf")}, feature=feat, mean_abs_coef=val))
    importance = pd.DataFrame(imp_rows)
    importance["area"] = importance["feature"].str.split("_").str[0]
    importance["band"] = importance["feature"].apply(lambda s: next(b for b in BANDS if s.endswith(b)))
    importance["hemisphere"] = importance["feature"].apply(lambda s: next((h for h in ("Left", "Right") if f"_{h}_" in s), "Both"))

    ps_path = out_dir / f"gait_phase_decoding_per_subject_{DATE}.csv"
    per_subject.to_csv(ps_path, index=False); write_sidecar(ps_path, meta)
    imp_path = out_dir / f"gait_phase_decoding_feature_importance_{DATE}.csv"
    importance.to_csv(imp_path, index=False); write_sidecar(imp_path, meta)

    # ---- confusion matrices summed per cohort
    cm_rows = []
    for (subj, fs, norm, dec, clf_name), d in conf.items():
        for i, a in enumerate(d["labels"]):
            for j, b in enumerate(d["labels"]):
                cm_rows.append(dict(subject=subj, cohort=cohort_of[subj], feature_set=fs, norm=norm, decoder=dec, clf=clf_name, true=a, predicted=b, count=int(d["cm"][i, j])))
    cm_df = pd.DataFrame(cm_rows)
    cm_sum = cm_df.groupby(["cohort", "feature_set", "norm", "decoder", "clf", "true", "predicted"], as_index=False)["count"].sum()
    cm_path = out_dir / f"gait_phase_decoding_confusion_{DATE}.csv"
    cm_sum.to_csv(cm_path, index=False); write_sidecar(cm_path, meta)

    # ---- group table
    group_rows = []
    for (fs, norm, dec, clf_name), g in per_subject.groupby(["feature_set", "norm", "decoder", "clf"]):
        a = g[g["cohort"] == PD]; b = g[g["cohort"] == HC]
        row = dict(feature_set=fs, norm=norm, decoder=dec, clf=clf_name, chance=CHANCE[dec], n_pd=len(a), n_hc=len(b),
                   pd_mean=a["balanced_accuracy"].mean(), pd_sd=a["balanced_accuracy"].std(ddof=1),
                   pd_f1_mean=a["macro_f1"].mean(), pd_above_chance=(a["perm_p"] < 0.05).sum(),
                   hc_mean=b["balanced_accuracy"].mean() if len(b) else np.nan, hc_sd=b["balanced_accuracy"].std(ddof=1) if len(b) > 1 else np.nan,
                   hc_f1_mean=b["macro_f1"].mean() if len(b) else np.nan, hc_above_chance=(b["perm_p"] < 0.05).sum() if len(b) else np.nan)
        # one-sample test against chance per cohort (is the cohort decodable at all?)
        row["pd_vs_chance_t_p"] = stats.ttest_1samp(a["balanced_accuracy"], CHANCE[dec]).pvalue
        row["pd_vs_chance_wilcoxon_p"] = stats.wilcoxon(a["balanced_accuracy"] - CHANCE[dec]).pvalue if len(a) > 1 else np.nan
        if len(b) > 1:
            row["hc_vs_chance_t_p"] = stats.ttest_1samp(b["balanced_accuracy"], CHANCE[dec]).pvalue
            row["hc_vs_chance_wilcoxon_p"] = stats.wilcoxon(b["balanced_accuracy"] - CHANCE[dec]).pvalue
            t, pt = stats.ttest_ind(a["balanced_accuracy"], b["balanced_accuracy"], equal_var=False)
            row.update(welch_t=t, welch_p=pt, mwu_p=stats.mannwhitneyu(a["balanced_accuracy"], b["balanced_accuracy"]).pvalue,
                       hedges_g=hedges_g(a["balanced_accuracy"].to_numpy(), b["balanced_accuracy"].to_numpy()),
                       fisher_above_chance_p=stats.fisher_exact([[row["pd_above_chance"], len(a) - row["pd_above_chance"]],
                                                                 [row["hc_above_chance"], len(b) - row["hc_above_chance"]]])[1])
        else:
            row.update(hc_vs_chance_t_p=np.nan, hc_vs_chance_wilcoxon_p=np.nan, welch_t=np.nan, welch_p=np.nan, mwu_p=np.nan, hedges_g=np.nan, fisher_above_chance_p=np.nan)
        group_rows.append(row)
    group = pd.DataFrame(group_rows)
    g_path = out_dir / f"gait_phase_decoding_{DATE}.csv"
    group.to_csv(g_path, index=False); write_sidecar(g_path, meta)

    # ---- learning curve (primary decoder)
    lc_rows = []
    for subj in subjects:
        key = (subj, "eeg", "avggait")
        X, phase, cycle, names, hemi = build_samples(tables[key], FEATURE_SETS["eeg_both_motor_frontal"][1])
        lc_rows += learning_curve_cell(subj, cohort_of[subj], X, phase, cycle)
    lc = pd.DataFrame(lc_rows)
    lc_path = out_dir / f"gait_phase_decoding_learning_curve_{DATE}.csv"
    lc.to_csv(lc_path, index=False); write_sidecar(lc_path, meta)

    # ---- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    col = {PD: "C3", HC: "C0"}
    fs_order = [f for f in FEATURE_SETS if f != "stn"] + ["stn"]
    dec_order = ["4class", "LLSvsRLS", "DSvsSwing"]
    for norm in NORMS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        for ax, dec in zip(axes, dec_order):
            for k, fs in enumerate(fs_order):
                for cohort, dx in ((PD, -0.17), (HC, 0.17)):
                    v = per_subject.query("feature_set == @fs and norm == @norm and decoder == @dec and clf == 'logreg' and cohort == @cohort")["balanced_accuracy"].to_numpy()
                    if len(v) == 0:
                        continue
                    ax.scatter(np.full(len(v), k + dx) + rng.uniform(-0.05, 0.05, len(v)), v, color=col[cohort], alpha=0.6, s=20)
                    ax.hlines(v.mean(), k + dx - 0.13, k + dx + 0.13, color=col[cohort], lw=2.5)
            ax.axhline(CHANCE[dec], color="k", ls=":", lw=1)
            ax.set_xticks(range(len(fs_order))); ax.set_xticklabels([f.replace("eeg_", "").replace("_", "\n") for f in fs_order], fontsize=8)
            ax.set_title(f"{dec} ({norm}, logistic regression)"); ax.set_ylabel("balanced accuracy (5-fold, cycle-grouped)")
        axes[0].scatter([], [], color=col[PD], label="PD"); axes[0].scatter([], [], color=col[HC], label="HC"); axes[0].legend(loc="upper left")
        fig.tight_layout(); fig.savefig(fig_dir / f"gait_phase_decoding_accuracy{'' if norm == 'avggait' else '_basenorm'}.png", dpi=130); plt.close(fig)

    # confusion matrices (row-normalised), primary decoder, plus STN
    panels = [(PD, "eeg_both_motor_frontal"), (HC, "eeg_both_motor_frontal"), (PD, "stn")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (cohort, fs) in zip(axes, panels):
        sub = cm_sum.query("cohort == @cohort and feature_set == @fs and norm == 'avggait' and decoder == '4class' and clf == 'logreg'")
        M = sub.pivot(index="true", columns="predicted", values="count").reindex(index=PHASES, columns=PHASES).to_numpy(float)
        Mn = M / M.sum(1, keepdims=True)
        im = ax.imshow(Mn, vmin=0, vmax=max(0.5, Mn.max()), cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{Mn[i, j]:.2f}\n({int(M[i, j])})", ha="center", va="center", fontsize=8, color="w" if Mn[i, j] > 0.3 else "k")
        ax.set_xticks(range(4)); ax.set_xticklabels(PHASES); ax.set_yticks(range(4)); ax.set_yticklabels(PHASES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{'PD' if cohort == PD else 'HC'}: {fs} (avggait)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Summed out-of-fold confusion matrices, 4-class, logistic regression (row-normalised; counts)")
    fig.tight_layout(); fig.savefig(fig_dir / "gait_phase_decoding_confusion.png", dpi=130); plt.close(fig)

    # feature-importance heatmaps: band x area per cohort, 4class and LLSvsRLS, avggait + basenorm
    imp_mean = importance.groupby(["cohort", "feature_set", "norm", "decoder", "clf", "area", "hemisphere", "band"], as_index=False)["mean_abs_coef"].mean()
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    specs = [(norm, dec) for norm in NORMS for dec in ("4class", "LLSvsRLS")]
    for c_, (norm, dec) in enumerate(specs):
        for r_, cohort in enumerate((PD, HC)):
            ax = axes[r_, c_]
            sub = imp_mean.query("cohort == @cohort and feature_set == 'eeg_LR_motor_frontal' and norm == @norm and decoder == @dec and clf == 'logreg'")
            sub = sub.assign(col=sub["area"] + "_" + sub["hemisphere"])
            M = sub.pivot(index="band", columns="col", values="mean_abs_coef").reindex(index=BANDS, columns=["motor_Left", "motor_Right", "frontal_Left", "frontal_Right"])
            im = ax.imshow(M.to_numpy(float), cmap="viridis", aspect="auto")
            ax.set_xticks(range(4)); ax.set_xticklabels(["motor L", "motor R", "frontal L", "frontal R"], fontsize=8)
            ax.set_yticks(range(5)); ax.set_yticklabels([b.replace("_", " ") for b in BANDS], fontsize=8)
            ax.set_title(f"{'PD' if cohort == PD else 'HC'}: {dec}, {norm}", fontsize=9); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Mean |standardised coefficient| per feature (logistic regression, Left + Right, motor + frontal), averaged per cohort")
    fig.tight_layout(); fig.savefig(fig_dir / "gait_phase_decoding_feature_importance.png", dpi=130); plt.close(fig)

    # learning curve
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    order = ["20", "40", "60", "all"]
    lc["n_train_cycles"] = lc["n_train_cycles"].astype(str)
    for cohort in (PD, HC):
        sub = lc[lc["cohort"] == cohort]
        for subj, s in sub.groupby("subject"):
            s = s.set_index("n_train_cycles").reindex(order)
            ax.plot(range(4), s["balanced_accuracy"], color=col[cohort], alpha=0.25, lw=1)
        m = sub.groupby("n_train_cycles")["balanced_accuracy"].mean().reindex(order)
        ax.plot(range(4), m, color=col[cohort], lw=2.5, marker="o", label="PD" if cohort == PD else "HC")
    ax.axhline(0.25, color="k", ls=":", lw=1); ax.set_xticks(range(4)); ax.set_xticklabels(order)
    ax.set_xlabel("training cycles (per outer fold)"); ax.set_ylabel("balanced accuracy, 4-class")
    ax.set_title("Learning curve, motor + frontal (Both), avggait, logistic regression"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "gait_phase_decoding_learning_curve.png", dpi=130); plt.close(fig)

    # ---- report
    def fmt_p(p):
        return "n/a" if pd.isna(p) else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")

    lines = [f"# Gait-phase decoding from single-cycle band power ({DATE})", "",
             "Script: `code/analysis/revision/gait_phase_decoding.py`. Samples = (cycle, phase); features = five bands per area/hemisphere "
             "(dB gERD/ERS). Classifiers: multinomial logistic regression (standardised, C = 1) and LDA. StratifiedGroupKFold (5 folds, "
             f"group = cycle). Balanced accuracy on pooled out-of-fold predictions; permutation baseline = {args.n_perm} label permutations within "
             "cycle, same folds. PD n = 10, HC n = 10 (EEG); STN n = 10 (PD, one hemisphere each: 4 left, 6 right).", "",
             "## Group table (balanced accuracy, mean ± SD across subjects; above = subjects with permutation p < 0.05)", ""]
    for clf_name in ("logreg", "lda"):
        for norm in NORMS:
            lines += [f"### {clf_name}, {norm}", "",
                      "| decoder | feature set | chance | PD mean ± SD | PD above | PD vs chance p (t / Wilcoxon) | HC mean ± SD | HC above | HC vs chance p | Welch p | MWU p | Hedges g | Fisher p (above) |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for dec in dec_order + ["contra_ipsi"]:
                for fs in fs_order:
                    r = group.query("feature_set == @fs and norm == @norm and decoder == @dec and clf == @clf_name")
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    hc = f"{r.hc_mean:.3f} ± {r.hc_sd:.3f}" if not pd.isna(r.hc_mean) else "—"
                    hc_ab = f"{int(r.hc_above_chance)}/{int(r.n_hc)}" if not pd.isna(r.hc_above_chance) else "—"
                    lines.append(f"| {dec} | {fs} | {r.chance:.2f} | {r.pd_mean:.3f} ± {r.pd_sd:.3f} | {int(r.pd_above_chance)}/{int(r.n_pd)} | "
                                 f"{fmt_p(r.pd_vs_chance_t_p)} / {fmt_p(r.pd_vs_chance_wilcoxon_p)} | {hc} | {hc_ab} | "
                                 f"{fmt_p(r.hc_vs_chance_t_p)} / {fmt_p(r.hc_vs_chance_wilcoxon_p)} | {fmt_p(r.welch_p)} | {fmt_p(r.mwu_p)} | "
                                 f"{'' if pd.isna(r.hedges_g) else f'{r.hedges_g:+.2f}'} | {fmt_p(r.fisher_above_chance_p)} |")
            lines.append("")

    # per-subject primary table
    lines += ["## Per-subject balanced accuracy, primary decoder (eeg_both_motor_frontal, avggait, logistic regression)", "",
              "| subject | cohort | cycles | 4-class | perm p | LLS vs RLS | perm p | DS vs swing | perm p |", "|---|---|---|---|---|---|---|---|---|"]
    for subj in subjects:
        q = per_subject.query("subject == @subj and feature_set == 'eeg_both_motor_frontal' and norm == 'avggait' and clf == 'logreg'").set_index("decoder")
        lines.append(f"| {subj} | {'PD' if cohort_of[subj] == PD else 'HC'} | {int(q.loc['4class', 'n_cycles'])} | "
                     f"{q.loc['4class', 'balanced_accuracy']:.3f} | {fmt_p(q.loc['4class', 'perm_p'])} | "
                     f"{q.loc['LLSvsRLS', 'balanced_accuracy']:.3f} | {fmt_p(q.loc['LLSvsRLS', 'perm_p'])} | "
                     f"{q.loc['DSvsSwing', 'balanced_accuracy']:.3f} | {fmt_p(q.loc['DSvsSwing', 'perm_p'])} |")
    lines.append("")

    # STN per subject
    lines += ["## STN per subject (avggait, logistic regression)", "",
              "| subject | hemisphere | cycles | 4-class | perm p | LLS vs RLS | contra vs ipsi | perm p |", "|---|---|---|---|---|---|---|---|"]
    for subj in subjects:
        q = per_subject.query("subject == @subj and feature_set == 'stn' and norm == 'avggait' and clf == 'logreg'")
        if q.empty:
            continue
        q = q.set_index("decoder")
        lines.append(f"| {subj} | {q.loc['4class', 'stn_hemisphere']} | {int(q.loc['4class', 'n_cycles'])} | {q.loc['4class', 'balanced_accuracy']:.3f} | "
                     f"{fmt_p(q.loc['4class', 'perm_p'])} | {q.loc['LLSvsRLS', 'balanced_accuracy']:.3f} | {q.loc['contra_ipsi', 'balanced_accuracy']:.3f} | {fmt_p(q.loc['contra_ipsi', 'perm_p'])} |")
    lines += ["", "Note on ipsi/contra realignment: the contralateral-vs-ipsilateral decoder is the LLS-vs-RLS decoder with the two labels renamed per "
              "subject (left STN: RLS = contralateral swing; right STN: LLS = contralateral swing). A binary classifier is invariant to a label "
              "swap, so per-subject accuracy is identical by construction; the realignment matters only when coefficients or confusion matrices "
              "are pooled across subjects with different recorded hemispheres.", ""]

    # confusion summary
    lines += ["## Confusion matrices (4-class, avggait, logistic regression; summed over subjects, row-normalised)", ""]
    for cohort, fs in panels:
        sub = cm_sum.query("cohort == @cohort and feature_set == @fs and norm == 'avggait' and decoder == '4class' and clf == 'logreg'")
        M = sub.pivot(index="true", columns="predicted", values="count").reindex(index=PHASES, columns=PHASES).to_numpy(float)
        Mn = M / M.sum(1, keepdims=True)
        lines += [f"{'PD' if cohort == PD else 'HC'}, {fs}:", "", "| true \\ pred | " + " | ".join(PHASES) + " |", "|---|---|---|---|---|"]
        for i, p in enumerate(PHASES):
            lines.append(f"| {p} | " + " | ".join(f"{Mn[i, j]:.2f}" for j in range(4)) + " |")
        lines.append("")

    # feature importance summary (Both, motor+frontal, 4class, avggait)
    lines += ["## Feature importance (mean |standardised coefficient|, 4-class, avggait, logistic regression, motor + frontal Both)", "",
              "| band | PD motor | PD frontal | HC motor | HC frontal |", "|---|---|---|---|---|"]
    fi = imp_mean.query("feature_set == 'eeg_both_motor_frontal' and norm == 'avggait' and decoder == '4class' and clf == 'logreg'")
    for band in BANDS:
        vals = [fi[(fi["cohort"] == c) & (fi["area"] == a) & (fi["band"] == band)]["mean_abs_coef"].iloc[0] for c in (PD, HC) for a in ("motor", "frontal")]
        lines.append(f"| {band.replace('_', ' ')} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    fi_stn = imp_mean.query("feature_set == 'stn' and norm == 'avggait' and decoder == '4class' and clf == 'logreg'")
    lines += ["", "STN (PD), 4-class avggait: " + ", ".join(f"{b.replace('_', ' ')} {fi_stn[fi_stn['band'] == b]['mean_abs_coef'].iloc[0]:.3f}" for b in BANDS), ""]

    # learning curve summary
    lines += ["## Learning curve (4-class, eeg_both_motor_frontal, avggait, logistic regression; mean balanced accuracy across subjects)", "",
              "| training cycles | PD | HC |", "|---|---|---|"]
    for n in order:
        a = lc.query("cohort == @PD and n_train_cycles == @n")["balanced_accuracy"]; b = lc.query("cohort == @HC and n_train_cycles == @n")["balanced_accuracy"]
        lines.append(f"| {n} | {a.mean():.3f} (n = {a.notna().sum()}) | {b.mean():.3f} (n = {b.notna().sum()}) |")
    lines += ["", "Subjects with fewer than 60 training cycles in a fold contribute only to the sizes they can support "
              "(P4-2003 has 36 cycles, so at most ~29 training cycles per fold).", ""]

    # how to present
    prim = group.query("feature_set == 'eeg_both_motor_frontal' and norm == 'avggait' and decoder == '4class' and clf == 'logreg'").iloc[0]
    lr = group.query("feature_set == 'eeg_LR_motor' and norm == 'avggait' and decoder == 'LLSvsRLS' and clf == 'logreg'").iloc[0]
    stn4 = group.query("feature_set == 'stn' and norm == 'avggait' and decoder == '4class' and clf == 'logreg'").iloc[0]
    lines += ["## How to present", "",
              f"Single-cycle gait phase is decodable from cortical band power in both cohorts: 4-class balanced accuracy {prim.pd_mean:.2f} ± {prim.pd_sd:.2f} in PD "
              f"and {prim.hc_mean:.2f} ± {prim.hc_sd:.2f} in HC (chance 0.25; {int(prim.pd_above_chance)}/10 and {int(prim.hc_above_chance)}/10 subjects above their "
              f"permutation baseline; Welch p = {fmt_p(prim.welch_p)}, g = {prim.hedges_g:+.2f}). "
              f"Left- versus right-leg swing from lateralised motor features reaches {lr.pd_mean:.2f} (PD) vs {lr.hc_mean:.2f} (HC), Welch p = {fmt_p(lr.welch_p)}. "
              f"STN activity carries {'weaker' if stn4.pd_mean < prim.pd_mean else 'comparable'} phase information ({stn4.pd_mean:.2f} ± {stn4.pd_sd:.2f}, {int(stn4.pd_above_chance)}/10 above chance). "
              "Suggested framing: one paragraph in Results (\"individual-level decodability\") plus one supplementary figure (accuracy points by cohort, "
              "confusion matrices, feature-importance heatmap) and the group table as a supplementary table; present it as a complement to the LMM "
              "conclusion that the gait-locked pattern is present in both groups rather than as a new main finding. State that avggait is the "
              "primary normalisation (phase information only), that folds are grouped by cycle, and that the learning curve shows the estimate is "
              "stable at the available cycle counts. TODO(verify): wording of the physiological interpretation once the feature-importance table "
              "has been read against the cluster-permutation results.", ""]
    rep_path = out_dir / f"gait_phase_decoding_report_{DATE}.md"
    rep_path.write_text("\n".join(lines)); write_sidecar(rep_path, meta)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
