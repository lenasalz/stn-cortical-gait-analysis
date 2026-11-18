"""Post-hoc QA of the gait-cycle warping stage for one subject.

Purpose:  Quantify data quality of a subject's gait-warped TFR: cycle
          segmentation regularity, warp distortion, autoreject impact,
          warped-TFR sanity vs cohort, and per-cycle gERD outliers.
Inputs:   data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif and/or
            .../<subj>_<block>-dbs-proc-raw.fif       (gait annotations)
          data/derivatives/autoreject/<subj>_<block>-mask.npz
          data/derivatives/tfr/<subj>_<block>-{eeg,dbs}-tfr-gait.h5
            (+ same files for cohort peers, + .meta.json sidecars)
          data/derivatives/gerd/master_{eeg,dbs}_<norm>.csv
Outputs:  outputs/stats/gait_warp_qa_<subj>_<block>_<YYYY-MM-DD>.txt (+ .meta.json)
          outputs/figures/qa/gait_warp_qa_<subj>_<block>.png (+ .meta.json)
Author:   Lena
Date:     2026-08-13
Seed:     42 (config repro.seed, applied via _common.setup_run)

Description:
  Re-derives the gait cycles with the exact pipeline chain used by
  compute_tfr_gait.py (filter_gait_events -> replace_events_in_mne ->
  segment_gait_cycles) and cross-checks the count against the warped
  TFR's sidecar. Reports:
    1. Cycle timing: duration stats, inter-cycle gaps (turns/breaks),
       per-cycle event-onset percentages vs the canonical 0/10/50/60/100
       warp targets, and per-segment warp stretch factors.
    2. Autoreject: kept/rejected counts, temporal clustering of
       rejections, kept-vs-rejected duration comparison.
    3. Warped TFR sanity: non-finite values, dB range, gait-phase
       modulation depth — each z-scored against same-cohort peers.
    4. Per-cycle gERD tables: within-subject outlier cycles (|z| > 3)
       and the subject's per-cycle SD vs cohort peers, per band.

Usage:
  python code/analysis/check_gait_warp_quality.py --subject P4-2005 \
      [--block pre8walk] [--modality both] [--norm basenorm]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (
    PROJECT_ROOT,
    cohort_of,
    load_config,
    output_path,
    patch_legacy_annotations,
    resolve_paths,
    setup_run,
    validate_block,
    validate_subject,
    write_sidecar,
)

# Okabe-Ito accents (CVD-safe pair, validated); grey is recessive context.
COLOR_SUBJECT = "#0072B2"
COLOR_REJECT = "#D55E00"
COLOR_CONTEXT = "#767676"

# Canonical warp targets from variables.event_percentages (0/10/50/60/100).
CANONICAL_PCT = {"TOleftlocs": 10.0, "HSleftlocs": 50.0, "TOrightlocs": 60.0}
SEGMENTS = [
    ("HSrightlocs", "TOleftlocs", 10.0),
    ("TOleftlocs", "HSleftlocs", 40.0),
    ("HSleftlocs", "TOrightlocs", 10.0),
    ("TOrightlocs", "HSrightlocs_end", 40.0),
]
BAND_COLS = [
    "ERD_ERS_Theta",
    "ERD_ERS_Alpha",
    "ERD_ERS_Low_Beta",
    "ERD_ERS_High_Beta",
    "ERD_ERS_Gamma",
]
BAND_LABELS = ["Theta", "Alpha", "Low Beta", "High Beta", "Gamma"]
EVENT_LABELS = {
    "TOleftlocs": "L toe-off (target 10%)",
    "HSleftlocs": "L heel-strike (target 50%)",
    "TOrightlocs": "R toe-off (target 60%)",
}


def derive_cycles(raw_path: Path):
    """Run the exact gait-cycle chain compute_tfr_gait.py uses."""
    import mne
    from epoching import (
        filter_gait_events,
        replace_events_in_mne,
        segment_gait_cycles,
    )
    from variables import gait_events_correct_sequence as gait_seq

    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="WARNING")
    patch_legacy_annotations(raw)
    fe = filter_gait_events(raw, gait_seq)
    raw_filt = replace_events_in_mne(raw, fe, gait_seq)
    cycles = segment_gait_cycles(raw_filt, gait_seq)
    del raw, raw_filt
    return cycles, len(fe)


def cycle_timing(cycles: list[dict]) -> dict:
    """Timing metrics for complete cycles (those carrying HSrightlocs_end)."""
    complete = [c for c in cycles if "HSrightlocs_end" in c]
    start = np.array([c["HSrightlocs"] for c in complete])
    end = np.array([c["HSrightlocs_end"] for c in complete])
    dur = end - start
    pct = {
        ev: (np.array([c[ev] for c in complete]) - start) / dur * 100.0
        for ev in CANONICAL_PCT
    }
    stretch = {}
    for ev_a, ev_b, target_span in SEGMENTS:
        t_a = np.array([c[ev_a] for c in complete])
        t_b = np.array([c[ev_b] for c in complete])
        actual_span = (t_b - t_a) / dur * 100.0
        with np.errstate(divide="ignore"):
            stretch[f"{ev_a}->{ev_b}"] = target_span / actual_span
    gaps = start[1:] - end[:-1]
    return {
        "n_total": len(cycles),
        "n_complete": len(complete),
        "start": start,
        "duration": dur,
        "event_pct": pct,
        "stretch": stretch,
        "gaps": gaps,
    }


def tfr_summary(h5_path: Path) -> dict:
    """Scalar sanity metrics for one gait-warped AverageTFR."""
    import mne

    tfr = mne.time_frequency.read_tfrs(h5_path)
    tfr = tfr[0] if isinstance(tfr, list) else tfr
    d = np.asarray(tfr.data)
    finite = np.isfinite(d)
    profile = np.nanmean(np.where(finite, d, np.nan), axis=(0, 1))
    return {
        "n_nonfinite": int((~finite).sum()),
        "mean_db": float(np.nanmean(d)),
        "sd_db": float(np.nanstd(d)),
        "absmax_db": float(np.nanmax(np.abs(d))),
        "mod_depth_db": float(np.nanmax(profile) - np.nanmin(profile)),
        "n_channels": d.shape[0],
    }


def zscore_vs_peers(value: float, peer_values: list[float]) -> float:
    peers = np.asarray(peer_values, dtype=float)
    if len(peers) < 2 or np.std(peers, ddof=1) == 0:
        return np.nan
    return (value - peers.mean()) / peers.std(ddof=1)


def run_lengths(bool_arr: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs."""
    runs, count = [], 0
    for v in bool_arr:
        if v:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", default="pre8walk")
    parser.add_argument("--modality", choices=["eeg", "dbs", "both"],
                        default="both")
    parser.add_argument("--norm", choices=["basenorm", "avggait"],
                        default="basenorm",
                        help="Which gERD master to check (default basenorm).")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)
    peers = [s for s in config["cohorts"][cohort] if s != subject]
    meta = setup_run("check_gait_warp_quality", config)

    modalities = ["eeg", "dbs"] if args.modality == "both" else [args.modality]
    raw_suffix = {"eeg": "-eeg-raw.fif", "dbs": "-dbs-proc-raw.fif"}

    stats_dir = PROJECT_ROOT / "outputs" / "stats"
    fig_dir = PROJECT_ROOT / "outputs" / "figures" / "qa"
    stats_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    txt_path = stats_dir / (
        f"gait_warp_qa_{subject}_{block}_{date.today().isoformat()}.txt"
    )
    fig_path = fig_dir / f"gait_warp_qa_{subject}_{block}.png"

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    def sec(title: str) -> None:
        emit()
        emit("=" * 72)
        emit(title)
        emit("=" * 72)

    emit(f"Gait-warp QA — {subject} ({cohort}), block={block}, "
         f"norm={args.norm}, seed={meta['seed']}")

    # ------------------------------------------------------------------ #
    # 1) Cycle segmentation + timing (per modality, exact pipeline chain)
    # ------------------------------------------------------------------ #
    timing_by_mod: dict[str, dict] = {}
    sidecar_by_mod: dict[str, dict] = {}
    for mod in modalities:
        raw_in = output_path(paths.raw_clean_dir, subject, block,
                             raw_suffix[mod])
        if not raw_in.exists():
            emit(f"[{mod}] raw missing, skipping modality: {raw_in.name}")
            continue
        sec(f"[{mod}] Cycle segmentation & timing")
        cycles, n_filtered_events = derive_cycles(raw_in)
        t = cycle_timing(cycles)
        timing_by_mod[mod] = t

        gait_h5 = output_path(paths.tfr_dir, subject, block,
                              f"-{mod}-tfr-gait.h5")
        side = Path(str(gait_h5) + ".meta.json")
        if side.exists():
            sc = json.loads(side.read_text())
            sidecar_by_mod[mod] = sc
            match = "PASS" if sc["n_cycles_segmented"] == t["n_total"] else "FAIL"
            emit(f"cross-check vs pipeline sidecar: segmented "
                 f"{t['n_total']} (re-derived) vs "
                 f"{sc['n_cycles_segmented']} (sidecar) -> {match}")
        emit(f"filtered gait events: {n_filtered_events}  |  cycles: "
             f"{t['n_total']} total, {t['n_complete']} complete "
             f"(with HSrightlocs_end)")
        d = t["duration"]
        emit(f"cycle duration: {d.mean():.3f} ± {d.std():.3f} s  "
             f"(CV {100 * d.std() / d.mean():.1f}%), "
             f"range [{d.min():.3f}, {d.max():.3f}] s")
        gaps = t["gaps"]
        big = gaps[gaps > 2.0]
        emit(f"inter-cycle gaps > 2 s (turns/breaks): {len(big)}"
             + (f", median {np.median(big):.1f} s, max {big.max():.1f} s"
                if len(big) else ""))
        for ev, target in CANONICAL_PCT.items():
            p = t["event_pct"][ev]
            emit(f"event {ev:<12} at {p.mean():5.1f} ± {p.std():4.1f}% "
                 f"of cycle (warp target {target:.0f}%, "
                 f"mean |dev| {np.abs(p - target).mean():.1f} pts)")
        for name, ratio in t["stretch"].items():
            emit(f"warp stretch {name:<28} x{np.median(ratio):.2f} median "
                 f"[{ratio.min():.2f}, {ratio.max():.2f}]")
        n_extreme = int(np.sum([(np.abs(np.log2(r)) > 1).sum()
                                for r in t["stretch"].values()]))
        emit(f"segments warped by more than 2x (|log2 stretch| > 1): "
             f"{n_extreme} of {4 * t['n_complete']}")

    if len(timing_by_mod) == 2:
        a, b = (timing_by_mod[m] for m in ("eeg", "dbs"))
        if a["n_complete"] == b["n_complete"]:
            onset_diff = np.abs(a["start"] - b["start"]).max()
            emit(f"\neeg/dbs cycle onsets agree to {onset_diff * 1000:.1f} ms "
                 f"(max abs) -> {'PASS' if onset_diff < 0.01 else 'CHECK'}")
        else:
            emit(f"\neeg/dbs cycle count differs: {a['n_complete']} vs "
                 f"{b['n_complete']} -> CHECK")

    if not timing_by_mod:
        print("No raw inputs found for any requested modality.",
              file=sys.stderr)
        return 2

    ref_mod = "eeg" if "eeg" in timing_by_mod else "dbs"
    t_ref = timing_by_mod[ref_mod]

    # ------------------------------------------------------------------ #
    # 2) Autoreject mask
    # ------------------------------------------------------------------ #
    sec("Autoreject mask")
    mask_npz = paths.derivatives_root / "autoreject" / f"{subject}_{block}-mask.npz"
    keep = None
    if mask_npz.exists():
        keep = np.load(mask_npz)["keep_mask"].astype(bool)
        emit(f"mask: {keep.sum()}/{len(keep)} cycles kept "
             f"({100 * (1 - keep.mean()):.1f}% rejected)")
        if len(keep) != t_ref["n_complete"]:
            emit(f"LENGTH MISMATCH: mask={len(keep)} vs complete cycles="
                 f"{t_ref['n_complete']} -> FAIL")
        rej_runs = run_lengths(~keep)
        if rej_runs:
            emit(f"rejection runs: {len(rej_runs)} "
                 f"(longest {max(rej_runs)} consecutive cycles)")
            rej_idx = np.where(~keep)[0]
            emit(f"rejected cycle indices: {rej_idx.tolist()}")
            if len(keep) == t_ref["n_complete"]:
                d = t_ref["duration"]
                emit(f"duration kept vs rejected (median): "
                     f"{np.median(d[keep]):.3f} vs "
                     f"{np.median(d[~keep]):.3f} s")
    else:
        emit(f"no per-subject mask at {mask_npz.name}")

    # ------------------------------------------------------------------ #
    # 3) Warped-TFR sanity vs cohort peers
    # ------------------------------------------------------------------ #
    tfr_z: dict[str, dict] = {}
    for mod in modalities:
        gait_h5 = output_path(paths.tfr_dir, subject, block,
                              f"-{mod}-tfr-gait.h5")
        if not gait_h5.exists():
            continue
        sec(f"[{mod}] Warped-TFR sanity vs cohort ({cohort})")
        own = tfr_summary(gait_h5)
        peer_metrics: dict[str, list[float]] = {}
        n_peers = 0
        for p in peers:
            ph5 = output_path(paths.tfr_dir, p, block, f"-{mod}-tfr-gait.h5")
            if not ph5.exists():
                continue
            n_peers += 1
            for k, v in tfr_summary(ph5).items():
                peer_metrics.setdefault(k, []).append(v)
        emit(f"non-finite values: {own['n_nonfinite']} "
             f"({'PASS' if own['n_nonfinite'] == 0 else 'FAIL'})  |  "
             f"channels: {own['n_channels']}")
        tfr_z[mod] = {}
        for k in ("mean_db", "sd_db", "absmax_db", "mod_depth_db"):
            z = zscore_vs_peers(own[k], peer_metrics.get(k, []))
            tfr_z[mod][k] = z
            flag = "" if np.isnan(z) or abs(z) < 2 else "  <-- |z| >= 2"
            emit(f"{k:<14} {own[k]:8.2f}   cohort z = {z:+.2f} "
                 f"(n_peers={n_peers}){flag}")

    # ------------------------------------------------------------------ #
    # 4) Per-cycle gERD outliers
    # ------------------------------------------------------------------ #
    gerd_sd: dict[str, dict] = {}
    for mod in modalities:
        master = paths.derivatives_root / "gerd" / f"master_{mod}_{args.norm}.csv"
        if not master.exists():
            continue
        sec(f"[{mod}] Per-cycle gERD ({args.norm} master)")
        m = pd.read_csv(master)
        m = m[m["Block"] == block]
        own = m[m["SubjectID"] == subject]
        if own.empty:
            emit(f"{subject} not present in {master.name}")
            continue
        epochs = np.sort(own["Epoch"].unique())
        emit(f"cycles in gERD table: {len(epochs)} "
             f"(epoch idx {epochs.min()}..{epochs.max()})")
        if keep is not None:
            kept_idx = set(np.where(keep)[0].tolist())
            agree = set(epochs.tolist()) == kept_idx
            emit(f"gERD epoch indices == autoreject kept indices -> "
                 f"{'PASS' if agree else 'FAIL'}")

        area = "STN" if mod == "dbs" else "motor"
        own_area = own[own["Area"] == area]
        outlier_cells = 0
        total_cells = 0
        outlier_epochs: dict[int, int] = {}
        for _, grp in own.groupby(["Area", "Hemisphere", "GaitPhase"]):
            for col in BAND_COLS:
                x = grp[col].to_numpy(dtype=float)
                if len(x) < 3 or np.std(x) == 0:
                    continue
                z = (x - x.mean()) / x.std()
                total_cells += len(x)
                bad = np.abs(z) > 3
                outlier_cells += int(bad.sum())
                for ep in grp["Epoch"].to_numpy()[bad]:
                    outlier_epochs[int(ep)] = outlier_epochs.get(int(ep), 0) + 1
        emit(f"within-subject outlier cells (|z| > 3): {outlier_cells} of "
             f"{total_cells} ({100 * outlier_cells / max(total_cells, 1):.2f}%)")
        if outlier_epochs:
            worst = sorted(outlier_epochs.items(), key=lambda kv: -kv[1])[:10]
            emit("worst cycles (epoch idx: n outlier cells): "
                 + ", ".join(f"{e}: {n}" for e, n in worst))

        gerd_sd[mod] = {"subject": {}, "peers": {}}
        for col, label in zip(BAND_COLS, BAND_LABELS):
            own_sd = float(own_area[col].std())
            peer_sds = [
                float(m[(m["SubjectID"] == p) & (m["Area"] == area)][col].std())
                for p in peers
                if not m[(m["SubjectID"] == p) & (m["Area"] == area)].empty
            ]
            gerd_sd[mod]["subject"][label] = own_sd
            gerd_sd[mod]["peers"][label] = peer_sds
            z = zscore_vs_peers(own_sd, peer_sds)
            flag = "" if np.isnan(z) or abs(z) < 2 else "  <-- |z| >= 2"
            emit(f"per-cycle SD {label:<10} ({area}) {own_sd:6.2f} dB   "
                 f"cohort z = {z:+.2f}{flag}")

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle(f"Gait-warp QA — {subject}, {block} ({args.norm})",
                 fontsize=13)

    # A: cycle duration over the recording, rejected cycles marked
    ax = axes[0, 0]
    start_rel = t_ref["start"] - t_ref["start"][0]
    dur = t_ref["duration"]
    if keep is not None and len(keep) == len(dur):
        ax.plot(start_rel[keep], dur[keep], "o", ms=4, color=COLOR_SUBJECT,
                label=f"kept (n={int(keep.sum())})")
        ax.plot(start_rel[~keep], dur[~keep], "x", ms=6, mew=2,
                color=COLOR_REJECT, label=f"rejected (n={int((~keep).sum())})")
    else:
        ax.plot(start_rel, dur, "o", ms=4, color=COLOR_SUBJECT,
                label=f"cycles (n={len(dur)})")
    ax.axhline(np.median(dur), ls="--", lw=1, color=COLOR_CONTEXT)
    ax.text(0.99, np.median(dur), " median", va="bottom", ha="right",
            color=COLOR_CONTEXT, fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xlabel("Time since first cycle (s)")
    ax.set_ylabel("Cycle duration (s)")
    ax.set_title(f"A  Cycle durations [{ref_mod}]", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)

    # B: event-onset percentages vs canonical warp targets
    ax = axes[0, 1]
    rng = np.random.default_rng(int(config["repro"]["seed"]))
    for row, (ev, target) in enumerate(CANONICAL_PCT.items()):
        p = t_ref["event_pct"][ev]
        jitter = rng.uniform(-0.18, 0.18, len(p))
        if keep is not None and len(keep) == len(p):
            ax.plot(p[keep], row + jitter[keep], "o", ms=3, alpha=0.6,
                    color=COLOR_SUBJECT)
            ax.plot(p[~keep], row + jitter[~keep], "x", ms=5, mew=1.5,
                    color=COLOR_REJECT)
        else:
            ax.plot(p, row + jitter, "o", ms=3, alpha=0.6, color=COLOR_SUBJECT)
        ax.plot([target, target], [row - 0.3, row + 0.3], ls="--", lw=1.2,
                color=COLOR_CONTEXT)
    ax.set_yticks(range(len(CANONICAL_PCT)))
    ax.set_yticklabels([EVENT_LABELS[e] for e in CANONICAL_PCT], fontsize=8)
    ax.set_xlabel("Event onset (% of gait cycle)")
    ax.set_title("B  Event timing vs warp targets (dashed)", loc="left",
                 fontsize=11)
    ax.grid(alpha=0.25, lw=0.5, axis="x")

    # C: rejection rate across the subject's cohort (from sidecars)
    ax = axes[1, 0]
    rates, names = [], []
    for s in config["cohorts"][cohort]:
        side = Path(str(output_path(paths.tfr_dir, s, block,
                                    f"-{ref_mod}-tfr-gait.h5")) + ".meta.json")
        if side.exists():
            sc = json.loads(side.read_text())
            rates.append(100 * (1 - sc["n_cycles_kept"]
                                / sc["n_cycles_segmented"]))
            names.append(s)
    order = np.argsort(rates)[::-1]
    for i, oi in enumerate(order):
        color = COLOR_SUBJECT if names[oi] == subject else COLOR_CONTEXT
        ax.bar(i, rates[oi], color=color, width=0.7)
        ax.text(i, rates[oi] + 0.3, f"{rates[oi]:.1f}", ha="center",
                fontsize=7, color=color)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[oi] for oi in order], rotation=45, fontsize=7,
                       ha="right")
    ax.set_ylabel("% cycles rejected")
    ax.set_title(f"C  Autoreject rate, {cohort} [{ref_mod}]", loc="left",
                 fontsize=11)
    ax.grid(alpha=0.25, lw=0.5, axis="y")

    # D: per-cycle gERD SD per band, subject vs cohort peers
    ax = axes[1, 1]
    mod_d = "eeg" if "eeg" in gerd_sd else ("dbs" if "dbs" in gerd_sd else None)
    if mod_d:
        area_d = "STN" if mod_d == "dbs" else "motor"
        for i, label in enumerate(BAND_LABELS):
            peer_vals = gerd_sd[mod_d]["peers"].get(label, [])
            jitter = rng.uniform(-0.12, 0.12, len(peer_vals))
            ax.plot(i + jitter, peer_vals, "o", ms=4, alpha=0.6,
                    color=COLOR_CONTEXT)
            ax.plot(i, gerd_sd[mod_d]["subject"][label], "D", ms=8,
                    color=COLOR_SUBJECT)
        ax.plot([], [], "o", color=COLOR_CONTEXT, label="cohort peers")
        ax.plot([], [], "D", color=COLOR_SUBJECT, label=subject)
        ax.set_xticks(range(len(BAND_LABELS)))
        ax.set_xticklabels(BAND_LABELS, fontsize=8)
        ax.set_ylabel("Per-cycle SD of gERD (dB)")
        ax.set_title(f"D  gERD cycle-to-cycle spread, {area_d} [{mod_d}]",
                     loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    else:
        ax.axis("off")

    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    emit(f"\nwrote figure: {fig_path.relative_to(PROJECT_ROOT)}")

    txt_path.write_text("\n".join(lines) + "\n")
    print(f"wrote report: {txt_path.relative_to(PROJECT_ROOT)}")

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "norm": args.norm, "modalities": list(timing_by_mod),
        "n_cycles_complete": int(t_ref["n_complete"]),
        "n_cycles_kept": int(keep.sum()) if keep is not None else None,
        "tfr_cohort_z": tfr_z,
        "outputs": [str(txt_path), str(fig_path)],
    })
    write_sidecar(txt_path, meta)
    write_sidecar(fig_path, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
