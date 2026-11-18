"""Whole-scalp cluster-based permutation tests on per-channel gait-warped TFRs.

Purpose:  Ask whether a PD-vs-HC difference in gait-cycle-locked cortical
          activity exists outside the two a-priori ROIs (motor C1 C3 Cz C2 C4;
          frontal AF3 F1 Fz F2 AF4) or at channel level, i.e. whether the ROI
          averaging used so far could have hidden a cohort difference. The
          test is run on the per-channel gait-warped TFRs with a combined
          channel x frequency x time adjacency, subjects as the exchangeable
          units. A per-channel 2-D (freq x time) test without any channel
          adjacency is added as a robustness check.
Inputs:   data/derivatives/tfr/<subj>_pre8walk-eeg-tfr-gait.h5          (basenorm, dB vs rest)
          data/derivatives/tfr/<subj>_pre8walk-eeg-tfr-gait-avggait.h5  (avggait, per-cycle centred)
          code/analysis/config.yaml (cohorts, freq_bands, repro.seed)
          Montage: mne standard_1020 (the files carry no digitisation).
          Test helper `_run_one` (per-channel 2-D test) is imported unchanged
          from code/analysis/cluster_stats_tfr.py.
Outputs:  data/derivatives/tfr_gavg/revision_wholescalp_stack_<norm>.nc
              (n_subj, n_union_ch, n_freq, 110) decimated per-subject stacks,
              NaN where a channel was dropped for a subject
          data/derivatives/tfr_gavg/revision_wholescalp_<set>_<norm>_<test>.nc
              (F_obs or t_obs, labelled cluster map for p < 0.1, cluster p values)
          outputs/stats/revision_2026-10/cluster_perm_wholescalp_channels_<date>.csv
          outputs/stats/revision_2026-10/cluster_perm_wholescalp_clusters_<date>.csv
          outputs/stats/revision_2026-10/cluster_perm_wholescalp_perchannel_<date>.csv
          outputs/stats/revision_2026-10/cluster_perm_wholescalp_<date>.md
          outputs/figures/revision_2026-10/wholescalp_topo_F_<set>.png
          outputs/figures/revision_2026-10/wholescalp_bestcluster_<set>_<norm>.png
Author:   Lena
Date:     2026-10-20
Seed:     42 (config repro.seed via _common.setup_run; passed to MNE's permutation seed)

Description:
  1. Channel sets. The per-subject gait TFRs carry different channel sets
     (bad channels dropped). "common" = channels present in all 20 subjects
     (primary, no interpolation). "ge16" = channels present in >= 16 subjects
     (secondary); for this set a subject's missing channels are filled by
     spherical-spline interpolation from that subject's own present channels
     (positions from standard_1020), so the result depends on the
     interpolation and is reported as a sensitivity analysis only.
  2. Time axis is reduced from 1100 to 110 bins by averaging consecutive
     10-sample bins (1 % of the warped cycle per bin) before stacking.
  3. Tests (tail = 0, cluster-forming threshold = MNE default, cluster
     statistic = sum of F or t, out_type = "mask", n_jobs = 1):
       <norm>_PDvsHC   : mne.stats.permutation_cluster_test, independent
                         samples PD vs HC, adjacency =
                         combine_adjacency(channel adjacency, n_freq, n_time)
                         with frequencies and time bins adjacent to their
                         neighbours
       avggait_1samp_<cohort> : mne.stats.permutation_cluster_1samp_test,
                         modulation != 0 within each cohort (positive control)
     Channel adjacency: mne.channels.find_ch_adjacency on an info object with
     the standard_1020 montage set (Delaunay triangulation of the analysed
     channels; MNE 1.12.1 has no built-in "easycap64" neighbour file).
  4. Per-channel robustness check: for every channel present in >= 16
     subjects, a 2-D (freq x time) PD-vs-HC cluster test on the subjects that
     have that channel (no interpolation); minimum cluster p per channel,
     Benjamini-Hochberg across the common channels and across all tested
     channels.

Usage:
  python code/analysis/revision/cluster_perm_wholescalp.py
  python code/analysis/revision/cluster_perm_wholescalp.py --n-perm 1000 --sets common
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import PROJECT_ROOT, load_config, resolve_paths, setup_run, write_sidecar  # noqa: E402
from cluster_stats_tfr import _run_one  # noqa: E402

PD, HC = "Patient_NFB", "AgeMatched_NFB"
NORMS = ("basenorm", "avggait")
N_TIME_DECIM = 110
MIN_SUBJ_SECONDARY = 16
ROI_CHANNELS = {"motor": ["C1", "C3", "Cz", "C2", "C4"], "frontal": ["AF3", "F1", "Fz", "F2", "AF4"]}


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def _read_tfr(path: Path):
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tfr = mne.time_frequency.read_tfrs(path, verbose="ERROR")
    return tfr[0] if isinstance(tfr, list) else tfr


def _decimate_time(data: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean over consecutive bins along the last axis (1100 -> n_bins)."""
    n_t = data.shape[-1]
    if n_t % n_bins:
        raise ValueError(f"{n_t} time samples not divisible into {n_bins} bins")
    step = n_t // n_bins
    return data.reshape(*data.shape[:-1], n_bins, step).mean(axis=-1)


def prepare_stacks(subjects: list[str], cohorts: dict[str, str], tfr_dir: Path,
                   gavg_dir: Path, force: bool) -> dict[str, xr.Dataset]:
    """Load one subject at a time, decimate, and stack on the union of channels."""
    out = {}
    paths = {norm: gavg_dir / f"revision_wholescalp_stack_{norm}.nc" for norm in NORMS}
    if all(p.exists() for p in paths.values()) and not force:
        for norm, p in paths.items():
            out[norm] = xr.load_dataset(p)
        return out

    # pass 1: channel union and axes (cheap: header + names only need one read per subject)
    union: list[str] = []
    per_subject_names: dict[str, list[str]] = {}
    freqs = times = None
    for s in subjects:
        tfr = _read_tfr(tfr_dir / f"{s}_pre8walk-eeg-tfr-gait.h5")
        per_subject_names[s] = list(tfr.ch_names)
        for c in tfr.ch_names:
            if c not in union:
                union.append(c)
        if freqs is None:
            freqs = tfr.freqs.copy()
            times = tfr.times.copy()
        else:
            assert np.allclose(freqs, tfr.freqs) and len(times) == len(tfr.times), s
        del tfr
    t_dec = _decimate_time(times[None, :], N_TIME_DECIM)[0]       # bin centres (s)
    cycle_pct = (t_dec - times[0]) / (times[-1] - times[0]) * 100  # % of the warped cycle

    # pass 2: fill stacks, one subject and one normalisation at a time
    for norm, suffix in (("basenorm", ""), ("avggait", "-avggait")):
        stack = np.full((len(subjects), len(union), len(freqs), N_TIME_DECIM), np.nan, dtype=np.float32)
        for i, s in enumerate(subjects):
            tfr = _read_tfr(tfr_dir / f"{s}_pre8walk-eeg-tfr-gait{suffix}.h5")
            assert list(tfr.ch_names) == per_subject_names[s], (s, norm)
            dec = _decimate_time(tfr.data, N_TIME_DECIM)
            for k, c in enumerate(tfr.ch_names):
                stack[i, union.index(c)] = dec[k]
            del tfr, dec
        ds = xr.Dataset(
            {"tfr": (("subject", "channel", "freq", "time"), stack),
             "present": (("subject", "channel"), ~np.isnan(stack[:, :, 0, 0]))},
            coords={"subject": subjects, "channel": union, "freq": freqs, "time": t_dec,
                    "cycle_pct": ("time", cycle_pct),
                    "cohort": ("subject", [cohorts[s] for s in subjects])},
            attrs={"normalisation": norm, "block": "pre8walk", "time_decim": "mean over 10-sample bins",
                   "n_time_original": int(len(times))},
        )
        ds.to_netcdf(paths[norm])
        out[norm] = ds
    return out


# ----------------------------------------------------------------------------
# Channel adjacency and interpolation
# ----------------------------------------------------------------------------
def make_info(ch_names: list[str]):
    import mne

    info = mne.create_info(ch_names, sfreq=1000.0, ch_types="eeg")
    info.set_montage("standard_1020")
    return info


def channel_adjacency(ch_names: list[str]):
    import mne

    adj, names = mne.channels.find_ch_adjacency(make_info(ch_names), "eeg")
    assert list(names) == list(ch_names)
    return adj


def interpolate_missing(stack: np.ndarray, union: list[str], target: list[str],
                        present: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Spherical-spline interpolation of a subject's missing target channels
    from all of that subject's present channels (standard_1020 positions).

    stack: (n_subj, n_union, n_freq, n_time); present: (n_subj, n_union) bool.
    Returns (n_subj, n_target, n_freq, n_time) and the number of interpolated
    channels per subject.
    """
    import mne
    from mne.channels.interpolation import _make_interpolation_matrix

    info = make_info(union)
    pos = np.array([info["chs"][k]["loc"][:3] for k in range(len(union))])
    _radius, origin, _dev = mne.bem.fit_sphere_to_headshape(info, dig_kinds="eeg", verbose="ERROR")
    pos = pos - origin
    tidx = [union.index(c) for c in target]
    out = np.empty((stack.shape[0], len(target)) + stack.shape[2:], dtype=stack.dtype)
    n_interp = []
    for i in range(stack.shape[0]):
        good = np.where(present[i])[0]
        bad = [k for k in tidx if not present[i, k]]
        n_interp.append(len(bad))
        if bad:
            M = _make_interpolation_matrix(pos[good], pos[bad])          # (n_bad, n_good)
            filled = np.tensordot(M, stack[i, good], axes=(1, 0))         # (n_bad, F, T)
        for j, k in enumerate(tidx):
            if present[i, k]:
                out[i, j] = stack[i, k]
            else:
                out[i, j] = filled[bad.index(k)]
    return out, n_interp


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def _cluster_test(data_pd, data_hc, adjacency, n_perm, seed):
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return mne.stats.permutation_cluster_test(
            [data_pd, data_hc], n_permutations=n_perm, tail=0, adjacency=adjacency,
            n_jobs=1, seed=seed, out_type="mask", verbose="WARNING")


def _cluster_test_1samp(data, adjacency, n_perm, seed):
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return mne.stats.permutation_cluster_1samp_test(
            data, n_permutations=n_perm, tail=0, adjacency=adjacency,
            n_jobs=1, seed=seed, out_type="mask", verbose="WARNING")


def describe_clusters(clusters, cluster_p, stat, diff, ch_names, freqs, cycle_pct,
                      p_report: float) -> tuple[list[dict], np.ndarray]:
    """Rows for clusters with p < p_report, and a labelled map (0 = none) of them,
    numbered by ascending p."""
    order = np.argsort(cluster_p)
    rows, label = [], np.zeros(stat.shape, dtype=np.int16)
    for rank, k in enumerate(order, start=1):
        p = float(cluster_p[k])
        if p >= p_report:
            break
        m = clusters[k]
        ci, fi, ti = np.where(m)
        label[m] = rank
        chans = [ch_names[c] for c in np.unique(ci)]
        in_roi = sorted({r for r, cs in ROI_CHANNELS.items() if set(chans) & set(cs)})
        rows.append(dict(
            cluster=rank, cluster_p=p, n_cells=int(m.sum()), n_channels=len(chans),
            channels=" ".join(chans),
            channels_outside_rois=" ".join(c for c in chans if not any(c in cs for cs in ROI_CHANNELS.values())),
            rois_touched=" ".join(in_roi) or "none",
            freq_lo_hz=float(freqs[fi.min()]), freq_hi_hz=float(freqs[fi.max()]),
            cycle_lo_pct=float(cycle_pct[ti.min()]), cycle_hi_pct=float(cycle_pct[ti.max()]),
            mean_stat=float(stat[m].mean()),
            pd_minus_hc_db=float(diff[m].mean()) if diff is not None else float("nan"),
            direction=("PD<HC" if diff[m].mean() < 0 else "PD>HC") if diff is not None
                      else ("negative" if stat[m].mean() < 0 else "positive"),
        ))
    return rows, label


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def band_masks(freqs: np.ndarray, bands: dict) -> dict[str, np.ndarray]:
    names = list(bands)
    out = {}
    for i, name in enumerate(names):
        lo, hi = bands[name]
        m = (freqs >= lo) & ((freqs <= hi) if i == len(names) - 1 else (freqs < hi))
        out[name] = m
    return out


def plot_topo_grid(results: dict, ch_names: list[str], freqs, bands, out_png: Path,
                   title: str, alpha: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    info = make_info(ch_names)
    bm = band_masks(freqs, bands)
    cols = list(bm) + ["1-45 Hz"]
    tests = [k for k in results if k.endswith("PDvsHC")]
    fig, axes = plt.subplots(len(tests), len(cols), figsize=(2.4 * len(cols), 2.6 * len(tests)), squeeze=False)
    for r, test in enumerate(tests):
        res = results[test]
        F = res["stat"]                                       # (ch, F, T)
        sig05 = np.zeros(len(ch_names), bool); sig10 = np.zeros(len(ch_names), bool)
        for row in res["rows"]:
            idx = [ch_names.index(c) for c in row["channels"].split()]
            (sig05 if row["cluster_p"] < alpha else sig10)[idx] = True
        sig10 &= ~sig05
        vmax = np.nanpercentile(F.mean(axis=2), 99)
        for c, col in enumerate(cols):
            ax = axes[r, c]
            fm = bm[col] if col in bm else np.ones(len(freqs), bool)
            vals = F[:, fm].mean(axis=(1, 2))
            im, _ = mne.viz.plot_topomap(vals, info, axes=ax, show=False, cmap="Reds", vlim=(0, vmax),
                                         mask=sig05, mask_params=dict(marker="o", markerfacecolor="k",
                                                                      markeredgecolor="k", markersize=5),
                                         sensors=True, contours=0)
            if sig10.any():
                pos = mne.channels.layout._find_topomap_coords(info, picks=np.where(sig10)[0])
                ax.scatter(pos[:, 0], pos[:, 1], s=28, facecolors="none", edgecolors="k", linewidths=1.0, zorder=5)
            ax.set_title(f"{test}\n{col}" if c == 0 else col, fontsize=8)
        fig.colorbar(im, ax=axes[r, :].tolist(), shrink=0.7, label="F (mean over band and cycle)")
    fig.suptitle(f"{title}\nfilled = channel in a cluster with p < {alpha}; open = p < 0.1", fontsize=9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_best_cluster(res: dict, ch_names: list[str], freqs, cycle_pct, out_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    rows = res["rows"]
    label = res["label"]
    F = res["stat"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw=dict(width_ratios=[1, 1.8]))
    info = make_info(ch_names)
    if rows:
        best = rows[0]
        m = label == 1
        frac = m.mean(axis=(1, 2))                                   # fraction of cells per channel
        chan_in = frac > 0
        mne.viz.plot_topomap(frac, info, axes=ax0, show=False, cmap="Blues", vlim=(0, max(frac.max(), 1e-6)),
                             mask=chan_in, mask_params=dict(marker="o", markerfacecolor="k",
                                                             markeredgecolor="k", markersize=4), contours=0)
        ax0.set_title(f"cluster 1: p = {best['cluster_p']:.3f}, {best['direction']}\n"
                      f"{best['n_channels']} ch, {best['freq_lo_hz']:.0f}-{best['freq_hi_hz']:.0f} Hz, "
                      f"{best['cycle_lo_pct']:.0f}-{best['cycle_hi_pct']:.0f} % cycle", fontsize=8)
        Fmap = F[chan_in].mean(axis=0)
        any_mask = m[chan_in].any(axis=0)
    else:
        ax0.text(0.5, 0.5, "no cluster with p < 0.1", ha="center", va="center", transform=ax0.transAxes); ax0.axis("off")
        Fmap = F.mean(axis=0); any_mask = np.zeros(Fmap.shape, bool)
    signed = res.get("signed", False)
    vmax = np.nanpercentile(np.abs(Fmap), 99)
    im = ax1.imshow(Fmap, aspect="auto", origin="lower", extent=[cycle_pct[0], cycle_pct[-1], freqs[0], freqs[-1]],
                    cmap="RdBu_r" if signed else "Reds", vmin=-vmax if signed else 0, vmax=vmax)
    if any_mask.any():
        ax1.contour(cycle_pct, freqs, any_mask.astype(float), levels=[0.5], colors="k", linewidths=1.0)
    for x in (10, 50, 60):
        ax1.axvline(x, color="k", lw=0.5, ls="--")
    ax1.set_xlabel("Gait cycle (%)"); ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("statistic averaged over cluster channels (contour = cluster extent)", fontsize=8)
    fig.colorbar(im, ax=ax1, label="t" if signed else "F")
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
def bh(p: np.ndarray) -> np.ndarray:
    from scipy.stats import false_discovery_control

    return false_discovery_control(p, method="bh")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--p-report", type=float, default=0.1)
    parser.add_argument("--sets", default="common,ge16")
    parser.add_argument("--date", default="2026-10-20")
    parser.add_argument("--force-prep", action="store_true")
    parser.add_argument("--skip-perchannel", action="store_true")
    args = parser.parse_args()
    t_start = time.time()

    config = load_config()
    paths = resolve_paths(config)
    meta = setup_run("cluster_perm_wholescalp", config)
    seed = int(config["repro"]["seed"])
    bands = config["freq_bands"]
    out_dir = PROJECT_ROOT / "outputs/stats/revision_2026-10"
    fig_dir = PROJECT_ROOT / "outputs/figures/revision_2026-10"
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    gavg_dir = paths.tfr_gavg_dir

    subjects = list(config["cohorts"][PD]) + list(config["cohorts"][HC])
    cohorts = {s: PD for s in config["cohorts"][PD]} | {s: HC for s in config["cohorts"][HC]}

    # ---- 1. stacks and channel sets ----
    stacks = prepare_stacks(subjects, cohorts, paths.tfr_dir, gavg_dir, args.force_prep)
    ds0 = stacks["basenorm"]
    union = list(ds0.channel.values); freqs = ds0.freq.values; cycle_pct = ds0.cycle_pct.values
    present = ds0.present.values
    n_present = present.sum(axis=0)
    is_pd = ds0.cohort.values == PD
    common = [c for c, n in zip(union, n_present) if n == len(subjects)]
    ge16 = [c for c, n in zip(union, n_present) if n >= MIN_SUBJ_SECONDARY]
    chan_rows = [dict(channel=c, n_subjects=int(n), n_pd=int(present[is_pd, k].sum()),
                      n_hc=int(present[~is_pd, k].sum()), in_common=c in common, in_ge16=c in ge16,
                      roi=next((r for r, cs in ROI_CHANNELS.items() if c in cs), ""))
                 for k, (c, n) in enumerate(zip(union, n_present))]
    chan_csv = out_dir / f"cluster_perm_wholescalp_channels_{args.date}.csv"
    with chan_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chan_rows[0])); w.writeheader(); w.writerows(chan_rows)
    print(f"union {len(union)} channels; common {len(common)}: {' '.join(common)}")
    print(f">= {MIN_SUBJ_SECONDARY} subjects: {len(ge16)}: {' '.join(ge16)}")

    # ---- 2. whole-scalp cluster tests ----
    lines = [f"# Whole-scalp cluster-based permutation tests ({args.date})", "",
             f"n_perm = {args.n_perm}, alpha = {args.alpha}, clusters reported at p < {args.p_report}, seed = {seed}, "
             f"MNE {meta['mne']}, tail = 0, cluster-forming threshold = MNE default, n_jobs = 1.",
             f"Time axis: {ds0.attrs['n_time_original']} -> {N_TIME_DECIM} bins (mean over 10-sample bins).", ""]
    lines += ["## Channel sets", "",
              f"Channels in the union of all per-subject gait TFRs: {len(union)}.",
              f"Per-subject channel counts: {', '.join(f'{s} {int(present[i].sum())}' for i, s in enumerate(subjects))}.",
              "",
              f"Common set (present in all {len(subjects)} subjects, primary, no interpolation): "
              f"**{len(common)}** channels: {' '.join(common)}.",
              f"Secondary set (present in >= {MIN_SUBJ_SECONDARY} subjects, missing channels interpolated per subject): "
              f"**{len(ge16)}** channels: {' '.join(ge16)}.",
              f"Channels in < {MIN_SUBJ_SECONDARY} subjects (never tested with adjacency): "
              f"{' '.join(f'{c} ({int(n)})' for c, n in zip(union, n_present) if n < MIN_SUBJ_SECONDARY) or 'none'}.",
              "ROI channels and their subject counts: "
              + "; ".join(f"{r}: " + ", ".join(f"{c} {int(n_present[union.index(c)]) if c in union else 0}" for c in cs)
                          for r, cs in ROI_CHANNELS.items()) + ".",
              "", "## Adjacency", "",
              "Channel adjacency from `mne.channels.find_ch_adjacency(info, 'eeg')` on an info object with the "
              "`standard_1020` montage (the TFR files carry no digitisation). MNE 1.12.1 has no built-in "
              "`easycap64` neighbour file (`read_ch_adjacency` lists `easycap64ch-avg`, `easycapM1`, ...), so "
              "`find_ch_adjacency` falls back to a Delaunay triangulation of the analysed channel positions. "
              "Full adjacency = `mne.stats.combine_adjacency(channel adjacency, n_freq, n_time)`, i.e. frequency "
              "bins and time bins are adjacent to their immediate neighbours.", ""]

    cluster_rows: list[dict] = []
    timings: dict[str, float] = {}
    sets = {"common": common, "ge16": ge16}
    for set_name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        ch_names = sets[set_name]
        import mne
        ch_adj = channel_adjacency(ch_names)
        A = ch_adj.toarray().astype(bool); np.fill_diagonal(A, False)
        n_edges = int(A.sum() // 2)
        edges = "; ".join(f"{c}: {' '.join(ch_names[j] for j in np.where(A[i])[0])}" for i, c in enumerate(ch_names))
        full_adj = mne.stats.combine_adjacency(ch_adj, len(freqs), N_TIME_DECIM)
        lines += [f"### Set `{set_name}` ({len(ch_names)} channels, {n_edges} channel edges)", "",
                  f"Neighbours: {edges}.", ""]
        results: dict[str, dict] = {}
        for norm in NORMS:
            ds = stacks[norm]
            stack = ds.tfr.values.astype(np.float64)
            if set_name == "common":
                data = stack[:, [union.index(c) for c in ch_names]]
                n_interp = [0] * len(subjects)
            else:
                data, n_interp = interpolate_missing(stack, union, ch_names, present)
            assert not np.isnan(data).any()
            d_pd, d_hc = data[is_pd], data[~is_pd]
            diff = d_pd.mean(axis=0) - d_hc.mean(axis=0)
            if norm == "basenorm":
                lines.append(f"Interpolated channels per subject: "
                             f"{', '.join(f'{s} {n}' for s, n in zip(subjects, n_interp))}.")
                lines.append("")

            tests = [(f"{norm}_PDvsHC", None)]
            if norm == "avggait":
                tests += [(f"{norm}_1samp_PD", "PD"), (f"{norm}_1samp_HC", "HC")]
            for test, cohort in tests:
                t0 = time.time()
                if cohort is None:
                    stat, clusters, cp, _ = _cluster_test(d_pd, d_hc, full_adj, args.n_perm, seed)
                    dd = diff
                else:
                    stat, clusters, cp, _ = _cluster_test_1samp(d_pd if cohort == "PD" else d_hc, full_adj,
                                                                args.n_perm, seed)
                    dd = None
                dt = time.time() - t0
                timings[f"{set_name}_{test}"] = dt
                rows, label = describe_clusters(clusters, cp, stat, dd, ch_names, freqs, cycle_pct, args.p_report)
                for r in rows:
                    cluster_rows.append(dict(channel_set=set_name, test=test, normalisation=norm,
                                             n_pd=int(is_pd.sum()), n_hc=int((~is_pd).sum()), **r))
                results[test] = dict(stat=stat, rows=rows, label=label, signed=cohort is not None)
                n_sig = int(sum(p < args.alpha for p in cp))
                minp = float(min(cp)) if len(cp) else float("nan")
                lines.append(f"- `{test}` ({set_name}): {len(cp)} clusters, {n_sig} with p < {args.alpha}, "
                             f"min cluster p = {minp:.4f} ({dt/60:.1f} min)")
                for r in rows:
                    lines.append(f"    - cluster {r['cluster']}: p = {r['cluster_p']:.4f}, {r['direction']}"
                                 + (f" ({r['pd_minus_hc_db']:+.2f} dB)" if dd is not None else "")
                                 + f", {r['freq_lo_hz']:.0f}-{r['freq_hi_hz']:.0f} Hz, "
                                   f"{r['cycle_lo_pct']:.0f}-{r['cycle_hi_pct']:.0f} % cycle, {r['n_cells']} cells, "
                                   f"{r['n_channels']} channels: {r['channels']} (outside ROIs: "
                                   f"{r['channels_outside_rois'] or 'none'}; ROIs touched: {r['rois_touched']})")
                nc = gavg_dir / f"revision_wholescalp_{set_name}_{norm}_{test}.nc"
                xr.Dataset({"stat": (("channel", "freq", "time"), stat),
                            "cluster_label": (("channel", "freq", "time"), label)},
                           coords={"channel": ch_names, "freq": freqs, "time": ds.time.values,
                                   "cycle_pct": ("time", cycle_pct)},
                           attrs=dict(test=test, channel_set=set_name, n_permutations=args.n_perm,
                                      alpha=args.alpha, seed=seed, statistic="t" if cohort else "F",
                                      cluster_p_sorted=str(sorted(float(p) for p in cp)[:20]),
                                      n_clusters=len(cp))).to_netcdf(nc)
                print(lines[-1 - len(rows)])
            lines.append("")
            # best-cluster figure for the PD vs HC test
            plot_best_cluster(results[f"{norm}_PDvsHC"], ch_names, freqs, cycle_pct,
                              fig_dir / f"wholescalp_bestcluster_{set_name}_{norm}.png",
                              f"PD vs HC, {norm}, channel set {set_name} (n_perm = {args.n_perm})")
        plot_topo_grid(results, ch_names, freqs, bands, fig_dir / f"wholescalp_topo_F_{set_name}.png",
                       f"PD vs HC F statistic, channel set {set_name} ({len(ch_names)} ch)", args.alpha)

    cl_csv = out_dir / f"cluster_perm_wholescalp_clusters_{args.date}.csv"
    fields = ["channel_set", "test", "normalisation", "n_pd", "n_hc", "cluster", "cluster_p", "n_cells", "n_channels",
              "channels", "channels_outside_rois", "rois_touched", "freq_lo_hz", "freq_hi_hz", "cycle_lo_pct",
              "cycle_hi_pct", "mean_stat", "pd_minus_hc_db", "direction"]
    with cl_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(cluster_rows)

    # ---- 3. per-channel 2-D tests (no channel adjacency, no interpolation) ----
    pc_rows: list[dict] = []
    if not args.skip_perchannel:
        lines += ["## Per-channel 2-D (freq x time) PD-vs-HC cluster tests", "",
                  f"One test per channel present in >= {MIN_SUBJ_SECONDARY} subjects, on the subjects that have the "
                  "channel (no interpolation, no channel adjacency); same test as `cluster_perm_revision.py` "
                  "(`cluster_stats_tfr._run_one`). BH correction across the common channels (primary) and across all "
                  "tested channels.", ""]
        t0 = time.time()
        for norm in NORMS:
            stack = stacks[norm].tfr.values.astype(np.float64)
            for c in ge16:
                k = union.index(c)
                keep = present[:, k]
                d_pd = stack[is_pd & keep, k]; d_hc = stack[~is_pd & keep, k]
                _T, _sig, cp = _run_one(d_pd, d_hc, args.n_perm, args.alpha, seed)
                pc_rows.append(dict(normalisation=norm, channel=c, in_common=c in common,
                                    roi=next((r for r, cs in ROI_CHANNELS.items() if c in cs), ""),
                                    n_pd=int(len(d_pd)), n_hc=int(len(d_hc)), n_clusters=len(cp),
                                    min_cluster_p=float(min(cp)) if len(cp) else 1.0))
            sub = [r for r in pc_rows if r["normalisation"] == norm]
            p_all = np.array([r["min_cluster_p"] for r in sub])
            q_all = bh(p_all)
            idx_common = [i for i, r in enumerate(sub) if r["in_common"]]
            q_common = bh(p_all[idx_common])
            for i, r in enumerate(sub):
                r["p_bh_all_channels"] = float(q_all[i]); r["p_bh_common_channels"] = float("nan")
            for j, i in enumerate(idx_common):
                sub[i]["p_bh_common_channels"] = float(q_common[j])
        timings["perchannel"] = time.time() - t0
        pc_csv = out_dir / f"cluster_perm_wholescalp_perchannel_{args.date}.csv"
        with pc_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pc_rows[0])); w.writeheader(); w.writerows(pc_rows)
        for norm in NORMS:
            sub = sorted([r for r in pc_rows if r["normalisation"] == norm], key=lambda r: r["min_cluster_p"])
            n_raw = sum(r["min_cluster_p"] < args.alpha for r in sub)
            n_bh = sum(r["p_bh_all_channels"] < args.alpha for r in sub)
            n_bh_c = sum(r["p_bh_common_channels"] < args.alpha for r in sub if r["in_common"])
            lines += [f"### {norm}: {len(sub)} channels; raw min p < {args.alpha} in {n_raw}; "
                      f"BH over all channels < {args.alpha} in {n_bh}; BH over common channels < {args.alpha} in {n_bh_c}",
                      "", "| channel | ROI | common | n PD/HC | min cluster p | BH (all) | BH (common) |", "|---|---|---|---|---|---|---|"]
            for r in sub:
                qc = f"{r['p_bh_common_channels']:.3f}" if r["in_common"] else "-"
                lines.append(f"| {r['channel']} | {r['roi'] or '-'} | {'yes' if r['in_common'] else 'no'} | "
                             f"{r['n_pd']}/{r['n_hc']} | {r['min_cluster_p']:.4f} | {r['p_bh_all_channels']:.3f} | {qc} |")
            lines.append("")

    # ---- 5. report ----
    lines += ["## Run time", "", ", ".join(f"{k} {v/60:.1f} min" for k, v in timings.items())
              + f"; total {(time.time() - t_start)/60:.1f} min.", "",
              "## Answer", "", "TODO(verify): written after inspecting the numbers above.", ""]
    md = out_dir / f"cluster_perm_wholescalp_{args.date}.md"
    md.write_text("\n".join(lines))
    write_sidecar(md, {**meta, "n_permutations": args.n_perm, "alpha": args.alpha, "p_report": args.p_report,
                       "channel_sets": {"common": common, "ge16": ge16}, "n_time_decim": N_TIME_DECIM,
                       "adjacency": "find_ch_adjacency(standard_1020) x freq x time", "timings_s": timings,
                       "outputs": [str(chan_csv), str(cl_csv), str(md)]})
    write_sidecar(cl_csv, {**meta, "n_permutations": args.n_perm, "alpha": args.alpha})
    print("\n".join(lines[-8:]))
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
