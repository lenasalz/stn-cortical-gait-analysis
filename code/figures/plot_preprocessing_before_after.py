"""Before / after preprocessing supplementary figure (response_matrix VP row 80).

Inputs:
  - $RESIDUAL_DATA_ROOT/<cohort>/<subj>/RawData/EEG/<block>/<block>_raw.set
      (EEGLAB .set, cut to recording length but NOT bandpassed / ICA-cleaned;
       the script also tries _prep.set as a fallback)
  - data/derivatives/raw_clean/<subj>_<block>-eeg-raw.fif
      (built from all_data_*.pkl by build_raw_from_pickle.py — post-ICA,
       post-autoreject, post-CAR, post-bandpass 4–100 Hz)
  - code/analysis/config.yaml (cohort lookup for SMB path)
Outputs:
  - outputs/figures/supplementary/<subj>_<block>_preprocessing_before_after.pdf
  - outputs/figures/supplementary/<subj>_<block>_preprocessing_before_after.pdf.meta.json
Description:
  Two-row, two-column comparison for one representative subject/block:
    row 1 (BEFORE): raw multi-channel time series + raw PSD
    row 2 (AFTER):  cleaned multi-channel time series + cleaned PSD
  The plot demonstrates the cumulative effect of the EEG pre-processing
  pipeline (bandpass 4–100 Hz, CAR, ICA + ICLabel, autoreject) on a
  short window. Channels are restricted to those present in BOTH files
  to make the comparison fair (outer-ring electrodes are dropped during
  preprocessing, so the after-file has fewer channels).

Usage:
  # Point RESIDUAL_DATA_ROOT at the input data first.
  export RESIDUAL_DATA_ROOT=/path/to/data
  python code/figures/plot_preprocessing_before_after.py \
      --subject P4-2003 --block pre8walk
  # Override window:
  python code/figures/plot_preprocessing_before_after.py \
      --subject P4-2003 --block pre8walk --tstart 60 --duration 8
  # Override channel subset (must exist in both files):
  python code/figures/plot_preprocessing_before_after.py \
      --subject P4-2003 --block pre8walk --channels C3,Cz,C4
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from _common import (  # noqa: E402
    PROJECT_ROOT,
    cohort_of,
    load_config,
    output_path,
    resolve_paths,
    setup_run,
    validate_block,
    validate_subject,
    write_sidecar,
)

DEFAULT_CHANNELS = ["F1", "Fz", "F2", "C3", "Cz", "C4"]
OUT_REL = Path("outputs") / "figures" / "supplementary"
PSD_FMAX = 100.0    # match the manuscript's 4–100 Hz bandpass


def _raw_set_path(paths, cohort: str, subject: str, block: str) -> Path:
    """SMB path to the EEGLAB .set file (the 'before' anchor). Tries
    `_raw.set` first (the canonical RawData name) and falls back to
    `_prep.set` for subjects whose folder uses the older naming."""
    base = paths.data_root / cohort / subject / "RawData" / "EEG" / block
    for suffix in ("_raw.set", "_prep.set"):
        cand = base / f"{block}{suffix}"
        if cand.exists():
            return cand
    return base / f"{block}_raw.set"   # canonical fallback so the error message is clear


def _slice_window(raw, t_start: float, duration: float, channels: list[str]):
    """Return (data_uV, times_s, present_channels) for the requested window.

    Raw EEG (no high-pass yet) carries large DC offsets / slow drift; we
    subtract the per-channel mean within the window so each trace sits
    near its stacked offset. The cleaning step is essentially a within-
    window mean removal — equivalent to an extreme high-pass for the
    display only — and does not affect the PSD panel (which uses the
    full unmodified Raw)."""
    sf = raw.info["sfreq"]
    tmax = raw.times[-1]
    t0 = max(0.0, min(t_start, max(0.0, tmax - duration)))
    t1 = min(tmax, t0 + duration)
    picks = [c for c in channels if c in raw.ch_names]
    if not picks:
        raise SystemExit(
            f"none of the requested channels {channels} are in this file "
            f"(file has {len(raw.ch_names)} channels: {raw.ch_names[:5]} …)"
        )
    sl = slice(int(round(t0 * sf)), int(round(t1 * sf)))
    data = raw.get_data(picks=picks)[:, sl] * 1e6                   # to µV
    data = data - data.mean(axis=1, keepdims=True)                  # DC removal per channel
    times = raw.times[sl] - raw.times[sl][0]
    return data, times, picks


def _nice_scale_bar(scale_uV: float) -> float:
    """Pick a rounded µV reference length close to half the inter-channel offset."""
    target = scale_uV / 2
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if step >= target:
            return float(step)
    return float(step)


def _plot_traces(ax, times, data, channels, title, scale_uV):
    """Stacked-trace plot, evenly spaced; each row offset by `scale_uV`.
    A vertical scale bar in the top-right corner makes amplitudes
    directly comparable across panels that share `scale_uV`."""
    offsets = np.arange(len(channels)) * scale_uV
    for i, ch in enumerate(channels):
        ax.plot(times, data[i] + offsets[i], lw=0.6, color="black")
    ax.set_yticks(offsets)
    ax.set_yticklabels(channels)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_ylim(-0.5 * scale_uV, offsets[-1] + 1.6 * scale_uV)
    ax.spines[["top", "right"]].set_visible(False)

    # Scale bar in the top-right corner, well above the topmost channel
    # label, with its µV reading printed above the bar.
    bar_uV = _nice_scale_bar(scale_uV)
    x_span = times[-1] - times[0]
    bar_x = times[0] + 0.98 * x_span
    bar_y0 = offsets[-1] + 0.55 * scale_uV
    bar_y1 = bar_y0 + bar_uV
    ax.plot([bar_x, bar_x], [bar_y0, bar_y1], color="black", lw=1.8,
            solid_capstyle="butt", clip_on=False)
    ax.annotate(
        f"{bar_uV:g} µV",
        xy=(bar_x, bar_y1),
        xytext=(0, 3), textcoords="offset points",
        ha="right", va="bottom", fontsize=8,
    )


def _plot_psd(ax, raw, channels, title):
    """Welch PSD in dB, averaged across the requested channels."""
    picks = [c for c in channels if c in raw.ch_names]
    if not picks:
        ax.set_visible(False)
        return
    spectrum = raw.compute_psd(picks=picks, fmax=PSD_FMAX, verbose="WARNING")
    psd = spectrum.get_data()                                       # (n_ch, n_freq)
    freqs = spectrum.freqs
    psd_db = 10 * np.log10(psd.mean(axis=0) + 1e-30)
    ax.plot(freqs, psd_db, lw=1.0, color="black")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.set_xlim(0, PSD_FMAX)
    ax.spines[["top", "right"]].set_visible(False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--tstart", type=float, default=30.0,
                        help="Window start (s) into the recording. Default 30 to skip "
                             "first-second start-up artifacts.")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Window length (s). Default 10.")
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS),
                        help="Comma-separated channel names to plot. Default "
                             "frontal + motor (F1, Fz, F2, C3, Cz, C4).")
    parser.add_argument("--scale", type=float, default=None,
                        help="Inter-channel vertical offset (µV) for the time-series "
                             "panels. Default: auto (2.5 × 95th-percentile of |signal|).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--anonymize", action="store_true",
                        help="Strip the subject code from figure titles for "
                             "manuscript use (project rule: shipped figures must "
                             "not embed subject IDs). Titles then read "
                             "'representative <cohort> participant'.")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]

    out_dir = PROJECT_ROOT / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}_{block}_preprocessing_before_after.pdf"
    if out_path.exists() and not args.force:
        print(f"skip (exists): {out_path}")
        return 0

    before_path = _raw_set_path(paths, cohort, subject, block)
    after_path = output_path(paths.raw_clean_dir, subject, block, "-eeg-raw.fif")
    for p, name in ((before_path, "raw _prep.set (before)"),
                    (after_path, "raw_clean -eeg-raw.fif (after)")):
        if not p.exists():
            print(f"missing {name}: {p}", file=sys.stderr)
            return 2

    meta = setup_run("plot_preprocessing_before_after", config)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not conform.*")
        before = mne.io.read_raw_eeglab(before_path, preload=True, verbose="WARNING")
        after = mne.io.read_raw_fif(after_path, preload=True, verbose="WARNING")

    common = [c for c in channels if c in before.ch_names and c in after.ch_names]
    if not common:
        print(
            f"none of the requested channels {channels} are in both files. "
            f"Before has {len(before.ch_names)} channels, after has "
            f"{len(after.ch_names)}. First few common channels: "
            f"{sorted(set(before.ch_names) & set(after.ch_names))[:10]}",
            file=sys.stderr,
        )
        return 3
    dropped = [c for c in channels if c not in common]
    if dropped:
        print(f"WARN dropping channels not present in both files: {dropped}",
              file=sys.stderr)

    before_data, before_t, _ = _slice_window(before, args.tstart, args.duration, common)
    after_data,  after_t,  _ = _slice_window(after,  args.tstart, args.duration, common)

    # Auto-scale: pick an offset that comfortably spans both datasets, but
    # let the user override via --scale. Use the 95th percentile of |data|
    # across both panels so the larger raw amplitudes drive the scale.
    if args.scale is None:
        scale_uV = float(
            np.percentile(np.abs(np.concatenate([before_data, after_data], axis=1)), 95) * 2.5
        )
        scale_uV = max(scale_uV, 20.0)
        print(f"auto scale_uV = {scale_uV:.1f}")
    else:
        scale_uV = float(args.scale)

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(11.0, 6.5),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    # Figure titles: anonymise for manuscript use (no subject IDs in shipped
    # figures, per project data rule); keep full identifiers otherwise.
    if args.anonymize:
        # Map internal cohort codes to the manuscript labels (PD / HC) so shipped
        # figures use the paper's terminology rather than the internal group name.
        cohort_disp = {"Patient_NFB": "PD", "AgeMatched_NFB": "HC"}.get(cohort, cohort)
        disp_label = f"representative {cohort_disp} participant"
        suptitle = f"EEG pre-processing: before vs after ({disp_label})"
    else:
        disp_label = f"{subject} {block}"
        suptitle = (f"EEG pre-processing: before vs after "
                    f"(subject {subject}, cohort {cohort}, block {block})")

    _plot_traces(axes[0, 0], before_t, before_data, common,
                 f"Before — {disp_label} ({args.tstart:.0f}–"
                 f"{args.tstart + args.duration:.0f} s)", scale_uV)
    _plot_psd(axes[0, 1], before, common, "Before — PSD (avg of plotted channels)")
    _plot_traces(axes[1, 0], after_t, after_data, common,
                 f"After preprocessing — {disp_label}", scale_uV)
    _plot_psd(axes[1, 1], after, common, "After — PSD (avg of plotted channels)")

    # Share the PSD y-axis across before/after so the two panels are directly
    # comparable (the cleaned spectrum then reads as a reduction of the same
    # data, not as a different recording).
    psd_lo = min(axes[0, 1].get_ylim()[0], axes[1, 1].get_ylim()[0])
    psd_hi = max(axes[0, 1].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[0, 1].set_ylim(psd_lo, psd_hi)
    axes[1, 1].set_ylim(psd_lo, psd_hi)

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "before_input": str(before_path),
        "after_input": str(after_path),
        "channels_requested": channels,
        "channels_plotted": common,
        "channels_dropped": dropped,
        "window_start_s": float(args.tstart),
        "window_duration_s": float(args.duration),
        "anonymized": bool(args.anonymize),
        "output": str(out_path),
    })
    write_sidecar(out_path, meta)
    print(f"wrote {out_path.relative_to(PROJECT_ROOT)} "
          f"(channels={common}, window={args.tstart:.0f}–"
          f"{args.tstart + args.duration:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
