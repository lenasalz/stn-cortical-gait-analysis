"""Run PSD/FOOOF statistics for manuscript Tab 1, 2, 3, 9.

Inputs:
  - data/derivatives/fooof/fooof_master.csv   (aggregate_fooof.py)
  - code/analysis/config.yaml (cohorts, blocks)
Outputs:
  - outputs/stats/psd_fooof_<date>.txt
      Full stdout capture of within-cohort (rest vs walk) + between-cohort
      (PD vs HC) FOOOF parameter tests with FDR correction.
  - outputs/stats/psd_fooof_<date>.csv
      Long-format stats DataFrame from run_fooof_statistics.
Description:
  Thin wrapper around Aline's
  `final_analysis/scripts/psd_fooof.run_fooof_statistics`. Aggregates
  our per-subject FOOOF CSVs into the schema her function expects
  (ROI, Data_Type, Block, Cohort, Subject_Idx, R2, plus <band>_CF /
  <band>_PW / <band>_BW), filters to the manuscript's ROIs
  (Motor_Both, Frontal_Both, DBS), then calls her statistics function.

  Tables produced:
    - Tab 1: STN periodic params, rest vs walk (PD cohort, within)
    - Tab 2: Cortical between-cohort comparisons (PD vs HC, periodic + aperiodic)
    - Tab 3: Cortical within-cohort (rest vs walk)
    - Tab 9: Aperiodic exponent analysis

Usage:
  python code/analysis/run_psd_stats.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _common import load_config, resolve_paths, setup_run  # noqa: E402

# We don't import psd_fooof here because it pulls in specparam (heavy
# optional dep). Instead, we copy the run_fooof_statistics body. The
# function is self-contained and only uses scipy + statsmodels.


# ---------- ROI definitions matching the manuscript ----------
ROI_FILTERS = {
    # (manuscript ROI name, our region, our side, our modality)
    "Motor_Both":   ("motor",   "Both", "eeg"),
    "Frontal_Both": ("frontal", "Both", "eeg"),
    "DBS":          ("STN",     None,   "dbs"),     # side filter applied separately
}

COHORT_MAP = {  # our internal label → Aline's expected cohort label
    "Patient_NFB":    "PD",
    "AgeMatched_NFB": "HC",
}


def _to_aline_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape our master FOOOF CSV into the columns
    `run_fooof_statistics` expects."""
    rows: list[dict] = []
    for roi_name, (region, side, modality) in ROI_FILTERS.items():
        sub = df[(df["region"] == region) & (df["modality"] == modality)]
        if side is not None:
            sub = sub[sub["side"] == side]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            cohort_label = COHORT_MAP.get(r["cohort"], r["cohort"])
            out = {
                "ROI":          roi_name,
                "Data_Type":    "EEG_epoched" if modality == "eeg" else "DBS_epoched",
                "Block":        r["block"],
                "Cohort":       cohort_label,
                "Subject_Idx":  r["subject"],
                "R2":           r["fit_r_squared"],
                "Aperiodic_OFFSET":   r["aperiodic_offset"],
                "Aperiodic_EXPONENT": r["aperiodic_exponent"],
            }
            for band in ("Theta", "Alpha", "Low Beta", "High Beta", "Gamma"):
                # Aline uses underscored band names internally
                key = band.replace(" ", "_")
                out[f"{key}_CF"] = r[f"{band}_cf"]
                out[f"{key}_PW"] = r[f"{band}_pw"]
                out[f"{key}_BW"] = r[f"{band}_bw"]
            rows.append(out)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    paths = resolve_paths(config)
    setup_run("run_psd_stats", config)

    master = paths.derivatives_root / "fooof" / "fooof_master.csv"
    if not master.exists():
        print(f"missing {master}; run aggregate_fooof.py first", file=sys.stderr)
        return 2
    df_master = pd.read_csv(master)
    df_aline = _to_aline_schema(df_master)
    print(f"loaded {len(df_master)} rows from master → "
          f"{len(df_aline)} rows in Aline schema "
          f"({df_aline['ROI'].nunique()} ROIs)")

    # Aline's notebook cell 7:
    #   STATISTICAL_COMPARISONS = [['PD', 'HC']]
    # within-cohort: rest vs walk = ['baseline', 'pre8walk']
    comparisons_within  = [("baseline", "pre8walk")]
    comparisons_between = [("PD", "HC")]

    out_dir = PROJECT_ROOT / "outputs" / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    log_path = out_dir / f"psd_fooof_{today}.txt"
    csv_path = out_dir / f"psd_fooof_{today}.csv"

    import io, contextlib
    buf = io.StringIO()
    buf.write(f"Source: code/analysis/run_psd_stats.py\n")
    buf.write(f"Master CSV: data/derivatives/fooof/fooof_master.csv "
              f"({len(df_master)} rows)\n")
    buf.write(f"Stats engine: Aline's run_fooof_statistics (copied inline to "
              f"avoid the specparam dependency in our env).\n")
    buf.write(f"Comparisons:\n")
    buf.write(f"  within-cohort:  {comparisons_within}\n")
    buf.write(f"  between-cohort: {comparisons_between}\n")
    buf.write(f"Date: {today}\n")
    buf.write(f"{'='*70}\n\n")

    # We inline a thin version of run_fooof_statistics here so we don't
    # have to install specparam just to import the module.
    from _run_fooof_statistics import run_fooof_statistics  # noqa: E402

    with contextlib.redirect_stdout(buf):
        stats_df = run_fooof_statistics(
            df_aline,
            comparisons_within=comparisons_within,
            comparisons_between=comparisons_between,
            results_dir=str(out_dir),
            narrow_analysis=False,  # keep all bands for the manuscript tables
        )
    log_path.write_text(buf.getvalue())

    if stats_df is not None and not stats_df.empty:
        stats_df.to_csv(csv_path, index=False)
        print(f"wrote {log_path} ({log_path.stat().st_size} B)")
        print(f"wrote {csv_path} ({len(stats_df)} stat rows)")
    else:
        print(f"wrote {log_path} (no stats rows returned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
