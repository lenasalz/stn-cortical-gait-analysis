"""Per-(subject, block) spectral connectivity for motor-frontal-STN.

Inputs:
  - data/derivatives/epochs/<subj>_<block>-eeg-epo.fif
      (build_epochs_from_raw.py / build_epochs.py)
  - data/derivatives/epochs/<subj>_<block>-dbs-epo.fif       (PD only)
  - data/derivatives/autoreject/<subj>_<block>-mask.npz
      (compute_autoreject_mask.py; applied to drop joint EEG-LFP bad cycles)
  - code/analysis/config.yaml
      (freq_bands; connectivity.channels_of_interest for motor + frontal pools)
Outputs:
  - data/derivatives/connectivity/<subj>_<block>-conn.nc
      xarray Dataset, vars: coh, plv, wpli; dims: (region_pair, band)
  - data/derivatives/connectivity/<subj>_<block>-conn.nc.meta.json
Description:
  Re-implementation of the connectivity analysis from Aline Brunner's
  master thesis (RELab, July 2025), driven by reviewer rows VP-19/VP-23.
  Method follows thesis §II.D.5 verbatim:
    - mne_connectivity.spectral_connectivity_epochs, mode='multitaper',
      faverage=True (one value per band)
    - metrics: coh (coherence), plv (phase-locking value), wpli
      (weighted phase-lag index)
    - region pairs: motor↔frontal (both cohorts), DBS↔motor + DBS↔frontal
      (PD only). Channel pools = Left + Right + Central per region
      (motor = C1,C3,C2,C4,Cz; frontal = AF3,F1,AF4,F2,Fz).
    - cross-channel pair indices built explicitly; values averaged across
      pairs to yield one number per (region_pair, band, metric).
    - bands match the manuscript's 5-band setup (theta, alpha, low-beta,
      high-beta, gamma); delta is excluded because the 4 Hz bandpass
      cut-off makes it meaningless on the preprocessed data.
    - EEG is resampled to 250 Hz (DBS rate) before combining; HC subjects
      yield only the motor-frontal pair (no DBS channel).
    - keep_mask from the per-subject npz is applied before connectivity so
      the same cycles drive every downstream computation in this pipeline.

Usage:
  python compute_connectivity.py --subject P4-2002 --block pre8walk
  python compute_connectivity.py --subject P4-1005 --block baseline
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from _common import (
    atomic_write,
    cohort_of,
    load_config,
    output_path,
    resolve_paths,
    setup_run,
    should_skip,
    validate_block,
    validate_subject,
    write_sidecar,
)

METHODS = ("coh", "plv", "wpli")
TARGET_SFREQ = 250.0     # match the DBS sample rate; thesis §II.D.5


def _pool_channels(config, region: str) -> list[str]:
    """Pool Left + Right + Central channels for a given cortical region.
    Matches the thesis convention of computing connectivity across the
    full motor (or frontal) array rather than within a sub-side."""
    coi = config["connectivity"]["channels_of_interest"][region]
    chans: list[str] = []
    for side in ("Left", "Right", "Central"):
        for c in coi.get(side, []):
            if c not in chans:
                chans.append(c)
    return chans


def _load_eeg_dbs(subject, block, paths, cohort):
    """Load EEG + DBS epochs, apply keep_mask, resample EEG to TARGET_SFREQ.
    Returns (eeg_epochs, dbs_epochs_or_None, n_kept, n_orig, mask_source)."""
    import mne

    eeg_path = output_path(paths.epochs_dir, subject, block, "-eeg-epo.fif")
    dbs_path = output_path(paths.epochs_dir, subject, block, "-dbs-epo.fif")
    mask_path = (
        paths.derivatives_root / "autoreject"
        / f"{subject}_{block}-mask.npz"
    )
    for p, name in ((eeg_path, "EEG epochs"),):
        if not p.exists():
            raise SystemExit(f"missing {name}: {p}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*does not conform.*")
        eeg = mne.read_epochs(eeg_path, preload=True, verbose="WARNING")
    n_orig = len(eeg)
    dbs = None
    if cohort == "Patient_NFB" and dbs_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not conform.*")
            dbs = mne.read_epochs(dbs_path, preload=True, verbose="WARNING")

    if mask_path.exists():
        keep_mask = np.load(mask_path)["keep_mask"].astype(bool)
        mask_source = "per_subject_npz"
        if len(keep_mask) != n_orig:
            print(f"WARN mask length mismatch ({len(keep_mask)} vs {n_orig}); "
                  f"keeping all cycles", file=sys.stderr)
            keep_mask = np.ones(n_orig, dtype=bool)
            mask_source = "length_mismatch_fallback"
    else:
        keep_mask = np.ones(n_orig, dtype=bool)
        mask_source = "no_mask"

    eeg = eeg[keep_mask]
    if dbs is not None:
        if len(dbs) != n_orig:
            print(f"WARN dbs n_epochs ({len(dbs)}) differs from eeg ({n_orig}); "
                  f"applying mask to min length", file=sys.stderr)
            common = min(len(dbs), len(keep_mask))
            dbs = dbs[: common]
            dbs = dbs[keep_mask[: common]]
        else:
            dbs = dbs[keep_mask]

    if eeg.info["sfreq"] != TARGET_SFREQ:
        eeg.resample(TARGET_SFREQ, verbose="WARNING")
    if dbs is not None and dbs.info["sfreq"] != TARGET_SFREQ:
        dbs.resample(TARGET_SFREQ, verbose="WARNING")

    n_kept = int(keep_mask.sum())
    return eeg, dbs, n_kept, n_orig, mask_source


def _combine_to_one_epochs(eeg, dbs, motor_chs, frontal_chs):
    """Build a single Epochs object whose channel order is
    motor (5) + frontal (5) [+ DBS (1)]. Returns (epochs, ch_groups)
    where ch_groups is a dict of {'motor': [idx...], 'frontal': [...],
    'dbs': [idx] or []}. Channels missing from the EEG file are skipped
    with a warning."""
    import mne

    motor_present = [c for c in motor_chs if c in eeg.ch_names]
    frontal_present = [c for c in frontal_chs if c in eeg.ch_names]
    if not motor_present or not frontal_present:
        raise SystemExit(
            f"missing channels: motor present={motor_present}, "
            f"frontal present={frontal_present}; cannot compute "
            f"motor-frontal connectivity"
        )
    picked = eeg.copy().pick(motor_present + frontal_present)
    n_motor = len(motor_present)
    n_frontal = len(frontal_present)
    groups = {
        "motor": list(range(0, n_motor)),
        "frontal": list(range(n_motor, n_motor + n_frontal)),
        "dbs": [],
    }
    if dbs is not None:
        # Concatenate the DBS channel onto each EEG epoch. Rebuild EpochsArray
        # because Epochs.add_channels requires matching info structure that
        # MNE's dbs-misc info will not satisfy out of the box.
        n_ep = min(len(picked), len(dbs))
        picked = picked[: n_ep]
        dbs_pruned = dbs[: n_ep]
        eeg_data = picked.get_data()
        dbs_data = dbs_pruned.get_data()
        # EEG (resampled to 250 Hz) and natively-250 Hz DBS can differ by one
        # sample due to MNE's tmax-inclusive epoch slicing on the DBS side
        # (e.g. baseline blocks: EEG 275, DBS 276). Truncate both to the min
        # so the channel-axis concat works.
        n_t = min(eeg_data.shape[-1], dbs_data.shape[-1])
        eeg_data = eeg_data[..., :n_t]
        dbs_data = dbs_data[..., :n_t]
        combined = np.concatenate([eeg_data, dbs_data], axis=1)
        ch_names = list(picked.ch_names) + list(dbs_pruned.ch_names)
        ch_types = (["eeg"] * picked.info["nchan"]
                    + ["misc"] * dbs_pruned.info["nchan"])
        info = mne.create_info(ch_names=ch_names, sfreq=TARGET_SFREQ,
                               ch_types=ch_types)
        picked = mne.EpochsArray(combined, info, tmin=picked.tmin, verbose="WARNING")
        groups["dbs"] = list(
            range(n_motor + n_frontal, n_motor + n_frontal + dbs_pruned.info["nchan"])
        )
    return picked, groups


def _pair_indices(left: list[int], right: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian product of channel indices: every channel in `left`
    paired with every channel in `right`. Returns
    (sources_idx, targets_idx) ready for spectral_connectivity_epochs."""
    s, t = [], []
    for i in left:
        for j in right:
            s.append(i); t.append(j)
    return np.asarray(s, dtype=int), np.asarray(t, dtype=int)


def _connectivity_per_band(epochs, indices, fmin, fmax):
    """Run spectral_connectivity_epochs for one band (faverage=True) and
    return a dict {method: mean_over_pairs} for METHODS."""
    from mne_connectivity import spectral_connectivity_epochs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {}
        for method in METHODS:
            con = spectral_connectivity_epochs(
                epochs, method=method,
                mode="multitaper", indices=indices,
                sfreq=TARGET_SFREQ, fmin=fmin, fmax=fmax,
                faverage=True, verbose="WARNING",
            )
            data = con.get_data(output="raveled")     # (n_pairs, 1)
            value = float(np.nanmean(np.abs(data)))
            out[method] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    paths = resolve_paths(config)
    subject = validate_subject(args.subject)
    block = validate_block(args.block, config)
    cohort = cohort_of(subject, config)

    out_path = paths.connectivity_dir / f"{subject}_{block}-conn.nc"
    if should_skip(out_path, args.force):
        print(f"skip {subject} {block} (exists)")
        return 0

    meta = setup_run("compute_connectivity", config)
    motor_chs = _pool_channels(config, "motor")
    frontal_chs = _pool_channels(config, "frontal")
    bands = {name: tuple(edges) for name, edges in config["freq_bands"].items()}

    eeg, dbs, n_kept, n_orig, mask_source = _load_eeg_dbs(
        subject, block, paths, cohort
    )
    if n_kept == 0:
        print(f"all cycles rejected for {subject} {block}", file=sys.stderr)
        return 5

    epochs, groups = _combine_to_one_epochs(eeg, dbs, motor_chs, frontal_chs)

    region_pairs = {"motor-frontal": (groups["motor"], groups["frontal"])}
    if groups["dbs"]:
        region_pairs["dbs-motor"] = (groups["dbs"], groups["motor"])
        region_pairs["dbs-frontal"] = (groups["dbs"], groups["frontal"])

    band_names = list(bands.keys())
    rp_names = list(region_pairs.keys())
    n_bands, n_rp = len(band_names), len(rp_names)
    arrays = {m: np.full((n_rp, n_bands), np.nan, dtype=float) for m in METHODS}

    for j, band_name in enumerate(band_names):
        fmin, fmax = bands[band_name]
        for i, rp_name in enumerate(rp_names):
            left, right = region_pairs[rp_name]
            indices = _pair_indices(left, right)
            vals = _connectivity_per_band(epochs, indices, fmin=fmin, fmax=fmax)
            for m in METHODS:
                arrays[m][i, j] = vals[m]

    import xarray as xr
    ds = xr.Dataset(
        {m: (("region_pair", "band"), arrays[m]) for m in METHODS},
        coords={
            "region_pair": rp_names,
            "band": band_names,
        },
        attrs={
            "subject": subject, "cohort": cohort, "block": block,
            "n_kept": n_kept, "n_orig": n_orig,
            "mask_source": mask_source,
            "target_sfreq": TARGET_SFREQ,
            "motor_channels": ",".join([c for c in motor_chs if c in eeg.ch_names]),
            "frontal_channels": ",".join([c for c in frontal_chs if c in eeg.ch_names]),
            "dbs_channels": (
                ",".join(dbs.ch_names) if dbs is not None else ""
            ),
        },
    )

    paths.connectivity_dir.mkdir(parents=True, exist_ok=True)
    with atomic_write(out_path) as tmp:
        ds.to_netcdf(tmp)

    meta.update({
        "subject": subject, "cohort": cohort, "block": block,
        "output": str(out_path),
        "n_orig": int(n_orig),
        "n_kept": int(n_kept),
        "mask_source": mask_source,
        "region_pairs": rp_names,
        "bands": band_names,
        "methods": list(METHODS),
        "target_sfreq": TARGET_SFREQ,
    })
    write_sidecar(out_path, meta)
    print(
        f"wrote {out_path.name}  region_pairs={rp_names}  bands={band_names}  "
        f"kept={n_kept}/{n_orig} (mask_source={mask_source})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
