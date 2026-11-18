"""Shared helpers for the per-subject derivative pipeline.

Inputs:  code/analysis/config.yaml; environment variable RESIDUAL_DATA_ROOT (optional override).
Outputs: helper functions only — no side effects on import.
Description: config loading, path resolution, anonymized-code validation,
seed/version stamping, and a skip-if-exists primitive used by every
compute script under code/analysis/.

Every compute script must call `setup_run(stage_name)` at the top of `main()`
to set the random seed, log library versions, and emit a sidecar JSON next
to its output file.
"""
from __future__ import annotations

import json
import os
import platform
import random
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "code" / "analysis" / "config.yaml"

# Subject codes must match the project's anonymization scheme. Anything
# else is rejected at the API boundary so we cannot accidentally accept
# a name or a hospital ID through argparse.
SUBJECT_CODE_RE = re.compile(r"^P4-\d{4}$")


@dataclass(frozen=True)
class Paths:
    data_root: Path
    derivatives_root: Path
    raw_clean_dir: Path
    epochs_dir: Path
    tfr_dir: Path
    psd_dir: Path
    fooof_dir: Path
    connectivity_dir: Path
    band_power_dir: Path
    tfr_gavg_dir: Path
    index_csv: Path


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read config.yaml. Raises FileNotFoundError if missing — never silently
    falls back, so a misconfigured run is loud. Also runs validate_config
    so anonymization-format violations on cohort rosters fail fast."""
    p = Path(path) if path else CONFIG_PATH
    with p.open() as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Reject any cohort member that does not match the anonymized format."""
    bad = [
        s for members in cfg.get("cohorts", {}).values()
        for s in members if not SUBJECT_CODE_RE.match(s)
    ]
    if bad:
        raise ValueError(
            f"Non-anonymized subject code(s) in config.yaml: {bad}. "
            "Project rule: refer to subjects by anonymized codes only."
        )


def resolve_paths(config: dict[str, Any]) -> Paths:
    """Resolve config paths into absolute Paths.

    `data_root` resolution order: env var $RESIDUAL_DATA_ROOT, then config
    `paths.data_root`. If neither is set, a warning is printed and
    `data_root` points at a placeholder that does not exist, so stages that
    only read derivatives still run and stages that read input data fail
    with a path naming the missing setting. Derivative dirs are always under
    the project tree (PROJECT_ROOT / paths.derivatives_root) and are
    auto-created.
    """
    root = os.environ.get("RESIDUAL_DATA_ROOT") or config["paths"].get("data_root")
    if root:
        data_root = Path(root)
    else:
        import warnings

        warnings.warn(
            "Input data location is not set: export RESIDUAL_DATA_ROOT or fill "
            "in paths.data_root in code/analysis/config.yaml.",
            stacklevel=2,
        )
        data_root = PROJECT_ROOT / "RESIDUAL_DATA_ROOT_is_not_set"
    derivatives_root = PROJECT_ROOT / config["paths"]["derivatives_root"]

    def sub(key: str) -> Path:
        return derivatives_root / config["paths"][key]

    paths = Paths(
        data_root=data_root,
        derivatives_root=derivatives_root,
        raw_clean_dir=sub("raw_clean_dir"),
        epochs_dir=sub("epochs_dir"),
        tfr_dir=sub("tfr_dir"),
        psd_dir=sub("psd_dir"),
        fooof_dir=sub("fooof_dir"),
        connectivity_dir=sub("connectivity_dir"),
        band_power_dir=sub("band_power_dir"),
        tfr_gavg_dir=sub("tfr_gavg_dir"),
        index_csv=derivatives_root / config["paths"]["index_csv"],
    )
    for d in (
        paths.derivatives_root,
        paths.raw_clean_dir,
        paths.epochs_dir,
        paths.tfr_dir,
        paths.psd_dir,
        paths.fooof_dir,
        paths.connectivity_dir,
        paths.band_power_dir,
        paths.tfr_gavg_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def validate_subject(code: str) -> str:
    """Reject anything that is not an anonymized P4-XXXX code."""
    if not SUBJECT_CODE_RE.match(code):
        raise ValueError(
            f"Subject code {code!r} is not in the anonymized format P4-XXXX. "
            "Project rule: refer to subjects by anonymized codes only."
        )
    return code


def validate_block(block: str, config: dict[str, Any]) -> str:
    # `extra_blocks` are processable per-subject on demand (e.g. post8walk
    # QA/exploration) but are NOT part of the manuscript pipeline: every
    # `--all` sweep (run_all.py, build_raw_per_subject.py, build_epochs.py)
    # iterates config["blocks"] only.
    allowed = list(config["blocks"]) + list(config.get("extra_blocks", []))
    if block not in allowed:
        raise ValueError(f"Block {block!r} not in config blocks {allowed}.")
    return block


def cohort_of(subject: str, config: dict[str, Any]) -> str:
    for cohort, members in config["cohorts"].items():
        if subject in members:
            return cohort
    raise ValueError(f"Subject {subject} not present in any cohort in config.yaml.")


def output_path(stage_dir: Path, subject: str, block: str, suffix: str) -> Path:
    """Canonical per-(subject, block) output path: <subj>_<block><suffix>."""
    return stage_dir / f"{subject}_{block}{suffix}"


@contextmanager
def atomic_write(out_path: Path):
    """Yield a temp path that gets `os.replace`d onto `out_path` on success.

    Avoids partial-write corruption when two `--force` runs race on the
    same `(subject, block)`. Compute scripts write to the yielded path,
    not to `out_path` directly.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=".tmp",
        dir=out_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        yield tmp_path
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def should_skip(out: Path, force: bool) -> bool:
    """Skip-if-exists primitive used by every compute script.
    Returns True iff the output already exists and --force was not passed."""
    if force:
        return False
    if out.exists() and out.stat().st_size > 0:
        return True
    return False


def patch_legacy_annotations(mne_obj) -> None:
    """Backfill `_extras` on Annotations carried by legacy-pickled MNE objects.

    The cohort/epoch pickles were saved with an older MNE whose Annotations
    object did not carry an `_extras` attribute. MNE 1.12+ accesses
    `annotations.extras` on save and AttributeErrors out. Set `_extras` to
    `[None] * len(annotations)` defensively so the object can round-trip
    through `Raw.save` / `Epochs.save`.
    """
    ann = getattr(mne_obj, "annotations", None)
    if ann is None or len(ann) == 0:
        return
    if getattr(ann, "_extras", None) is None:
        ann._extras = [None] * len(ann)


def write_sidecar(out_path: Path, info: dict[str, Any]) -> None:
    """Write a `<out>.meta.json` next to every artifact, recording seed,
    library versions, config hash, and run timestamp. Required by
    project repro rule 1 ("set a random seed at the top") and rule 5
    ("docs/pipeline.md mapping figure → script → input → output").
    """
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    with meta_path.open("w") as f:
        json.dump(info, f, indent=2, default=str)


def setup_run(stage: str, config: dict[str, Any]) -> dict[str, Any]:
    """Set seed, return a metadata dict every script writes alongside its output."""
    seed = int(config["repro"]["seed"])
    np.random.seed(seed)
    random.seed(seed)
    try:
        import mne  # imported lazily so non-MNE stages don't pay the cost
        mne.set_log_level("WARNING")
        mne_version = mne.__version__
    except ImportError:
        mne_version = None

    return {
        "stage": stage,
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "mne": mne_version,
        "config_path": str(CONFIG_PATH),
    }
