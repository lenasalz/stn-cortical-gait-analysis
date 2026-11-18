# Residual activity — analysis code

Analysis code for the manuscript by Salzmann, Brunner et al.
<!-- TODO(verify): exact paper title, journal and DOI once available -->

This repository contains code only. No data, derivatives, figures or
manuscript files are included.

## Contents

- `code/preprocessing/` — builds per-subject continuous recordings
- `code/analysis/` — per-subject analysis stages, aggregation and statistics
  (Python), with the linear mixed models in `code/analysis/R/`
- `code/analysis/revision/` — additional analyses
- `code/figures/` — figure scripts
- `code/scripts/` — shell wrappers that chain the stages
- `tests/` — checks for selected analysis functions

Most scripts start with a header listing their inputs, outputs and purpose.

## Setup

    conda env create -f environment.yml
    conda activate residual_activity

`environment.yml` pins the Python package versions the analyses were run with.

The R scripts were run with R 4.5.1 and these package versions:

| Package | Version |
|---|---|
| lme4 | 1.1.37 |
| lmerTest | 3.1.3 |
| emmeans | 1.11.2.8 |
| nlme | 3.1.168 |
| dplyr | 1.2.1 |
| tidyr | 1.3.2 |
| tibble | 3.3.0 |
| data.table | 1.17.8 |
| readxl | 1.5.0 |

## Data location

The data are not part of this repository. Scripts read the input location
from the environment variable `RESIDUAL_DATA_ROOT` (or `paths.data_root` in
`code/analysis/config.yaml`):

    export RESIDUAL_DATA_ROOT=/path/to/data

Derivatives are written to `data/derivatives/` and outputs to `outputs/`,
both relative to the repository root and both ignored by git.

## License

MIT — see `LICENSE`.
