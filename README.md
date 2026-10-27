# PS-FiLM-TCN: source-constrained FiLM modulation for zero-shot streamflow prediction

Reference **code-only** implementation accompanying the manuscript
*Source-constrained FiLM modulation for zero-shot cross-catchment streamflow prediction*.

## What this package provides

This package contains tested computational implementations of the TCN and FiLM
architecture, source-derived coefficient projection, source-only preprocessing,
basin-balanced training, convex-hull analysis, evaluation metrics, and examples
for synthetic-data smoke tests. It is a **method-level reproducibility package**.

It does **not** include the Dongjiang hourly precipitation/discharge records,
trained model checkpoints, saved prediction arrays, or manuscript result CSVs.
Without authorized study observations and checkpoints, this package cannot
independently regenerate the numerical results in the manuscript.

## Contents

- `src/`: data-independent reference implementations, training and inference tools.
- `examples/`: synthetic smoke tests and entry points for locally authorized inputs.
- `tests/`: unit tests using synthetic or test-generated arrays.
- `supplementary_reference/`: reference joint-support geometry implementation.
- `configs/authorized_data_manifest.example.json`: **placeholder only**, no private paths.
- `docs/`: reproducibility boundaries, exact source/target conditions, replay protocols,
  and historical validation status.
- `environment/`: minimum requirements and local tested-version specifications.
- `requirements.txt`: frozen local environment snapshot (not a clean-install guarantee).

## Quick checks (from repository root)

```bash
python -m pip install -r environment/requirements-min.txt
python examples/run_smoke_test.py
python -m pytest tests -q
```

The smoke test uses **synthetic data**. Successful execution is not evidence
of numerical reproduction of the published Dongjiang metrics.

## Scientific scope and caveats

The reported formal study uses source-only scaler fitting and coefficient
bounds, 24 stored records per input sequence (without forced hourly reindexing),
and target-hour precipitation together with antecedent observed discharge.
Consequently it must not be described as a target-hour forecast issued before
that hour's rainfall is observed.

The frozen replay comparison covered 30 LOBO fold/seed cases. **29 met** the
specified v2 numerical comparison policy; one extreme Zengjiang Raw-FiLM dry
NSE case was a separately reviewed exception, not a strict match. Details are
in `docs/STAGE2E4_REPLAY_EVIDENCE_STATUS.md`.

## Data rights and citation

Observational data and trained weights are excluded. Access to original
observations is subject to data-provider permission.

**License and citation:** this code is provided under the MIT License (see
`LICENSE`). Software authors and the preferred citation are recorded in
`CITATION.cff`. A Zenodo DOI will be added after an archival release.

**Reproducibility boundary:** this repository distributes code only. The
restricted hydrometeorological data, trained checkpoints, saved predictions,
and result CSVs are not included; their absence prevents independent numerical
reproduction of the paper's reported catchment scores.
