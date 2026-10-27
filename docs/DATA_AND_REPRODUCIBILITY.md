# Data and reproducibility scope

## Publicly included

This repository includes source code implementing the manuscript's main computational definitions, documented experiment rules and synthetic smoke tests; manuscript numeric-summary CSVs are not redistributed in this code-only package.

## Not publicly included

The following are intentionally excluded:

1. Raw or processed hourly precipitation/discharge observations supplied by the Guangzhou Hydrology Bureau.
2. Trained neural-network checkpoints produced from those restricted observations.
3. Intermediate arrays that would effectively redistribute provider-supplied time series.
4. Local absolute filesystem paths and machine-specific project directories.

## What can be reproduced without restricted data

The public package verifies sequence indexing, the TCN/FiLM forward architecture, source-derived componentwise coefficient projection, exact small-source convex-hull projection, sampling conventions, metric definitions, and the manuscript's fixed hyperparameter/configuration values.

## What requires authorized data access

Exact reproduction of Tables 3–5, Figs. 4–6, and Supplementary Tables S1–S12 requires the original hourly Dongjiang observations and, for replay-only diagnostics, trained checkpoints. Access to the observational data remains subject to the data provider's authorization.

## Numerical reproducibility note

The manuscript's replay audit found metric-level agreement despite small float32/GPU-level differences in a limited number of predictions. The public reference implementation is intended to preserve the stated numerical conventions rather than guarantee bitwise equivalence across hardware/software stacks.
