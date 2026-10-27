# Stage2E-4 historical checkpoint replay status

## Coverage

- Frozen experiment: six-basin leave-one-basin-out (LOBO), five official random
  seeds, 30 fold/seed evaluations.
- Each case reuses the separately trained MS-TCN(RS) and Shared-FiLM checkpoints.
  Raw FiLM and PS-FiLM use the same Shared-FiLM state; Persistence is a
  non-trained comparator.
- All 30 cases were actually replayed against authorized local inputs and
  compared with historical Stage18 aggregated model results.
- The aggregate comparison checks 72 scalar values per case (four reported
  models). The historical table also includes additional ablation models not
  independently replayed in Stage2E-4.

## Final numerical assessment

- **29/30 cases satisfy the Stage2E-4 v2 numerical comparison policy.**
  This count includes the seed-42 Andunhe case: its immutable v1 report retains
  two post-2022 PBIAS discrepancies; v2 uses a narrowly documented
  0.00002-percentage-point acceptance cap for that metric.
- **1/30 case does not satisfy the v2 criterion:** held-out Zengjiang,
  seed 2024, Raw FiLM (`FiLM-TCN`) dry-season `NSE_dry`.
  Historical result `-4064636025508.155`; replay
  `-4064645857866.642`; absolute difference `9832358.487304688`;
  relative difference `2.41900097e-6` (about `0.0002419%`).
- The separately reviewed extreme-NSE difference is a **post-hoc numerical
  exception**, not a v2 PASS. All 71 other compared values in that case satisfy
  v2's policy, including all PS-FiLM values and held-out valid-target counts.
- There are zero unexecuted fold/seed cases; the original failed comparison
  records are retained and not overwritten.

## Interpretation and audit boundary

The raw model's extremely negative NSE shows catastrophic negative transfer in
this case; it also amplifies small numerical changes when squared prediction
errors are aggregated. Numerical precision is a plausible explanation for the
replay mismatch, but it has not been isolated with a timestamp-level comparison.
Do not represent the evidence as 30/30 strict matches or claim confirmed
bitwise-identical predictions.

The complete audit JSON and Markdown generated in Stage2E-4 are retained
privately outside the Git repository. This page conveys summary findings and
contains no original hydrological time series, hourly predictions, private
local absolute paths or checkpoint content.
