# Environment and reproducibility scope

## Scientific protocol

This repository implements PS-FiLM-TCN (M6 ParameterClip), the inference-time
source-parameter-support constraint, together with the frozen MS-TCN(RS),
Raw FiLM, and Persistence comparators. Raw FiLM and PS-FiLM share one trained
Shared-FiLM checkpoint. MS-TCN(RS) is trained independently. PS inference does
not create a new trained model.

The formal design covers six held-out basin folds and five seeds (42, 123,
2024, 2025, 3407). The model input uses 24 *stored records*, without imposing
hourly reindexing or filling missing observations. The indexed target is at
`i+23`, with the original `N-24` dataset-length convention. Inputs include
precipitation at the target record and prior observed discharge state, so the
published experiments must not be described unconditionally as operational
one-hour-ahead forecasts issued before the target hour. Fold-specific scalers
and the PS source envelope use training data from source basins only; no held-out
basin records are used to fit them. Historical metrics are evaluated on the
original held-out valid-target mask.

## Validated environment snapshot

The environment below was reported by a local, read-only audit on 2026-10-20:

| Component | Observed version |
| --- | --- |
| PyTorch | 2.4.1+cu121 |
| NumPy | 1.26.4 |
| pandas | 2.1.4 |
| scikit-learn | 1.8.0 |
| SciPy | 1.16.3 |
| Matplotlib | 3.10.8 |
| PyYAML | 6.0.3 |
| pytest | 9.0.2 |

`requirements.txt` pins the reported Python package versions and is a draft
replay-environment manifest. It has not yet been verified by performing a clean
installation on a second machine. CUDA `+cu121` comes from the observed wheel
build; CPU- versus CUDA-specific PyTorch installations require appropriate
platform-specific wheel sources. Python interpreter version and all transitive
packages must be captured separately for a fully locked environment.

## Local checks

Run commands from the repository root using an appropriately prepared Python
environment. These commands do not download restricted observations:

```shell
python -m pytest tests -q
python examples/run_stage2e3_smoke_test.py
```

The smoke test uses generated data and temporary model states, and does **not**
reproduce historical Dongjiang basin metrics. Consult
`docs/STAGE2E3_REPLAY_PROTOCOL.md` for a single-fold replay that requires a
locally authorized manifest, trusted checkpoints and explicit `--execute`.
The authorized inputs, checkpoints and aggregate JSON output must be stored
outside the repository. No training is required to replay frozen checkpoints.

## Historical evidence and limits

All 30 six-basin LOBO fold/seed cases were replayed against saved Stage18
aggregate results. Twenty-nine satisfy the numerical comparison policy v2;
one extreme Raw-FiLM dry-period NSE deviation has a separately documented
post-hoc numerical review and **does not** satisfy v2's strict criterion.
See `docs/STAGE2E4_REPLAY_EVIDENCE_STATUS.md` for the precise distinction.

This comparison demonstrates aggregate-metric checkpoint reproducibility within
its stated tolerances. It does not prove bitwise identity of every hourly
prediction, retraining determinism or independent reproduction without
restricted observations. Access to private hydrological data, historical model
weights or related artifacts must be separately authorized.
