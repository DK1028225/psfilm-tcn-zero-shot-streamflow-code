# Stage2E-3: Frozen single-fold checkpoint replay and evaluation

## Scientific identity

The method is PS-FiLM-TCN / M6 ParameterClip. The evaluator reuses the already trained Shared-FiLM checkpoint twice: (a) Raw FiLM, with no coefficient constraint; (b) PS-FiLM, with channel-wise gamma/beta empirical source-support bounds at inference. Neither arm trains another model or alters weights. The MS-TCN(RS) checkpoint is independently trained. Persistence is computed from the previous stored record.

## Reused modules

- `src/data_pipeline.py` (Stage2E-1): frozen original preprocessing, train/validation split, source-only x/y/static scaler fitting, `N-24` window construction, held-out loader.
- `src/formal_training_models.py` (Stage2E-2): MS-TCN(RS) frozen architecture and shape-compatible weights.
- `src/inference.py` (Stage2D): raw/PS same-checkpoint predictions, unit inverse transform, target mask, all reported metrics, source envelopes.
- `src/formal_fold_replay.py` (Stage2E-3): read-only connection layer, separate checkpoint type/seed checks, exact observation/time/persistence alignment, four-model evaluation.

## Frozen causality and data rules

- Original stored record order is retained. No hourly reindexing, interpolation or deduplication.
- Precipitation of the target stored record is available; this is not automatically an operational t-1-issued forecast.
- The lagged target-discharge state is created before chronological source partitioning. The model never consumes current observed discharge as an input.
- Dynamic window length 24 stored records. Target index `i+23`; candidate dataset length `N-24`.
- Target discharge missingness is masked; negative discharge is clipped to zero in the historical formal preprocessing and must not be silently discarded.
- All scalers and parameter envelopes use current-fold source basins only. The held-out basin does not fit any scaler, loss or model parameter.
- Source train/validation ratio and sampling weights remain defined by Stage2E-1; Stage2E-3 never modifies them.

## Models, metrics and evidence limits

The aggregate report has four entries: MS-TCN(RS), FiLM-TCN, PS-FiLM-TCN and Persistence. Every model must share identical valid target counts, observed discharge, persistence discharge and stored dates. Reported metrics reuse the existing Stage2D NSE, KGE, RMSE, MAE, PBIAS, flood/dry NSE, persistence-relative skill and post-2022 robust subset conventions. Persistence contains no fitted parameters.

`examples/run_stage2e3_smoke_test.py` verifies the integration only with generated data and temporary checkpoints. It cannot substantiate a claim of reproducing historical Dongjiang results. True historical checkpoint replay requires authorization and comparison against existing Stage18 results; the latter must be recorded separately, including tolerance and environment settings.

## Authorized single-fold replay (no retraining)

Use local manifest fields from `configs/authorized_data_manifest.example.json` but keep the populated manifest outside the public repository. First run `--check`; this verifies paths, official seed, role and output protections without reading data or loading checkpoints. Only after confirming the sources/fold and your access, run `--execute`, explicitly, to load authorized data, fit source-only scalers, load trusted local checkpoints and write aggregate JSON outside the repository.

```
python examples/run_authorized_fold_replay.py --manifest G:/private/manifest.json --experiment lobo --heldout Zengjiang --seed 42 --ms-checkpoint G:/private/.../MS-TCNRS/best_model.pth --film-checkpoint G:/private/.../Shared-FiLM/best_model.pth --summary-out G:/private/results/replay.json --check
```

No Stage2E-3 program automatically trains models. It must never overwrite existing checkpoints or historical predictions, never publish authorized data, and never commit private paths, NPZ predictions or checkpoints. The GitHub repository remains private during this validation stage.
