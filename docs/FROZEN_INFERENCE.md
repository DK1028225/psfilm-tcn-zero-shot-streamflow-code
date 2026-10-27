# Frozen LOBO checkpoint replay and evaluation

This module reproduces the **Stage18B inference and evaluation calculations** for
one held-out basin, using already prepared, authorized inputs. It does not train
a model, select hyperparameters, or use target-basin labels for scaler fitting.

## Core modules

- `src/inference.py` contains the Stage18B-compatible inverse flow conversion,
  full/seasonal/post-2022 metrics, persistence skill, and one-checkpoint paired
  Raw/PS-FiLM inference.
- `src/prepared_fold.py` validates an already preprocessed private fold with the
  formal `N-24` window indexing convention and provides a sequential DataLoader.
- `examples/run_inference_smoke_test.py` checks Raw/PS checkpoint identity and
  masked target alignment with synthetic tensors.
- `examples/run_prepared_fold_smoke_test.py` generates a temporary synthetic
  private NPZ and checkpoint, invokes the actual command-line replay entry,
  and removes the temporary files.
- `examples/replay_prepared_fold.py` is the restricted-data replay entry point.

## Command-line replay on authorized local inputs

```bash
python examples/replay_prepared_fold.py \
  --prepared-input /path/to/PRIVATE/prepared_heldout_fold.npz \
  --checkpoint /path/to/PRIVATE/Shared-FiLM/best_model.pth \
  --summary-out /path/to/PRIVATE/aggregate_metrics.json \
  --device cpu
```

The same frozen `Shared-FiLM` checkpoint is loaded strictly once. Raw inference
uses generated FiLM parameters without restriction. PS inference computes
componentwise gamma/beta limits from the **training-source static descriptors**
and applies these limits inside the same model's conditioned blocks. Target
records, checkpoint weights, and dates must match for both methods.

**Do not upload the prepared private NPZ or full prediction series to GitHub**.
The default CLI writes only aggregate metrics in JSON.

## Private prepared fold schema

The NPZ must use numeric/string NumPy arrays, not object arrays. All arrays
must originate from the fixed preprocessing and source-only scaling protocol.

| Key | Shape | Meaning |
| --- | --- | --- |
| `dynamic_x` | `(N,24,2)` | Source-fitted standardized rain and antecedent runoff inputs |
| `target_scaled` | `(N,)` | Already transformed and standardized target labels, for interface alignment |
| `missing_mask` | `(N,)` | `0` for valid original Q target, `1` for missing Q target |
| `q_previous` | `(N,)` | Previous stored-row processed discharge in m³/s |
| `target_dates` | `(N,)` | ISO date/time strings for the target rows |
| `q_full` | `(N+24,)` | Processed basin discharge (private); observation for sample i is row i+23 |
| `source_static_scaled` | `(S,4)` | Fold training-source static descriptor vectors scaled with source-only scaler |
| `target_static_scaled` | `(4,)` | Held-out static descriptor transformed using that same source-only scaler |
| `y_scaler_mean` | scalar or `(1,)` | Source-only target-log-modulus scaler mean |
| `y_scaler_scale` | scalar or `(1,)` | Source-only target-log-modulus scaler scale |
| `basin_area_km2` | scalar or `(1,)` | Held-out basin drainage area |
| `heldout_name` | optional scalar string | For output provenance |
| `source_names` | optional `(S,)` strings | For checking held-out exclusion |
| `basin_id` | optional scalar int | Held-out dataset index |

**Safety boundary:** This private NPZ itself must never be published because it
contains observational runoff values and may be subject to provider licensing.
The adapter validates shapes and source-name exclusion, but cannot prove from
an NPZ alone that the original scalers were actually source-only fitted. That
provenance must be enforced by the forthcoming Stage2E preparation pipeline.

## Statistical conventions

- Valid target: `missing_mask == 0`, corresponding to the formal original-Q mask.
- Flow inverse: `StandardScaler.inverse_transform`, `expm1`, nonnegative clipping,
  and multiplication by basin area/1000.
- NSE, KGE, RMSE, MAE, PBIAS use the Stage1 frozen formal definitions.
- Flood-season NSE uses April–September and requires 100+ target records.
- Post-2022 metrics require at least 10 target records; they report an explicit
  NaN otherwise.
- Persistence relative skill uses the common finite three-array mask.
- The JSON writer substitutes `null` for nonfinite aggregate values.

## What this stage does not do

This stage does not reconstruct the six-basin input files, retrain networks,
run MS-TCN(RS), select PS-FiLM hyperparameters, or automatically orchestrate
all 30 held-out folds and seeds. The separate Stage2C training module is frozen,
and the subsequent work will connect it to an authorized preprocessing adapter
and official fold scheduler.
