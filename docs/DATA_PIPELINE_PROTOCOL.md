# Frozen Stage16 and Stage18 data protocol

This module extracts unchanged AST definitions from the archived official experimental scripts:

- Stage16: three-source weighted sampling, `1.0 / 0.5 / 1.0` per sample.
- Stage18A/18B: five-source regional LOBO sampling, inverse source-dataset size per sample.
- All three scripts contain AST-identical implementations of the six core data functions.

## Important conditions

1. Original UTF-8-SIG CSV reading; GB18030 on UnicodeDecodeError.
2. Column names and Thiessen weights come directly from the formal code, without renormalization.
3. Rain station nonnumeric and missing values become zero.
4. Discharge missing marker and original-negative marker are retained. Discharge itself is filled/clipped to zero.
5. Log runoff modulus and lag-1 state are constructed **before invalid dates are dropped**.
6. Invalid timestamps are removed; duplicate or non-hourly timestamps are preserved, with no resampling.
7. Train/validation/test partition lengths use `int(N * .70)` and `int(N * .15)`.
8. Three scalers are fit on source-training data only; static scaler has one 4-attribute vector per source basin.
9. Dataset targets begin at row 23; `len(dataset) = len(df) - 24`, excluding the final potential target.
10. Window `i:i+24` includes target precipitation and lagged runoff states. The actual hourly spacing is not guaranteed.
11. Mask `0` means the target is valid. Dataset emits the historical seven-tuple `(x, static, y, mask, basin_id, prior_q, time_label)`.
12. Both sampling modes use `replacement=True` and draw `len(concatenated_training_dataset)` examples per epoch.
13. Held-out dataset and loader reuse the formal `collect_heldout_base_arrays` definition; the held-out basin does not enter scaler fitting or training sampling.
14. The module does not perform model fitting or evaluate target performance.

## Testing and evidence boundaries

Synthetic smoke tests validate mechanics and confirm the six frozen AST definitions have the same behavior as the archived official code. Running this package alone **does not** establish reproduction of historical neural-network checkpoints or published NSE values. Public files must never include restricted hourly observations, full NPZ predictions or trained checkpoints.

## Safety and merge rule

The package is staged in `public_files/` and is **not** to be copied over an existing repository file without inspecting differences. Run the separate local-only inspection tool first. Commit only after testing in the final repository with `torchhydro_env`.
