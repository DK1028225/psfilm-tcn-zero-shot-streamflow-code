# Stage2E-2 single-fold formal training entry

This layer assembles existing **Stage2E-1** source-only data partitions and weighted samplers with the **Stage2C** frozen training function `train_one_model`. No optimization criterion, training objective, or projection rule has been changed. Full historical outputs require access to authorized observations and checkpoint/history comparisons, which are not included.

## Frozen model definitions

- MS-TCN(RS): 239,169 learnable parameters, independent dynamic-only network.
- Shared-FiLM: 281,729 learnable parameters, trained with gate `1.0` on source records.
- Raw FiLM: inference from Shared-FiLM checkpoint without coefficient projection.
- PS-FiLM-TCN: inference from the **same** Shared-FiLM checkpoint, projecting coefficients onto source-derived channelwise parameter ranges. **Never train a separate PS network.**

`training.train_one_model` expects a `gate` vector as the FiLM model's third argument. The pre-existing public reference `SharedFiLMTCN` accepts `envelopes` as its third argument for inference. `TrainingCompatibleSharedFiLM` in `formal_training_models.py` explicitly reconciles the signatures. Its state dictionary keys, tensor shapes, seeded initialization, train/eval forward outputs, and gradients were independently checked against the historical Stage18B model classes using a local-only comparison script. It does not add parameters. Load its checkpoint state into the unmodified inference `SharedFiLMTCN` with `strict=True`.

The MS-TCN architecture is implemented separately and includes the second convolution ReLU from the original model. Do not replace it with a FiLM network with identity modulation.

## Training and loading

1. **Official schemes**: predefined three-source (Andunhe/Mixishui/Chuantanghe, per-sample weights 1.0/0.5/1.0) and regional six-fold LOBO (equal source-basin sampling mass); `replacement=True`, `num_samples=len(concatenated_training_dataset)`.
2. **Official seeds**: `42, 123, 2024, 2025, 3407`; development seeds are not substituted.
3. **Partition and scale**: only source-basin 70% training rows fit dynamic, target and static scalers. Validation is a subsequent source-only 15% split. The 24-record windows use historical stored-record indexing.
4. **Model initialization**: `set_seed(seed)` immediately before instantiating each model. The Stage2C trainer also invokes `set_seed(seed)` internally. Reset the current sampler's independent generator to the fold seed before each independent model fit.
5. **Frozen optimizer**: Adam, LR 5e-4, maximum 80 epochs, batch size 256, patience 15, ReduceLROnPlateau (factor 0.5, patience 5, minimum LR 1e-5), gradient clipping 1.0 and masked MSE. Even all-invalid batches undergo backward and optimizer-step on finite zero loss, as in the original trainer.
6. **Outputs**: best-model checkpoint, training log and metadata go into a caller-controlled local directory **outside the repository**. Run-directory overwrite is refused.
7. **Authorized mode**: `examples/run_authorized_training.py --check` confirms manifest and frozen choices without fitting; `--execute` explicitly trains **one** selected independent network. The repository does not supply or archive observed hydrological time series.
8. **Synthetic mode**: `examples/run_training_entry_smoke_test.py` creates and discards synthetic source records and temporary checkpoints; the one-epoch limit is only for the disposable smoke test, never an official hyperparameter change.

## Evidence levels

- Historical architecture comparison: formal class initialization, state keys, outputs and gradients match in local CPU tests for the same random seed. This is a model-implementation consistency check.
- Existing Stage2C training-loop comparison was completed in earlier stages; no change to its function.
- Stage2E-2 smoke: synthetic single fold + original training implementation + checkpoint output.
- Historical metric reproduction and authorized full reruns are **not** established by synthetic tests alone.

The module does not provide an approved pathway to overwrite historical experiment checkpoints. Keep actual data, trained checkpoints, private configuration paths, temporary outputs and local-only historical verification scripts outside the public repository.
