# Frozen Stage16/Stage18B training core

`src/training.py` provides a cleaned data-independent implementation of the training code used in the formal experiments. It was numerically compared with the exact Stage16 and Stage18B functions in the private Stage2C audit package.

## Scope

The function accepts **prepared source-only** training and validation PyTorch DataLoaders, a model created with the frozen architecture, an already-selected seed, and an output directory. It **does not** choose source basins, fit scalers, build datasets, define an experiment fold, or run PS-FiLM coefficient projection. Those steps need separate tested entry points.

## Default settings

- Adam, learning rate `0.0005`.
- At most 80 epochs, batch size 256 set by the upstream DataLoader.
- ReduceLROnPlateau on validation loss, `mode='min'`, `factor=0.5`, `patience=5`, `min_lr=0.00001`.
- Gradient-norm clipping at `1.0`.
- Early stopping after 15 epochs without a strict decrease in mean validation loss.
- Every batch with finite loss contributes equally to the epoch mean; batches are not reweighted by their number of valid targets.
- Mask value `0` means a valid target. All-invalid batches yield differentiable zero loss and continue through backward and optimizer step, consistent with the formal script.
- Best model weights are restored after training and saved to `best_model.pth`; `train_log.csv` is written alongside.
- The training function resets random seeds **after the model has been created**, matching the formal training function. The calling experiment must maintain the original model-construction seed/order if reproducing training from scratch.
- The formal code enables `torch.backends.cudnn.benchmark = True` and does not claim strict bitwise GPU determinism.

## Minimal smoke test

```
python examples/run_training_smoke_test.py
```

## Data flow

The expected batch contains seven values in this exact order:

`dynamic_x, static_features, transformed_target, missing_target_mask, basin_id, previous_discharge, timestamp`

Two model signatures are supported: `model(dynamic_x)` for `MS-TCN(RS)` and `model(dynamic_x, static_features, gate_ones)` for the trained unconstrained FiLM-TCN. **PS-FiLM is applied at inference to the same frozen FiLM weights.**

## Audit boundary

Passing the training smoke test and Stage2C parity audit confirms the tested training loop semantics with synthetic examples. It does not reconstruct the paper's held-out NSE values without the restricted hourly observations and model checkpoint files.
