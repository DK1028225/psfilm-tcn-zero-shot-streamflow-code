# Stage2E-3: evidence levels and historical comparison checklist

## Level 1 — code syntax and synthetic tests

- Python compilation succeeds for all four new `.py` files.
- Existing repository tests plus Stage2E-3 tests pass.
- Synthetic unit tests verify independent MS vs FiLM weights, Raw/PS shared state, target observation/time/antecedent Q alignment, valid-target mask, source-only static scaler, checkpoint model type and seed.
- One-command synthetic smoke passes. No historical discharge has been accessed.

## Level 2 — formal implementation consistency

The new adapter is assembled from Stage18-compatible, separately validated Stage2D/Stage2E-1/Stage2E-2 modules. It does not provide additional evidence about the historical checkpoint/hardware equivalence unless verified on the original checkpoint and authorized inputs.

## Level 3 — real historical checkpoint replay (pending authorized local execution)

For one officially frozen seed/fold (recommended Zengjiang LOBO seed42):

1. Verify exact legacy input files, original row order and source basin role.
2. Check the two trusted checkpoint paths and the official seed. Confirm both are from the same fold.
3. Run the opt-in CLI to compute aligned MS, Raw, PS and Persistence summary metrics. Keep JSON outside Git.
4. Compare with the original Stage18 basin–seed row: identical target counts and dates, tolerance-based numerical agreement of NSE/KGE/RMSE/MAE/PBIAS and no new training.
5. Record hardware, PyTorch/CUDA/cuDNN versions, absolute deviations and data/checkpoint fingerprints locally without publishing data or private paths.

## Level 4 — full end-to-end experiment reproduction (not claimed)

Stage2E-3 does not repeat Stage18 official training and cannot independently reconstruct manuscript results without permission to access observations and checkpoints. The future Stage2E-4 batch dispatcher must preserve the frozen official protocol, not initiate a new parameter search or extra training merely to raise the evidence level.
