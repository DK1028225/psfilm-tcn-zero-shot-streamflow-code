# Manuscript-to-code map

| Manuscript item | Public implementation |
|---|---|
| Section 2.3, runoff transformation | `runoff_transform` in `src/psfilm_reference.py` |
| Section 2.3, one-record antecedent state | `make_lagged_state` |
| Section 2.3, 24-record sequence indexing | `build_record_position_sequences` |
| Sections 2.3/2.6, source-only standardization | `fit_source_scalers` |
| Section 2.4, four-block TCN + block-specific FiLM | `SharedFiLMTCN`, `FiLMBlock` |
| Section 2.5, source-derived coefficient intervals | `build_source_envelopes` |
| Section 2.5, PS-FiLM interval projection | `project_coefficients` and `FiLMBlock.forward(..., bounds=...)` |
| Supplementary Method S3, joint convex-hull projection | `project_to_convex_hull` |
| Section 2.6, basin-balanced LOBO sampling | `build_lobo_weighted_sampler` |
| Section 2.6, predefined 1.0/0.5/1.0 sampling | `build_predefined_weighted_sampler` |
| Section 2.7 and Supplementary Method S5, NSE/KGE/RMSE/MAE/PBIAS | `metrics` |
| Section 2.7, across-seed sample SD | `summarize_seed_values` |
| Numerical settings and official/development seeds | `docs/REPRODUCIBILITY_AND_SETUP.md`, `docs/STAGE2E2_TRAINING_PROTOCOL.md`, `src/training.py` (consult manuscript for complete fixed settings) |

The public reference implementation is intentionally data-agnostic. Full manuscript-number reproduction requires the restricted Dongjiang observations and the original trained checkpoints.
