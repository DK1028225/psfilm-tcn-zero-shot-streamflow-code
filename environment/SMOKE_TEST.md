# Smoke-test record

Tested on 2026-10-17 in the manuscript development environment; see `requirements-tested.txt` for the recorded version pins.

Command:

```bash
python examples/run_smoke_test.py
```

Observed output:

```text
Smoke test PASS
Example metrics: {'NSE': 0.9441882516428092, 'KGE': 0.9674975024751507, 'RMSE': 0.5440671619811392, 'MAE': 0.41405002420317605, 'PBIAS': -0.2599665189841524}
```

The smoke test uses synthetic data and verifies code execution and numerical definitions. It is not intended to reproduce the manuscript's Dongjiang performance values.
