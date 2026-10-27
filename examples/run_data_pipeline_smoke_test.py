"""Run synthetic data tests, no private records or training."""
import subprocess
import sys
from pathlib import Path
repo = Path(__file__).resolve().parents[1]
cmd = [sys.executable,'-m','unittest','discover','-s','tests','-p','test_data_pipeline.py','-v']
subprocess.run(cmd,cwd=repo,check=True)
print('[STATUS] STAGE2E1_SYNTHETIC_SMOKE_PASS = True')
