"""Synthetic Stage2E-3 four-model replay; no authorized observations loaded."""
from pathlib import Path
import sys
import tempfile
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    # Execute exactly the new Stage2E-3 contract tests; these write checkpoint
    # fixtures only in TemporaryDirectory, never under GitHub.
    result = subprocess.run(
        [sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT/'tests'),
         '-p', 'test_formal_fold_replay.py', '-v'],
        cwd=str(ROOT),check=True,
    )
    print('[STATUS] STAGE2E3_SYNTHETIC_FOUR_MODEL_PASS = True')
    print('Evidence: synthetic checkpoint replay only; no official metrics reproduced.')


if __name__ == '__main__':
    main()
