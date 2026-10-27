from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psfilm_reference import smoke_test

if __name__ == "__main__":
    smoke_test()
