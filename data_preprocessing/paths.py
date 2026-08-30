"""Where the raw downloads and the processed archives live."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_ROOT as PROCESSED_ROOT  # noqa: E402

DATA_ROOT = Path(os.environ.get("CAFG_DATA_ROOT", PROCESSED_ROOT.parent))
RAW_ROOT = DATA_ROOT / "raw"
OUT_ROOT = DATA_ROOT / "processed"
