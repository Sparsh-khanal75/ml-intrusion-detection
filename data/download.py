#!/usr/bin/env python3
"""Download the NSL-KDD benchmark (Tavallaee et al., 2009) into ./data."""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
EXPECTED_ROWS = {"KDDTrain+.txt": 125_973, "KDDTest+.txt": 22_544}

for name, rows in EXPECTED_ROWS.items():
    dest = Path(__file__).parent / name
    if not dest.exists():
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BASE + name, dest)
    n = sum(1 for _ in open(dest))
    assert n == rows, f"{name}: expected {rows} rows, got {n}"
    print(f"{name}: {n:,} rows OK")
