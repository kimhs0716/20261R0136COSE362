from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fmo_context_ablation.context_features import build_context, list_contexts
from fmo_context_ablation.data import DEFAULT_MERGED_PATH
from fmo_context_ablation.hamiltonian import gauge_fix_encode


DATA_PATH = DEFAULT_MERGED_PATH


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run scripts/merge_datasets.py first: {DATA_PATH}")

    with np.load(DATA_PATH) as d:
        x = gauge_fix_encode(d["H_params"])
        print(f"data: {DATA_PATH}")
        print(f"H27: {x.shape}")
        print("")
        for name in list_contexts():
            y, names = build_context(d, name)
            print(f"{name}: {y.shape[1]}D")
            print("  " + ", ".join(names))


if __name__ == "__main__":
    main()

