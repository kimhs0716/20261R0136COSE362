from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .context_features import LABELS, build_context
from .hamiltonian import gauge_fix_encode


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_MERGED_PATH = REPO_ROOT / "data" / "merged_h27_140k.npz"


def portable_path(path: str | Path) -> str:
    """Return a stable project-relative path for saved metadata."""
    p = Path(path).resolve()
    for root in (REPO_ROOT.resolve(), PROJECT_ROOT.resolve()):
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            pass
    return p.as_posix()


def sample_count(d: np.lib.npyio.NpzFile) -> int:
    if "H_params" in d.files:
        return int(len(d["H_params"]))
    for key in LABELS:
        if key in d.files:
            return int(len(d[key]))
    raise KeyError("Cannot infer sample count: missing H_params and labels")


def load_h27_and_context(path: str | Path, context: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path) as d:
        x = gauge_fix_encode(np.asarray(d["H_params"], dtype=np.float32)).astype(np.float32)
        y, names = build_context(d, context)
    if len(x) != len(y):
        raise ValueError((x.shape, y.shape))
    return x, y, names


def normalize_train_val(
    x: np.ndarray,
    y: np.ndarray,
    *,
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    n_val = int(round(len(idx) * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    stats = {
        "x_mu": x_train.mean(axis=0).astype(np.float32),
        "x_sd": (x_train.std(axis=0) + 1e-8).astype(np.float32),
        "y_mu": y_train.mean(axis=0).astype(np.float32),
        "y_sd": (y_train.std(axis=0) + 1e-8).astype(np.float32),
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
    }
    x_norm = ((x - stats["x_mu"]) / stats["x_sd"]).astype(np.float32)
    y_norm = ((y - stats["y_mu"]) / stats["y_sd"]).astype(np.float32)
    return x_norm, y_norm, train_idx, val_idx, stats


def json_dump(path: Path, payload: dict) -> None:
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2, ensure_ascii=False), encoding="utf-8")
