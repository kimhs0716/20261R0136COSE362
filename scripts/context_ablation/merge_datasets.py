from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fmo_context_ablation.data import DEFAULT_MERGED_PATH, portable_path, sample_count


OUTPUT_PATH = DEFAULT_MERGED_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Input .npz dataset files to merge.")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    sources = [Path(p) for p in args.inputs]
    if not sources:
        raise FileNotFoundError("No source datasets provided.")
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source datasets: {missing}")

    payloads = []
    counts = []
    common_keys: set[str] | None = None
    for path in sources:
        d = np.load(path)
        n = sample_count(d)
        payloads.append((path, d, n))
        counts.append(n)
        keys = set(d.files)
        common_keys = keys if common_keys is None else common_keys.intersection(keys)

    assert common_keys is not None
    merge_keys = []
    skipped = {}
    for key in sorted(common_keys):
        ok = True
        shapes = []
        for path, d, n in payloads:
            arr = d[key]
            shapes.append(tuple(arr.shape))
            if arr.shape == () or arr.shape[0] != n:
                ok = False
        if ok:
            merge_keys.append(key)
        else:
            skipped[key] = shapes

    out = {}
    for key in merge_keys:
        out[key] = np.concatenate([d[key] for _, d, _ in payloads], axis=0)

    constant_keys = []
    for key in sorted(common_keys.difference(merge_keys)):
        arrays = [d[key] for _, d, _ in payloads]
        first = arrays[0]
        if all(arr.shape == first.shape and np.array_equal(arr, first) for arr in arrays[1:]):
            out[key] = first
            constant_keys.append(key)

    source_file_id = []
    source_row = []
    source_files = []
    for file_id, (path, _d, n) in enumerate(payloads):
        source_files.append(portable_path(path))
        source_file_id.append(np.full(n, file_id, dtype=np.int32))
        source_row.append(np.arange(n, dtype=np.int32))
    out["source_file_id"] = np.concatenate(source_file_id)
    out["source_row"] = np.concatenate(source_row)
    out["source_files"] = np.asarray(source_files)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)

    manifest = {
        "output": portable_path(args.out),
        "n_total": int(sum(counts)),
        "sources": [{"path": portable_path(path), "n": int(n)} for path, _d, n in payloads],
        "merged_keys": merge_keys,
        "constant_keys": constant_keys,
        "skipped_common_keys": skipped,
        "note": "Only common arrays with sample axis length N are concatenated.",
    }
    manifest_path = args.out.with_name(args.out.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for _path, d, _n in payloads:
        d.close()

    print(f"saved: {args.out}")
    print(f"n_total: {sum(counts)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

