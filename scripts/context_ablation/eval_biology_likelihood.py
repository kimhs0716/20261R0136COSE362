from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fmo_context_ablation.context_features import LABELS, build_context
from fmo_context_ablation.data import DEFAULT_MERGED_PATH, json_dump, portable_path
from fmo_context_ablation.hamiltonian import matrix_to_h27, gauge_fix_encode
from fmo_context_ablation.nsf import build_flow, resolve_device
from fmo_context_ablation import simulator as sim


DATA_PATH = DEFAULT_MERGED_PATH
TRAINING_DIR = ROOT / "outputs" / "training"
OUTPUT_DIR = ROOT / "outputs" / "biology_likelihood"
DEFAULT_RUN_NAME = "nsf_h27_c5_seed0"
DEFAULT_REFERENCE_SAMPLES = 50000
DEFAULT_BATCH_SIZE = 4096
DEFAULT_SEED = 716


BIOLOGICAL_HAMILTONIANS = {
    "FMO_Adolphs_Renger": sim.H_FMO_CM.astype(np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate biological Hamiltonian likelihood rank under a trained NSF p(H27 | context)."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--reference-samples", type=int, default=DEFAULT_REFERENCE_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def load_model(run_name: str, device: torch.device):
    run_dir = TRAINING_DIR / run_name
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    flow = build_flow(
        feature_dim=int(payload["feature_dim"]),
        context_dim=int(payload["context_dim"]),
        device=device,
        transforms=payload["args"].get("transforms", 8),
        hidden=payload["args"].get("hidden") or 128,
        bins=payload["args"].get("bins", 8),
    )
    flow.load_state_dict(payload["state_dict"])
    flow.eval()
    return summary, payload, flow


def fmo_context_dict() -> dict[str, np.ndarray]:
    """FMO Hamiltonian에서 현재 context 설정이 요구하는 feature를 만든다."""
    labels = sim.simulate(sim.H_FMO_CM, lambda_reorg=35.0, return_traj=True)
    tlist, rho_t = labels.pop("_traj")

    diag = np.real(np.diagonal(rho_t, axis1=1, axis2=2)).astype(np.float64)
    target_times = np.asarray(_load_dataset_times(), dtype=np.float64)
    pop_t = np.stack([np.interp(target_times, tlist, diag[:, i]) for i in range(sim.DIM)], axis=1)

    sys = rho_t[:, : sim.N_SITE, : sim.N_SITE]
    tr_sys = np.real(np.trace(sys, axis1=1, axis2=2))
    rho_n = np.zeros_like(sys, dtype=np.complex128)
    valid = tr_sys > 1e-12
    rho_n[valid] = sys[valid] / tr_sys[valid, None, None]

    pops = np.real(np.diagonal(rho_n, axis1=1, axis2=2))
    ipr_series = np.sum(pops**2, axis=1)
    purity_series = np.real(np.trace(rho_n @ rho_n, axis1=1, axis2=2))
    cl1_series = np.sum(np.abs(rho_n) * (1.0 - np.eye(sim.N_SITE)[None]), axis=(1, 2))

    eigs = np.linalg.eigvalsh(sim.H_FMO_CM).astype(np.float32)
    out = {
        key: np.asarray([labels[key]], dtype=np.float32)
        for key in LABELS
    }
    out.update(
        {
            "eigs": eigs[None, :].astype(np.float32),
            "times": target_times.astype(np.float32),
            "pop_t": pop_t[None, :, :].astype(np.float32),
            "cl1_t": np.interp(target_times, tlist, cl1_series)[None, :].astype(np.float32),
            "purity_t": np.interp(target_times, tlist, purity_series)[None, :].astype(np.float32),
            "ipr_t": np.interp(target_times, tlist, ipr_series)[None, :].astype(np.float32),
        }
    )
    return out


def _load_dataset_times() -> np.ndarray:
    with np.load(DATA_PATH) as d:
        return np.asarray(d["times"], dtype=np.float32)


def log_prob_batches(flow, x_norm: np.ndarray, y_norm: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    out = []
    flow.eval()
    with torch.no_grad():
        y_base = torch.tensor(y_norm, dtype=torch.float32, device=device)
        for start in range(0, len(x_norm), batch_size):
            xb = torch.tensor(x_norm[start : start + batch_size], dtype=torch.float32, device=device)
            if len(y_base) == 1:
                yb = y_base.expand(len(xb), -1)
            else:
                yb = y_base[start : start + batch_size]
            out.append(flow(yb).log_prob(xb).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def percentile_rank(value: float, reference: np.ndarray) -> float:
    """높은 log-probability가 좋은 방향의 percentile. 95면 reference의 95%보다 그럴듯하다는 뜻이다."""
    return float((reference <= value).mean() * 100.0)


def summarize_reference(name: str, bio_logp: float, reference_logp: np.ndarray) -> dict:
    return {
        "reference": name,
        "n_reference": int(len(reference_logp)),
        "bio_logp": float(bio_logp),
        "reference_mean": float(np.mean(reference_logp)),
        "reference_median": float(np.median(reference_logp)),
        "reference_p05": float(np.percentile(reference_logp, 5)),
        "reference_p95": float(np.percentile(reference_logp, 95)),
        "bio_percentile_high_is_better": percentile_rank(bio_logp, reference_logp),
        "bio_top_fraction_high_is_better": float((reference_logp > bio_logp).mean()),
    }


def plot_histogram(out_path: Path, bio_name: str, bio_logp: float, references: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(8.5, 5.0))
    for label, values in references.items():
        plt.hist(values, bins=60, density=True, alpha=0.45, label=label)
    plt.axvline(bio_logp, color="black", linestyle="--", linewidth=2.0, label=f"{bio_name} logp={bio_logp:.2f}")
    plt.xlabel("log p(H | c_bio)")
    plt.ylabel("density")
    plt.title("Biological Hamiltonian likelihood rank")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)
    summary, payload, flow = load_model(args.run_name, device)
    context = payload["context"]
    stats = payload["stats"]

    out_dir = OUTPUT_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bio_context_source = fmo_context_dict()
    bio_y, context_names = build_context(bio_context_source, context)
    bio_y_norm = ((bio_y - stats["y_mu"]) / stats["y_sd"]).astype(np.float32)

    rows = []
    raw_outputs = {}
    with np.load(DATA_PATH) as d:
        x_all = gauge_fix_encode(np.asarray(d["H_params"], dtype=np.float32)).astype(np.float32)
        eta_all = np.asarray(d["eta"], dtype=np.float32)
        n_ref = min(int(args.reference_samples), len(x_all))
        ref_idx = rng.choice(len(x_all), size=n_ref, replace=False)
        high_eta_idx_all = np.where(eta_all >= 0.95)[0]
        n_hi = min(n_ref, len(high_eta_idx_all))
        high_eta_idx = rng.choice(high_eta_idx_all, size=n_hi, replace=False) if n_hi > 0 else np.array([], dtype=int)

        x_ref_norm = ((x_all[ref_idx] - stats["x_mu"]) / stats["x_sd"]).astype(np.float32)
        x_hi_norm = ((x_all[high_eta_idx] - stats["x_mu"]) / stats["x_sd"]).astype(np.float32)

    reference_logp = log_prob_batches(flow, x_ref_norm, bio_y_norm, device, args.batch_size)
    high_eta_logp = (
        log_prob_batches(flow, x_hi_norm, bio_y_norm, device, args.batch_size)
        if len(x_hi_norm)
        else np.array([], dtype=np.float64)
    )

    for bio_name, h in BIOLOGICAL_HAMILTONIANS.items():
        h27 = matrix_to_h27(h).astype(np.float32)[None, :]
        h27_norm = ((h27 - stats["x_mu"]) / stats["x_sd"]).astype(np.float32)
        bio_logp = float(log_prob_batches(flow, h27_norm, bio_y_norm, device, args.batch_size)[0])
        rows.append({"bio_hamiltonian": bio_name, **summarize_reference("dataset_random_subset_at_bio_condition", bio_logp, reference_logp)})
        if len(high_eta_logp):
            rows.append({"bio_hamiltonian": bio_name, **summarize_reference("dataset_high_eta_subset_at_bio_condition", bio_logp, high_eta_logp)})
        plot_histogram(
            out_dir / f"{bio_name}_likelihood_hist.png",
            bio_name,
            bio_logp,
            {
                "dataset random subset": reference_logp,
                "dataset high-eta subset": high_eta_logp,
            }
            if len(high_eta_logp)
            else {"dataset random subset": reference_logp},
        )
        raw_outputs[f"{bio_name}_logp"] = bio_logp

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "biology_likelihood_summary.csv", index=False)
    np.savez_compressed(
        out_dir / "biology_likelihood_raw.npz",
        reference_logp=reference_logp,
        high_eta_logp=high_eta_logp,
        **raw_outputs,
    )
    json_dump(
        out_dir / "biology_likelihood_manifest.json",
        {
            "run_name": args.run_name,
            "context": context,
            "context_dim": int(payload["context_dim"]),
            "context_names": context_names,
            "data_path": portable_path(DATA_PATH),
            "reference_samples": int(len(reference_logp)),
            "high_eta_reference_samples": int(len(high_eta_logp)),
            "high_eta_threshold": 0.95,
            "seed": int(args.seed),
            "device": str(device),
            "training_best_epoch": summary.get("best_epoch"),
            "training_best_val": summary.get("best_val"),
            "interpretation": "Higher log-probability is better. Percentile is computed within dataset Hamiltonians evaluated at the biological condition c_bio.",
        },
    )
    print(result.to_string(index=False))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()

