"""Local driver — Conditional Diffusion (DDPM/DDIM) 학습 + η=0.95 샘플링.

train_diffusion.ipynb 의 핵심 흐름을 Windows + 로컬 GPU 에서 headless 로 재현.
- 데이터: data/dataset_full.npz + dataset_full_v2.npz (총 60k)
- 모델: standard capacity (hidden=384, layers=8 ≈ 2.5M) — docs 가 권장하는 multimodal capacity
        (노트북 본문의 192×6 ≈ 0.2M 는 docs/09 가 지적한 under-capacity 변형)
- 출력: outputs/checkpoints/diffusion_v1.pt, outputs/metrics/, outputs/figures/

사용:  python run_diffusion.py [--epochs N] [--hidden H] [--layers L]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Windows 콘솔(cp949) 에서 한글/≈/η 출력 위해 UTF-8 강제
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 프로젝트 root 보정 + import
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cnf import load_data, normalize, denormalize_H, H27_to_matrix
from src.diffusion import build_diffusion, train_diffusion, sample_diffusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n_sample", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, torch: {torch.__version__}", flush=True)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    for d in ("outputs/checkpoints", "outputs/metrics", "outputs/figures"):
        os.makedirs(d, exist_ok=True)

    # ---- 1. 데이터 ----
    X, Y = load_data(["data/dataset_full.npz", "data/dataset_full_v2.npz"],
                     include_a1=True, include_a4=True)
    Xn, Yn, stats = normalize(X, Y)
    print(f"N={len(X)}, X {X.shape}, Y {Y.shape}", flush=True)

    # ---- 2. 모델 ----
    model, sched = build_diffusion(device, feature_dim=27, context_dim=Y.shape[1],
                                   hidden=args.hidden, layers=args.layers,
                                   T=1000, schedule="cosine")
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Diffusion params: {n_param/1e6:.2f}M (hidden={args.hidden}, layers={args.layers})",
          flush=True)

    # ---- 3. 학습 ----
    hist = train_diffusion(model, sched, Xn, Yn, device=device,
                           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                           val_frac=0.1, log_every=10,
                           num_workers=0, pin_memory=(device == "cuda"))

    # ---- 4. 저장 ----
    ckpt = "outputs/checkpoints/diffusion_v1.pt"
    torch.save({"state_dict": model.state_dict(), "sched": sched, "stats": stats,
                "hist": hist,
                "config": {"feature_dim": 27, "context_dim": int(Y.shape[1]),
                           "hidden": args.hidden, "layers": args.layers,
                           "T": 1000, "schedule": "cosine"}}, ckpt)
    with open("outputs/metrics/diffusion_v1_history.json", "w") as f:
        json.dump({"train": hist["train"], "val": hist["val"]}, f, indent=2)
    print(f"saved: {ckpt}  (final val MSE {hist['val'][-1]:.4f})", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(hist["train"], label="train")
        plt.plot(hist["val"], label="val")
        plt.xlabel("epoch"); plt.ylabel("eps prediction MSE")
        plt.title(f"Diffusion loss — final val {hist['val'][-1]:.4f}")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("outputs/figures/diffusion_loss.png", dpi=120)
        print("saved: outputs/figures/diffusion_loss.png", flush=True)
    except Exception as e:
        print(f"(matplotlib skip: {e})", flush=True)

    # ---- 5. η=0.95 조건 샘플링 ----
    mask = np.abs(Y[:, 0] - 0.95) < 0.05
    subset = np.where(mask)[0]
    print(f"\nη≈0.95 subset N = {len(subset)}", flush=True)
    rng = np.random.default_rng(0)
    n_cond = min(args.n_sample // 2, len(subset))
    chosen = rng.choice(subset, size=n_cond, replace=False)
    samples = []
    for idx in chosen:
        s = sample_diffusion(model, sched, Yn[idx], n=2, device=device, n_steps=50, eta=0.0)
        samples.append(s)
    S = np.concatenate(samples, axis=0)           # (≈n_sample, 27) normalized
    S_real = denormalize_H(S, stats)              # cm^-1
    np.savez_compressed("outputs/metrics/diffusion_samples_eta095.npz",
                        S_norm=S.astype(np.float32), S_real=S_real.astype(np.float32),
                        cond_idx=chosen)
    print(f"sampled {len(S)} H @ η≈0.95 → outputs/metrics/diffusion_samples_eta095.npz",
          flush=True)

    # ---- 6. 샘플 sanity (parameter space) ----
    # 생성 H 의 비대각 coupling 크기 + eigenvalue 범위를 학습 데이터와 비교
    couplings = []
    eigs = []
    for h27 in S_real:
        H = H27_to_matrix(h27)                    # 7×7
        off = H[np.triu_indices(7, k=1)]
        couplings.append(np.abs(off))
        eigs.append(np.linalg.eigvalsh(H))
    couplings = np.concatenate(couplings)
    eigs = np.array(eigs)

    # 학습 데이터의 같은 통계 (η≈0.95 subset). X 는 load_data 가 준 gauge-fix 27-d (cm^-1).
    data_off, data_eigs = [], []
    for h27 in X[subset]:
        H = H27_to_matrix(h27)
        data_off.append(np.abs(H[np.triu_indices(7, k=1)]))
        data_eigs.append(np.linalg.eigvalsh(H))
    data_off = np.concatenate(data_off)
    data_eigs = np.array(data_eigs)

    print("\n=== sample sanity (η≈0.95) ===", flush=True)
    print(f"  |coupling|  생성:  median {np.median(couplings):6.1f}  "
          f"p95 {np.percentile(couplings,95):6.1f}  max {couplings.max():6.1f} cm^-1", flush=True)
    print(f"  |coupling|  학습:  median {np.median(data_off):6.1f}  "
          f"p95 {np.percentile(data_off,95):6.1f}  max {data_off.max():6.1f} cm^-1", flush=True)
    print(f"  eig range   생성:  [{eigs.min():7.1f}, {eigs.max():7.1f}] cm^-1  "
          f"(mean spread {np.ptp(eigs,axis=1).mean():.1f})", flush=True)
    print(f"  eig range   학습:  [{data_eigs.min():7.1f}, {data_eigs.max():7.1f}] cm^-1  "
          f"(mean spread {np.ptp(data_eigs,axis=1).mean():.1f})", flush=True)
    # 다양성: 생성 샘플들이 한 점으로 collapse 안 했는지 (정규화 좌표 std)
    print(f"  sample std (norm coords) per-dim mean: {S.std(0).mean():.3f}  "
          f"(≈1.0 → prior 폭만큼 다양, ≈0 → parameter-space collapse)", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
