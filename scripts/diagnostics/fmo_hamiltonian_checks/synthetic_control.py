"""⑥ Synthetic positive control — "데이터가 unimodal" vs "모델이 multimodal 을 못 냄" 분리.

정답을 아는 *conditional multimodal* 데이터를 만들어 같은 Diffusion 으로 학습:
  - 조건 c ∈ R^C 마다 H | c 는 2-component mixture (50:50):
      μ_A(c) = m(c) + ĝ·(SEP/2),  μ_B(c) = m(c) − ĝ·(SEP/2),  H = μ_{A|B} + σ·noise
      m(c) = W c  (조건이 두 mode 를 *함께* 이동시킴; A/B 선택은 c 와 무관 = 진짜 conditional 다봉)
  - 분리도 SEP (σ 단위) 를 스윕 → 모델·탐지의 임계값.

평가 (한 조건 c 고정):
  - 모델 샘플 N개 → 알려진 두 center 중 가까운 쪽으로 분류 → recovered minority fraction
      ~0.5 = 두 mode 완전 복원 (collapse 아님),  ~0 = collapse.
  - 같은 dip test 파이프라인을 ĝ 투영에 적용 → 우리 탐지가 이 SEP 를 잡나.
  - 학습 데이터에도 같은 측정 → ground-truth 가 실제 다봉인지 + 탐지 sanity.

결론 논리:
  - diffusion 이 (실데이터-급 SEP 에서) 50:50 복원 → 모델은 multimodal 가능 → 실데이터의
    unimodal 출력은 *데이터가 unimodal* 이라서다. ("collapse" 변명 불가)
  - diffusion 이 collapse → 모델 한계 → 실데이터 결론 보류.

사용:  python synthetic_control.py [--seps 2,4,8] [--epochs 150]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diptest
from src.diffusion import build_diffusion, train_diffusion, sample_diffusion

F, C = 27, 4                                  # feature dim (실데이터와 동일), 조건 dim


def gen_synthetic(N, sep, sigma=1.0, seed=0):
    """conditional 2-mode mixture. 반환: X(N,F), Y(N,C), 그리고 (W, ghat, sep) — 평가용."""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.5, size=(F, C))            # m(c) = W c
    ghat = rng.normal(size=F); ghat /= np.linalg.norm(ghat)   # 분리 방향 (단위)
    c = rng.normal(size=(N, C)).astype(np.float32)
    m = c @ W.T                                       # (N, F)
    z = rng.integers(0, 2, size=N) * 2 - 1            # ±1 (mode A/B), c 와 독립
    centers = m + np.outer(z, ghat) * (sep / 2.0)
    X = (centers + sigma * rng.normal(size=(N, F))).astype(np.float32)
    return X, c, dict(W=W, ghat=ghat.astype(np.float32), sep=sep, z=z)


def dip_p(x):
    x = np.asarray(x, float)
    if x.std() < 1e-9 or len(np.unique(x)) < 4:
        return 1.0
    return float(diptest.diptest(x)[1])


def eval_modes(samples_raw, c_vec, info):
    """ĝ 투영 → minority fraction + dip p. center 기준 분류."""
    m = c_vec @ info["W"].T                            # (F,)
    proj = (samples_raw - m) @ info["ghat"]            # (N,) ; 진짜 두 mode = ±sep/2
    frac_pos = float((proj > 0).mean())
    minority = min(frac_pos, 1 - frac_pos)
    return minority, dip_p(proj), proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seps", default="2,4,8")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--n_cond", type=int, default=12)
    ap.add_argument("--n_per", type=int, default=400)
    args = ap.parse_args()
    seps = [float(s) for s in args.seps.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  | F={F} C={C} N={args.n} epochs={args.epochs}\n", flush=True)
    print("=" * 72, flush=True)
    print("⑥ Synthetic positive control — 정답 아는 conditional 2-mode 데이터", flush=True)
    print("=" * 72, flush=True)

    all_res = []
    for sep in seps:
        X, Y, info = gen_synthetic(args.n, sep, seed=0)
        # 정규화 (실파이프라인과 동일 방식)
        x_mu, x_sd = X.mean(0), X.std(0) + 1e-8
        y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-8
        Xn = (X - x_mu) / x_sd
        Yn = (Y - y_mu) / y_sd

        model, sched = build_diffusion(device, F, C, hidden=384, layers=8, T=1000)
        print(f"\n[SEP={sep}σ]  학습 ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)...",
              flush=True)
        train_diffusion(model, sched, Xn, Yn, device=device, epochs=args.epochs,
                        batch_size=2048, lr=2e-3, log_every=max(1, args.epochs // 3),
                        num_workers=0, pin_memory=(device == "cuda"))

        # 평가: n_cond 개 조건에서 모델 샘플 + 학습이웃, 두 측정
        rng = np.random.default_rng(123)
        cond_idx = rng.choice(len(X), args.n_cond, replace=False)
        mdl_minor, mdl_dipp, tr_minor, tr_dipp = [], [], [], []
        for ci in cond_idx:
            c_vec = Y[ci]
            # 모델 샘플
            Sn = sample_diffusion(model, sched, Yn[ci], n=args.n_per, device=device,
                                  n_steps=50, eta=0.0)
            S_raw = Sn * x_sd + x_mu
            mn, dp, _ = eval_modes(S_raw, c_vec, info)
            mdl_minor.append(mn); mdl_dipp.append(dp)
            # 학습 데이터 이웃 (조건공간 최근접 n_per) — ground truth 가 실제 다봉인지
            d = np.linalg.norm(Y - c_vec, axis=1)
            knn = np.argsort(d)[:args.n_per]
            tn, tdp, _ = eval_modes(X[knn], c_vec, info)
            tr_minor.append(tn); tr_dipp.append(tdp)

        mdl_minor = np.array(mdl_minor); tr_minor = np.array(tr_minor)
        mdl_sig = np.mean(np.array(mdl_dipp) < 0.05)
        tr_sig = np.mean(np.array(tr_dipp) < 0.05)
        print(f"  ── SEP={sep}σ 결과 (조건 {args.n_cond}개 평균) ──", flush=True)
        print(f"     학습데이터:  minority {tr_minor.mean():.2f}±{tr_minor.std():.2f}  "
              f"| dip 다봉검출 {100*tr_sig:.0f}%", flush=True)
        print(f"     모델 샘플:   minority {mdl_minor.mean():.2f}±{mdl_minor.std():.2f}  "
              f"| dip 다봉검출 {100*mdl_sig:.0f}%", flush=True)
        verdict = ("모델 두 mode 복원 (collapse 아님)" if mdl_minor.mean() > 0.30
                   else "모델 COLLAPSE (한 mode)" if mdl_minor.mean() < 0.10
                   else "부분 복원")
        print(f"     ▶ {verdict}", flush=True)
        all_res.append({"sep": sep, "train_minority": float(tr_minor.mean()),
                        "train_dip_detect": float(tr_sig),
                        "model_minority": float(mdl_minor.mean()),
                        "model_minority_std": float(mdl_minor.std()),
                        "model_dip_detect": float(mdl_sig), "verdict": verdict})

    print("\n" + "=" * 72, flush=True)
    print("종합 (minority≈0.5 = 두 mode 완전 복원, ≈0 = collapse):", flush=True)
    for r in all_res:
        print(f"   SEP={r['sep']:>4}σ:  학습 {r['train_minority']:.2f} → 모델 "
              f"{r['model_minority']:.2f}  ({r['verdict']})", flush=True)
    print("\n해석:", flush=True)
    print("  · 모델이 실데이터-급 SEP(≈4σ, 실데이터 PC1 spread 수준)에서 두 mode 복원 →", flush=True)
    print("    diffusion 은 conditional multimodal 가능 → 실데이터의 unimodal 출력은 *데이터가", flush=True)
    print("    unimodal* 이라서다. 'mode collapse 못 함' 변명 불가 → contribution 무효 확정.", flush=True)
    print("  · dip 검출률: 우리 탐지 파이프라인이 어느 SEP 부터 다봉을 잡는지 = 실데이터 null", flush=True)
    print("    결론의 민감도 보정.", flush=True)

    with open("outputs/metrics/synthetic_control_results.json", "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print("\n저장: outputs/metrics/synthetic_control_results.json\nDONE.", flush=True)


if __name__ == "__main__":
    main()
