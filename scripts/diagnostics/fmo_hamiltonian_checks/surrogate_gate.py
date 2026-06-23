"""Path-1 de-risking gate — train a fast H -> trajectory surrogate and check its accuracy
WHERE the generative mismatch lives (late times t=20,50ps; states site1, trap, loss).

If the surrogate predicts those well (high R²), it can provide a useful dynamics-fidelity signal
=> Path 1 (PINN penalty) is viable. If it can't predict the late-time/site1 dynamics, the penalty
would be uninformative there => Path 1 is risky, prefer Path 2 (consolidate).

Input: 27-d gauge-fixed H. Target: pop_t (7 times x 9 states = 63), from stored data (no sim).
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.cnf import gauge_fix_encode

STATE_NAMES = [f"site{i+1}" for i in range(7)] + ["trap(eta)", "loss"]


class MLP(nn.Module):
    def __init__(self, d_in=27, d_out=63, h=384, layers=4):
        super().__init__()
        net = [nn.Linear(d_in, h), nn.SiLU()]
        for _ in range(layers - 1):
            net += [nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU()]
        net += [nn.Linear(h, d_out)]
        self.net = nn.Sequential(*net)
    def forward(self, x): return self.net(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xs, Ys, times = [], [], None
    for p in ("data/dataset_full.npz", "data/dataset_full_v2.npz"):
        d = np.load(p)
        Xs.append(gauge_fix_encode(d["H_params"]).astype(np.float32))
        Ys.append(d["pop_t"].reshape(len(d["pop_t"]), 63).astype(np.float32))  # 7 times x 9 states
        times = d["times"]
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    TIMES = np.asarray(times, float)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[ok], Y[ok]
    xmu, xsd = X.mean(0), X.std(0) + 1e-8
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-8
    Xn = (X - xmu) / xsd; Yn = (Y - ymu) / ysd
    print(f"N={len(X)}, in=27, out=63 (7 times x 9 states)", flush=True)

    ds = TensorDataset(torch.tensor(Xn), torch.tensor(Yn))
    g = torch.Generator().manual_seed(0)
    ntr = int(len(ds) * 0.9)
    tr, te = random_split(ds, [ntr, len(ds) - ntr], generator=g)
    tl = DataLoader(tr, batch_size=2048, shuffle=True, drop_last=True)

    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    print(f"surrogate params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M; training...", flush=True)
    for ep in range(150):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            loss = ((model(xb) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 30 == 0:
            print(f"  ep {ep+1}/150  train MSE {loss.item():.4f}", flush=True)

    # test R² per output feature
    model.eval()
    Xte = torch.tensor(Xn[te.indices]).to(device)
    Yte = Yn[te.indices]
    with torch.no_grad():
        pred = model(Xte).cpu().numpy()
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    R2 = r2.reshape(7, 9)                     # (time, state)

    print(f"\n=== surrogate test R² (overall {r2.mean():.3f}) ===", flush=True)
    print("\n[per TIME] R² (avg over 9 states):", flush=True)
    for ti, t in enumerate(TIMES):
        flag = "  ← LATE (mismatch)" if t >= 20 else ""
        print(f"   t={t:4.1f}ps : R² {R2[ti].mean():.3f}{flag}", flush=True)
    print("\n[per STATE] R² (avg over 7 times):", flush=True)
    order = np.argsort(R2.mean(0))
    for s in order:
        flag = "  ← mismatch-heavy" if STATE_NAMES[s] in ("site1", "trap(eta)", "loss") else ""
        print(f"   {STATE_NAMES[s]:11s}: R² {R2[:,s].mean():.3f}{flag}", flush=True)

    # the decisive cells: late times x mismatch states
    key = [(ti, s) for ti in range(7) if TIMES[ti] >= 20 for s in (0, 7, 8)]  # site1,trap,loss @ t>=20
    key_r2 = np.mean([R2[ti, s] for ti, s in key])
    print(f"\n[GATE] late(t>=20) x (site1/trap/loss) mean R² = {key_r2:.3f}", flush=True)
    if key_r2 > 0.8:
        verdict = "→ Path 1 VIABLE: surrogate captures the mismatch region well. PINN signal usable."
    elif key_r2 > 0.6:
        verdict = "→ Path 1 부분적: 후반 신호 쓸 만하나 noisy. 신중히 진행 가능."
    else:
        verdict = "→ Path 1 RISKY: surrogate가 후반-mismatch를 못 잡음. Path 2(정리) 권장."
    print(verdict, flush=True)

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/surrogate_gate.json").write_text(json.dumps(
        {"overall_r2": float(r2.mean()), "r2_time_state": R2.tolist(),
         "times": TIMES.tolist(), "states": STATE_NAMES,
         "gate_late_mismatch_r2": float(key_r2)}, indent=2), encoding="utf-8")
    print("saved: outputs/metrics/surrogate_gate.json", flush=True)


if __name__ == "__main__":
    main()
