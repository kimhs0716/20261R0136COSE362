"""Path-1 PoC — diffusion trained with a surrogate-based dynamics-fidelity penalty.

The diversity-collapse diagnostic showed the diffusion doesn't tightly satisfy its condition
(even the explicitly-conditioned site pops at t=1,5,10). Here we add a penalty: the generated H
(one-step x0), run through a frozen H->trajectory surrogate, must reproduce the condition's
target pop_t at t=1,5,10. The surrogate gate passed (R²~0.91 in the mismatch region), so the
signal is usable.

Key: the surrogate uses the SAME X normalization as the diffusion (stats x_mu/x_sd), so x0_pred
feeds directly into the surrogate with no renormalization.

Outputs: outputs/checkpoints/surrogate.pt, outputs/checkpoints/diffusion_pinn.pt
"""
import os, sys, argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.cnf import load_data, normalize
from src.diffusion import build_diffusion, make_beta_schedule

# 33-d condition layout: [5 labels, 7 eigs, 21 pop_t(t=1,5,10 x 7 sites)] -> pop dims 12:33
COND_POP = slice(12, 33)
# surrogate 63-d = 7 times x 9 states; condition times = indices [1,3,4]=t1,5,10, sites 0..6
SURR_IDX = [ti * 9 + s for ti in (1, 3, 4) for s in range(7)]   # 21 indices


class Surrogate(nn.Module):
    def __init__(self, d_in=27, d_out=63, h=384, layers=4):
        super().__init__()
        net = [nn.Linear(d_in, h), nn.SiLU()]
        for _ in range(layers - 1):
            net += [nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU()]
        net += [nn.Linear(h, d_out)]
        self.net = nn.Sequential(*net)
    def forward(self, x): return self.net(x)


def train_surrogate(Xn, POP, device, epochs=150):
    """Xn already normalized by diffusion stats; predict standardized pop_t (63-d)."""
    ymu, ysd = POP.mean(0), POP.std(0) + 1e-8
    Yn = (POP - ymu) / ysd
    ds = TensorDataset(torch.tensor(Xn), torch.tensor(Yn.astype(np.float32)))
    tl = DataLoader(ds, batch_size=2048, shuffle=True, drop_last=True)
    m = Surrogate().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(epochs):
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            loss = ((m(xb) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    print(f"  surrogate trained (final MSE {loss.item():.4f})", flush=True)
    return m, ymu.astype(np.float32), ysd.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lam", type=float, default=0.1, help="fidelity penalty weight")
    ap.add_argument("--surr_epochs", type=int, default=150)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y = load_data(["data/dataset_full.npz", "data/dataset_full_v2.npz"])
    Xn, Yn, stats = normalize(X, Y)
    POP = np.concatenate([np.load(p)["pop_t"].reshape(-1, 63) for p in
                          ("data/dataset_full.npz", "data/dataset_full_v2.npz")]).astype(np.float32)
    print(f"N={len(X)} device={device}", flush=True)

    # --- surrogate (same X normalization as diffusion) ---
    surr, surr_ymu, surr_ysd = train_surrogate(Xn, POP, device, args.surr_epochs)
    torch.save({"state_dict": surr.state_dict(), "ymu": surr_ymu, "ysd": surr_ysd},
               "outputs/checkpoints/surrogate.pt")

    # tensors for the penalty
    y_mu = torch.tensor(stats["y_mu"][COND_POP], device=device)
    y_sd = torch.tensor(stats["y_sd"][COND_POP], device=device)
    s_ymu = torch.tensor(surr_ymu[SURR_IDX], device=device)
    s_ysd = torch.tensor(surr_ysd[SURR_IDX], device=device)
    sidx = torch.tensor(SURR_IDX, device=device)

    # --- diffusion + fidelity penalty ---
    model, sched = build_diffusion(device, 27, Y.shape[1], hidden=384, layers=8, T=1000)
    alpha_bars = sched["alpha_bars"]; T = sched["T"]
    ds = TensorDataset(torch.tensor(Xn), torch.tensor(Yn))
    g = torch.Generator().manual_seed(0)
    tr, va = random_split(ds, [len(ds) - 6000, 6000], generator=g)
    tl = DataLoader(tr, batch_size=2048, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    print(f"diffusion+PINN training (lam={args.lam})...", flush=True)
    hist = {"train": [], "mse": [], "fid": []}
    for ep in range(args.epochs):
        model.train(); tot_mse = tot_fid = n = 0
        for x, c in tl:
            x, c = x.to(device), c.to(device); B = x.size(0)
            t_idx = torch.randint(0, T, (B,), device=device)
            a_bar = alpha_bars[t_idx].view(-1, 1)
            eps = torch.randn_like(x)
            x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * eps
            eps_pred = model(x_t, t_idx.float() / T, c)
            mse = ((eps_pred - eps) ** 2).mean()
            # one-step x0 estimate -> surrogate -> condition pop_t (t=1,5,10)
            x0 = (x_t - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
            pred = surr(x0)[:, sidx]                                   # surrogate-normalized
            tgt_raw = c[:, COND_POP] * y_sd + y_mu                     # condition pop -> raw
            tgt = (tgt_raw - s_ymu) / s_ysd                           # -> surrogate-normalized
            fid = ((pred - tgt) ** 2).mean()
            loss = mse + args.lam * fid
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot_mse += mse.item() * B; tot_fid += fid.item() * B; n += B
        hist["mse"].append(tot_mse / n); hist["fid"].append(tot_fid / n)
        hist["train"].append(tot_mse / n + args.lam * tot_fid / n)
        if ep == 0 or (ep + 1) % 25 == 0:
            print(f"  ep {ep+1}/{args.epochs}  mse {tot_mse/n:.4f}  fid {tot_fid/n:.4f}", flush=True)

    torch.save({"state_dict": model.state_dict(), "sched": sched, "stats": stats, "hist": hist,
                "config": {"feature_dim": 27, "context_dim": int(Y.shape[1]), "hidden": 384,
                           "layers": 8, "T": 1000, "schedule": "cosine"}, "lam": args.lam},
               "outputs/checkpoints/diffusion_pinn.pt")
    print(f"saved: outputs/checkpoints/diffusion_pinn.pt (final mse {hist['mse'][-1]:.4f}, "
          f"fid {hist['fid'][-1]:.4f})", flush=True)


if __name__ == "__main__":
    main()
