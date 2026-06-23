"""Lambda sweep for the surrogate fidelity penalty — find the recall/precision sweet spot.

lam=0.1 improved precision (0.45->0.64) and fixed over-scatter (traj spread 1.45->0.82) but
slightly over-corrected (spread <1) so recall only went 0.27->0.30. Hypothesis: smaller lambda
keeps more conditional spread (better recall) while retaining the calibration gain.

Reuses the saved surrogate. For each lambda: trains a diffusion with the penalty, then runs the
diversity diagnostic (recall/precision/spread vs data conditional). Cached baseline(0.0) and 0.1
are included in the final table.
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
from sklearn.neighbors import NearestNeighbors
from src.cnf import load_data, normalize, denormalize_H, H27_to_matrix
from src.diffusion import build_diffusion, sample_diffusion
from src import simulator as sim

COND_POP = slice(12, 33)
SURR_IDX = [ti * 9 + s for ti in (1, 3, 4) for s in range(7)]
LAMBDAS = [0.02, 0.05, 0.2]
CACHED = {0.0: dict(recall=0.27, precision=0.45, r_param=0.88, r_traj=1.45),
          0.1: dict(recall=0.30, precision=0.64, r_param=0.87, r_traj=0.82)}


class Surrogate(nn.Module):
    def __init__(self, d_in=27, d_out=63, h=384, layers=4):
        super().__init__()
        net = [nn.Linear(d_in, h), nn.SiLU()]
        for _ in range(layers - 1):
            net += [nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU()]
        net += [nn.Linear(h, d_out)]
        self.net = nn.Sequential(*net)
    def forward(self, x): return self.net(x)


def spread(X): return float(np.sqrt(np.sum(X.var(0))))


def coverage(data, model, radius):
    d2m = NearestNeighbors(n_neighbors=1).fit(model).kneighbors(data)[0][:, 0]
    d2d = NearestNeighbors(n_neighbors=1).fit(data).kneighbors(model)[0][:, 0]
    return float((d2m <= radius).mean()), float((d2d <= radius).mean())


def interp_pop(rho, tl, TIMES):
    diag = np.real(np.diagonal(rho, axis1=1, axis2=2))[:, :7]
    return np.array([[np.interp(t, tl, diag[:, s]) for s in range(7)] for t in TIMES]).reshape(-1)


def train_diffusion_pinn(Xn, Yn, surr, device, lam, pen, epochs=200):
    model, sched = build_diffusion(device, 27, Yn.shape[1], hidden=384, layers=8, T=1000)
    ab, T = sched["alpha_bars"], sched["T"]
    ds = TensorDataset(torch.tensor(Xn), torch.tensor(Yn))
    g = torch.Generator().manual_seed(0)
    tr, _ = random_split(ds, [len(ds) - 6000, 6000], generator=g)
    tl = DataLoader(tr, batch_size=2048, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    y_mu, y_sd, s_ymu, s_ysd, sidx = pen
    for ep in range(epochs):
        model.train()
        for x, c in tl:
            x, c = x.to(device), c.to(device); B = x.size(0)
            t_idx = torch.randint(0, T, (B,), device=device); a = ab[t_idx].view(-1, 1)
            eps = torch.randn_like(x); x_t = torch.sqrt(a) * x + torch.sqrt(1 - a) * eps
            ep_pred = model(x_t, t_idx.float() / T, c)
            mse = ((ep_pred - eps) ** 2).mean()
            x0 = (x_t - torch.sqrt(1 - a) * ep_pred) / torch.sqrt(a)
            tgt = (c[:, COND_POP] * y_sd + y_mu - s_ymu) / s_ysd
            fid = ((surr(x0)[:, sidx] - tgt) ** 2).mean()
            opt.zero_grad(); (mse + lam * fid).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model, sched


def diagnose(model, sched, X, Yn, stats, POP, TIMES, nn33, anchors, eta, device, NM=60):
    res = []
    for a in anchors:
        c = Yn[a]; nbr = nn33.kneighbors(c[None])[1][0]
        dX, dPOP = X[nbr], POP[nbr]
        S = sample_diffusion(model, sched, c, n=NM, device=device, n_steps=50, eta=0.0)
        mX = denormalize_H(S, stats)
        mPOP = np.array([_sim_pop(h, TIMES) for h in mX])
        mu, sd = dX.mean(0), dX.std(0) + 1e-9
        rp = spread((mX - mu) / sd) / spread((dX - mu) / sd)
        muP, sdP = dPOP.mean(0), dPOP.std(0) + 1e-9
        dPz, mPz = (dPOP - muP) / sdP, (mPOP - muP) / sdP
        rt = spread(mPz) / spread(dPz)
        rad = np.median(NearestNeighbors(n_neighbors=2).fit(dPz).kneighbors(dPz)[0][:, 1])
        rec, prec = coverage(dPz, mPz, rad)
        res.append((rp, rt, rec, prec))
    res = np.array(res)
    return dict(r_param=res[:,0].mean(), r_traj=res[:,1].mean(),
                recall=res[:,2].mean(), precision=res[:,3].mean())


def _sim_pop(h, TIMES):
    out = sim.simulate(H27_to_matrix(h), 35.0, return_traj=True)
    tl, rho = out["_traj"]; return interp_pop(rho, tl, TIMES)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = load_data(["data/dataset_full.npz", "data/dataset_full_v2.npz"])
    Xn, Yn, stats = normalize(X, Y)
    POP, TIMES = [], None
    for p in ("data/dataset_full.npz", "data/dataset_full_v2.npz"):
        d = np.load(p); POP.append(d["pop_t"][:, :, :7].reshape(len(d["pop_t"]), -1)); TIMES = d["times"]
    POP = np.concatenate(POP).astype(float); TIMES = np.asarray(TIMES, float)

    sk = torch.load("outputs/checkpoints/surrogate.pt", map_location=device, weights_only=False)
    surr = Surrogate().to(device); surr.load_state_dict(sk["state_dict"]); surr.eval()
    for prm in surr.parameters(): prm.requires_grad_(False)
    pen = (torch.tensor(stats["y_mu"][COND_POP], device=device),
           torch.tensor(stats["y_sd"][COND_POP], device=device),
           torch.tensor(sk["ymu"][SURR_IDX], device=device),
           torch.tensor(sk["ysd"][SURR_IDX], device=device),
           torch.tensor(SURR_IDX, device=device))

    eta = Y[:, 0]; subset = np.where(np.abs(eta - 0.95) < 0.03)[0]
    rng = np.random.default_rng(0)
    nn33 = NearestNeighbors(n_neighbors=100).fit(Yn)
    anchors = subset[rng.choice(len(subset), 3, replace=False)]

    results = dict(CACHED)
    for lam in LAMBDAS:
        print(f"\n■ training lam={lam} ...", flush=True)
        model, sched = train_diffusion_pinn(Xn, Yn, surr, device, lam, pen)
        d = diagnose(model, sched, X, Yn, stats, POP, TIMES, nn33, anchors, eta, device)
        results[lam] = d
        print(f"   lam={lam}: recall {d['recall']:.2f}  precision {d['precision']:.2f}  "
              f"spread(param {d['r_param']:.2f}, traj {d['r_traj']:.2f})", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("LAMBDA SWEEP — recall/precision/spread vs fidelity weight", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'lam':>6} | {'recall':>7} {'precis':>7} | {'spread_param':>12} {'spread_traj':>11}", flush=True)
    for lam in sorted(results):
        r = results[lam]
        tag = " (baseline)" if lam == 0.0 else ""
        print(f"  {lam:>6} | {r['recall']:>7.2f} {r['precision']:>7.2f} | "
              f"{r['r_param']:>12.2f} {r['r_traj']:>11.2f}{tag}", flush=True)
    best = max([l for l in results], key=lambda l: results[l]["recall"])
    print(f"\n  최대 recall: lam={best} (recall {results[best]['recall']:.2f}, "
          f"precision {results[best]['precision']:.2f})", flush=True)
    print("  목표: traj spread ≈ 1.0 유지하며 recall·precision 동시 ↑ 인 lambda", flush=True)
    Path("outputs/metrics/lambda_sweep.json").write_text(
        json.dumps({str(k): v for k, v in results.items()}, indent=2), encoding="utf-8")
    print("saved: outputs/metrics/lambda_sweep.json", flush=True)


if __name__ == "__main__":
    main()
