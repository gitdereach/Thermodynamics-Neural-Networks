"""
TANN that discovers its internal variable from stress-strain data only.

Training data: strain eps and stress sigma along loading paths (what a testing machine gives).
NOT used for training: plastic strain, free energy, dissipation (kept only for evaluation).

Model (1D, rate independent, isothermal):
  free energy        psi = Psi(eps, z)                      (NN, smooth)
  stress             sigma = d psi / d eps                  (autodiff)
  evolution law      dz = |d eps| * F(eps, z, sign(d eps))  (NN; rate independent by construction)
  dissipation        D = -d psi / d z * dz >= 0            (penalised if negative)
  z(t=0) = 0, psi(0, 0) = 0

Training = trajectory fitting: z is integrated along each path starting from 0 and the loss is the stress error
along the whole path (+ dissipation-inequality penalty). The learned z is then compared with the true plastic strain.

Usage (from this directory):
    python discover_internal_var.py                     # train on 50% of the paths, evaluate on the test paths
    python discover_internal_var.py --train-frac 0.1
    python discover_internal_var.py --load ../results/discover_weights.pt
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from identify_params import load_data, split, fit_S_flow, offset_yield

torch.manual_seed(0)
np.random.seed(0)
DT = torch.float64


def mlp(i, o, h, act):
    layers, d = [], i
    for hd in h:
        layers += [nn.Linear(d, hd, dtype=DT), act()]
        d = hd
    layers.append(nn.Linear(d, o, dtype=DT))
    return nn.Sequential(*layers)


class DiscoveryTANN(nn.Module):
    def __init__(self, e_scale, s_scale, nz=1):
        super().__init__()
        self.e_scale, self.s_scale, self.nz = e_scale, s_scale, nz
        self.Psi = mlp(1 + nz, 1, [64, 64], nn.Softplus)        # smooth -> well defined stiffness
        self.F = mlp(2 + nz, nz, [64, 64, 64], nn.GELU)

    # all network quantities are dimensionless: en = eps / e_scale, psi_n such that sigma = s_scale * dpsi_n/den
    def psi(self, en, z):
        x = torch.cat((en, z), 1)
        return self.Psi(x) - self.Psi(torch.zeros_like(x))

    def stress_and_force(self, en, z):
        psi = self.psi(en, z)
        g_e, g_z = torch.autograd.grad(psi, (en, z), torch.ones_like(psi), create_graph=True)
        return psi, g_e * self.s_scale, -g_z            # psi_n, sigma (physical), thermodynamic force (dimensionless)

    def rollout(self, eps, create_graph=True):
        """eps: (n_paths, n_steps+1) physical strain, eps[:,0] = 0. Returns sigma, z, psi_n, D_n along paths."""
        n, T = eps.shape
        en_all = eps / self.e_scale
        z = torch.zeros(n, self.nz, dtype=DT)
        sig, zs, psis, Ds = [], [], [], []
        for k in range(T - 1):
            en_k = en_all[:, k:k + 1]
            den = en_all[:, k + 1:k + 2] - en_k
            dz = den.abs() * self.F(torch.cat((en_k, z, torch.sign(den)), 1))
            z = z + dz
            en1 = en_all[:, k + 1:k + 2].clone().requires_grad_(True)
            zz = z if z.requires_grad else z.clone().requires_grad_(True)
            psi, s, force = self.stress_and_force(en1, zz)
            D = (force * dz).sum(1, keepdim=True)
            if not create_graph:
                s, psi, D, z = s.detach(), psi.detach(), D.detach(), z.detach()
            sig.append(s); zs.append(z); psis.append(psi); Ds.append(D)
        cat = lambda L: torch.cat(L, 1) if L[0].shape[1] == 1 else torch.stack(L, 1)
        return cat(sig), cat(zs), cat(psis), cat(Ds)


def path_tensors(dd):
    """(n_paths, 500, 1) arrays -> eps (n, 501) starting at 0, sigma (n, 500) at the end of each step."""
    eps = torch.cat((dd['e_t'][:, :1, 0], dd['e_tdt'][:, :, 0]), 1)
    return eps, dd['s_tdt'][:, :, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./data_1D_hypo_500real')
    ap.add_argument('--train-frac', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--w-diss', type=float, default=10.)
    ap.add_argument('--load', default=None)
    ap.add_argument('--save', default='../results/discover_weights.pt')
    ap.add_argument('--out', default='../results/discover_internal_var.json')
    args = ap.parse_args()

    d = load_data(args.data)
    tv, test, ntrain = split(d, args.train_frac)
    tr = {k: v[:ntrain] for k, v in tv.items()}
    eps_tr, sig_tr = path_tensors(tr)
    e_scale, s_scale = float(eps_tr.abs().max()), float(sig_tr.abs().max())
    model = DiscoveryTANN(e_scale, s_scale)
    print(f'training paths: {ntrain}, test paths: {test["e_t"].shape[0]}  (data used: strain, stress only)', flush=True)

    if args.load:
        model.load_state_dict(torch.load(args.load))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr / 100)
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            s, z, psi, D = model.rollout(eps_tr)
            l_sig = torch.mean(((s - sig_tr) / s_scale) ** 2)
            l_diss = torch.mean(torch.relu(-D) ** 2) * 1e4
            loss = l_sig + args.w_diss * l_diss
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if ep % 100 == 0 or ep == 1:
                print(f'epoch {ep:5d}  L_sigma {l_sig.item():.3e}  L_diss {l_diss.item():.3e}  ({time.time() - t0:.0f}s)', flush=True)
        torch.save(model.state_dict(), args.save)

    # ------------------------------ evaluation on unseen test paths ------------------------------
    eps_te, sig_te = path_tensors(test)
    s, z, psi_n, D_n = model.rollout(eps_te, create_graph=False)
    s, z = s.numpy(), z.numpy()
    pl = test['p_tdt'][:, :, 0].numpy()
    rel = lambda p, r: float(np.sqrt(np.mean((p - r) ** 2)) / np.max(np.abs(r)))

    # z vs plastic strain: linear map fitted on the test paths
    A = np.vstack([z.ravel(), np.ones(z.size)]).T
    coef, *_ = np.linalg.lstsq(A, pl.ravel(), rcond=None)
    pl_from_z = (A @ coef).reshape(pl.shape)
    r2 = 1 - np.sum((pl - pl_from_z) ** 2) / np.sum((pl - pl.mean()) ** 2)

    # physical energy / dissipation (psi_phys = s_scale * e_scale * psi_n)
    psi_phys = (psi_n.numpy() * s_scale * e_scale)
    D_phys = D_n.numpy() * s_scale * e_scale

    # stiffness d2psi/deps2 at fixed z, on test states
    en = torch.tensor(eps_te[:, 1:].reshape(-1, 1) / e_scale).requires_grad_(True)
    zz = torch.tensor(z.reshape(-1, 1)).requires_grad_(True)
    psi = model.psi(en, zz)
    g_e, = torch.autograd.grad(psi, en, torch.ones_like(psi), create_graph=True)
    H, = torch.autograd.grad(g_e, en, torch.ones_like(g_e))
    E_nn = H.numpy().ravel() * s_scale / e_scale
    # limit stress from the learned flow rule, expressed in terms of the recovered plastic strain
    dpl = np.diff(np.concatenate((np.zeros((pl.shape[0], 1)), pl_from_z), 1), axis=1)
    de = np.diff(eps_te.numpy(), axis=1)
    s_prev = np.concatenate((np.zeros((s.shape[0], 1)), s[:, :-1]), 1)
    S_nn = fit_S_flow(de.ravel(), dpl.ravel(), 0.5 * (s_prev + s).ravel())

    # monotonic tension
    eps_m = torch.linspace(0, float(d['e_t'].abs().max()), 61, dtype=DT)[None, :]
    sm, *_ = model.rollout(eps_m, create_graph=False)
    eps_m, sig_m = eps_m.numpy().ravel(), np.concatenate(([0.], sm.numpy().ravel()))
    E_med = float(np.median(E_nn))

    res = {
        'train_frac': args.train_frac, 'n_train_paths': ntrain, 'data_used_for_training': ['strain', 'stress'],
        'test_rel_rmse_full_path': {'sigma': rel(s, sig_te.numpy()), 'psi': rel(psi_phys, test['psi_tdt'][:, :, 0].numpy()),
                                    'dissipation': rel(D_phys, test['D'][:, :, 0].numpy())},
        'z_vs_plastic_strain': {'R2_linear': float(r2), 'pl = a*z + b': [float(coef[0]), float(coef[1])]},
        'negative_dissipation_fraction': float(np.mean(D_phys < -1e-9)),
        'E_d2psi_deps2_median': E_med, 'E_p5_p95': [float(np.percentile(E_nn, 5)), float(np.percentile(E_nn, 95))],
        'S_limit_stress': float(S_nn),
        'Rp0.05': offset_yield(eps_m, sig_m, E_med, 5e-4), 'Rp0.2': offset_yield(eps_m, sig_m, E_med, 2e-3),
        'sigma_at_eps_max': float(sig_m[-1]),
        'curves_test_path0': {'eps': eps_te[0, 1:].tolist(), 'sigma_ref': sig_te[0].tolist(), 'sigma_pred': s[0].tolist(),
                              'pl_ref': pl[0].tolist(), 'pl_from_z': pl_from_z[0].tolist(), 'z': z[0].ravel().tolist(),
                              'psi_ref': test['psi_tdt'][0, :, 0].tolist(), 'psi_pred': psi_phys[0].ravel().tolist(),
                              'D_ref': test['D'][0, :, 0].tolist(), 'D_pred': D_phys[0].ravel().tolist()},
        'monotonic_curve': {'eps': eps_m.tolist(), 'sigma': sig_m.tolist()},
    }
    with open(args.out, 'w') as fo:
        json.dump(res, fo, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('curves_test_path0', 'monotonic_curve')}, indent=1))


if __name__ == '__main__':
    main()
