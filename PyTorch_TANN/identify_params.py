"""
Train the 1D elasto-hypoplastic TANN (same setup as TANN_1D_elasto-hypoplastic_mat.ipynb)
and identify physical constitutive parameters from the trained network:

  * elastic stiffness       E   = d2psi/d eps^2   (and -d2psi/d eps d eps_pl)
  * limit (saturation) stress S from the learned flow rule  d eps_pl = |d eps| sigma / S
  * offset yield stresses Rp0.05, Rp0.2 from a simulated monotonic tension test

The same quantities are identified from the reference data (ground truth), so both can be compared.

Usage (from this directory):
    python identify_params.py                 # train (5000 epochs, ~2 min CPU) and analyse
    python identify_params.py --load W.pt     # analyse previously saved weights
"""
import argparse
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares

torch.manual_seed(0)
np.random.seed(0)
DT = torch.float64
BATCH_TIME = 500
NAMES = 's_t s_tdt p_t p_tdt eel_t eel_tdt e_t e_tdt psi_t psi_tdt D'.split()


def get_prm(x):
    """[-1, 1] normalisation parameters (A, B) as in misc.preprocessing.GetParams."""
    return torch.tensor([0.5 * (x.max() - x.min()), 0.5 * (x.max() + x.min())], dtype=DT)


def nrm(x, p):
    return (x - p[1]) / p[0]


def dnrm(x, p):
    return x * p[0] + p[1]


def mlp(i, o, h, act):
    layers, d = [], i
    for hd in h:
        layers += [nn.Linear(d, hd, dtype=DT), act()]
        d = hd
    layers.append(nn.Linear(d, o, dtype=DT))
    return nn.Sequential(*layers)


class TANN(nn.Module):
    def __init__(self, prm):
        super().__init__()
        self.prm = prm
        self.NNz = mlp(3, 1, [32, 32, 32], nn.ReLU)   # evolution law  d eps_pl
        self.NNf = mlp(2, 1, [144, 144], nn.GELU)     # free energy    psi(eps_pl, eps)

    def psi(self, eps, pl):
        """Free energy in physical units as a function of physical eps and eps_pl."""
        P = self.prm
        return dnrm(self.NNf(torch.cat((nrm(pl, P['z']), nrm(eps, P['e'])), 1)), P['f'])

    def dpl(self, eps_t, pl_t, deps):
        P = self.prm
        x = torch.cat((nrm(eps_t, P['e']), nrm(pl_t, P['z']), nrm(deps, P['de'])), 1)
        return dnrm(self.NNz(x), P['dz'])

    def forward(self, eps_t, pl_t, deps):
        """Physical inputs -> physical (d eps_pl, psi, sigma, dissipation)."""
        dpl = self.dpl(eps_t, pl_t, deps)
        eps = (eps_t + deps).detach().requires_grad_(True)
        pl = pl_t + dpl
        psi = self.psi(eps, pl)
        sig, dpsi_dpl = torch.autograd.grad(psi, (eps, pl), torch.ones_like(psi), create_graph=True)
        return dpl, psi, sig, -dpsi_dpl * dpl


def load_data(path):
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    d = {n: torch.tensor(np.asarray(v, dtype=np.float64).reshape(-1, BATCH_TIME, 1)) for n, v in zip(NAMES, raw)}
    d['de'] = d['e_tdt'] - d['e_t']
    d['dp'] = d['p_tdt'] - d['p_t']
    return d


def split(d):
    n = d['e_t'].shape[0]
    ntrain = int(round(n * 0.1)); nval = int(round(n * 0.45))
    tv = {k: v[:ntrain + nval] for k, v in d.items()}
    test = {k: v[ntrain + nval:] for k, v in d.items()}
    return tv, test, ntrain


def train(model, tv, ntrain, niters, lr=1e-4):
    P = model.prm
    opt = torch.optim.NAdam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    for it in range(1, niters + 1):
        tot = 0.
        for i in range(ntrain):
            dpl, psi, sig, D = model(tv['e_t'][i], tv['p_t'][i], tv['de'][i])
            loss = (mse(nrm(dpl, P['dz']), nrm(tv['dp'][i], P['dz'])) + mse(nrm(psi, P['f']), nrm(tv['psi_tdt'][i], P['f']))
                    + mse(nrm(sig, P['s']), nrm(tv['s_tdt'][i], P['s'])) + mse(nrm(D, P['d']), nrm(tv['D'][i], P['d'])))
            loss.backward(); opt.step(); opt.zero_grad()
            tot += loss.item()
        if it % 500 == 0 or it == 1:
            print(f'epoch {it:5d}  loss {tot:.3e}', flush=True)


def fit_S_exact(de, s0, s1, E):
    """Reference law, exactly integrated: sigma_tdt = sgn*S + (sigma_t - sgn*S) exp(-E|de|/S)."""
    sg = np.sign(de)
    r = least_squares(lambda x: sg * x[0] + (s0 - sg * x[0]) * np.exp(-E * np.abs(de) / x[0]) - s1, [100.])
    return r.x[0]


def fit_S_flow(de, dp, s_mid):
    """Flow rule d eps_pl = |d eps| sigma / S  (sigma at mid-step)."""
    a = np.abs(de) * s_mid
    return np.sum(a * a) / np.sum(a * dp)


def offset_yield(eps, sig, E, offset):
    pl = eps - sig / E
    return float(np.interp(offset, pl, sig))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./data_1D_hypo_500real')
    ap.add_argument('--niters', type=int, default=5000)
    ap.add_argument('--load', default=None)
    ap.add_argument('--save', default='../results/tann_weights.pt')
    ap.add_argument('--out', default='../results/identified_params.json')
    args = ap.parse_args()

    d = load_data(args.data)
    tv, test, ntrain = split(d)
    prm = {'e': get_prm(tv['e_tdt']), 'de': get_prm(tv['de']), 's': get_prm(tv['s_tdt']),
           'z': get_prm(tv['p_tdt']), 'dz': get_prm(tv['dp']), 'f': get_prm(tv['psi_tdt']), 'd': get_prm(tv['D'])}
    model = TANN(prm)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    else:
        train(model, tv, ntrain, args.niters)
        torch.save(model.state_dict(), args.save)

    # ---------------- reference (ground truth) parameters ----------------
    f = lambda k: d[k].numpy().ravel()
    E_ref = float(np.median(f('s_tdt')[np.abs(f('eel_tdt')) > 1e-6] / f('eel_tdt')[np.abs(f('eel_tdt')) > 1e-6]))
    S_ref = fit_S_exact(f('de'), f('s_t'), f('s_tdt'), E_ref)

    # ---------------- TANN: stiffness from Hessian of psi on test states ----------------
    eps = test['e_tdt'].reshape(-1, 1).clone().requires_grad_(True)
    pl = test['p_tdt'].reshape(-1, 1).clone().requires_grad_(True)
    psi = model.psi(eps, pl)
    sig, dpsi_dpl = torch.autograd.grad(psi, (eps, pl), torch.ones_like(psi), create_graph=True)
    H_ee, H_ep = torch.autograd.grad(sig, (eps, pl), torch.ones_like(sig))
    H_ee, H_ep = H_ee.numpy().ravel(), H_ep.numpy().ravel()
    E_nn = float(np.median(H_ee))

    # ---------------- TANN: one-step predictions on test set ----------------
    with torch.enable_grad():
        dpl_p, psi_p, sig_p, D_p = [t.detach().numpy().ravel() for t in
                                    model(test['e_t'].reshape(-1, 1), test['p_t'].reshape(-1, 1), test['de'].reshape(-1, 1))]
    ft = lambda k: test[k].numpy().ravel()
    S_nn = fit_S_flow(ft('de'), dpl_p, 0.5 * (ft('s_t') + sig_p))
    rel = lambda p, r: float(np.sqrt(np.mean((p - r) ** 2)) / np.max(np.abs(r)))
    errors = {'d_eps_pl': rel(dpl_p, ft('dp')), 'psi': rel(psi_p, ft('psi_tdt')),
              'sigma': rel(sig_p, ft('s_tdt')), 'dissipation': rel(D_p, ft('D'))}

    # ---------------- monotonic tension: TANN recursive vs analytic ----------------
    n_steps = int(round(d['e_t'].abs().max().item() / 1e-4))
    e, p, eps_l, sig_l = torch.zeros(1, 1, dtype=DT), torch.zeros(1, 1, dtype=DT), [0.], [0.]
    de = torch.full((1, 1), 1e-4, dtype=DT)
    for _ in range(n_steps):
        dp_, _, s_, _ = model(e, p, de)
        e, p = e + de, (p + dp_).detach()
        eps_l.append(e.item()); sig_l.append(s_.item())
    eps_m, sig_nn = np.array(eps_l), np.array(sig_l)
    sig_ref = S_ref * (1 - np.exp(-E_ref * eps_m / S_ref))

    res = {
        'reference': {'E': E_ref, 'S_limit_stress': S_ref,
                      'Rp0.05': offset_yield(eps_m, sig_ref, E_ref, 5e-4), 'Rp0.2': offset_yield(eps_m, sig_ref, E_ref, 2e-3),
                      'sigma_at_eps_max': float(sig_ref[-1])},
        'TANN': {'E_d2psi_deps2_median': E_nn, 'E_p5_p95': [float(np.percentile(H_ee, 5)), float(np.percentile(H_ee, 95))],
                 'minus_d2psi_deps_dpl_median': float(np.median(-H_ep)),
                 'S_limit_stress_from_flow_rule': float(S_nn),
                 'Rp0.05': offset_yield(eps_m, sig_nn, E_nn, 5e-4), 'Rp0.2': offset_yield(eps_m, sig_nn, E_nn, 2e-3),
                 'sigma_at_eps_max': float(sig_nn[-1]),
                 'test_rel_rmse': errors},
        'monotonic_curve': {'eps': eps_m.tolist(), 'sigma_ref': sig_ref.tolist(), 'sigma_tann': sig_nn.tolist()},
    }
    with open(args.out, 'w') as fo:
        json.dump(res, fo, indent=1)
    print(json.dumps({k: res[k] for k in ('reference', 'TANN')}, indent=1))


if __name__ == '__main__':
    main()
