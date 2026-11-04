"""Steps 3-5: nominal/robust x efficient/fair charging-capacity siting and the robust fairness premium.

Budgeted capacitated coverage model on the zones of the selected dataset (scipy.milp / HiGHS):
  y_j open site, c_j capacity (kW), a_ij share of zone-i demand served at j (dist_ij <= R).
  max  sum_i d_i sum_j a_ij                       (served nominal demand)
  s.t. a_ij <= y_j,  sum_j a_ij <= 1,  c_j <= M_j y_j,  F sum y + kappa sum c <= B
       sum_i a_ij d_i + G p_j + sum_i q_ij <= c_j,  p_j + q_ij >= a_ij r_i    (Bertsimas-Sim; r=0 -> nominal)
       sum_j a_ij >= tau                                                      (fair variant)
       LinDistFlow voltage limits on IEEE-33 feeder copies                    (optional grid layer)
Cost of a plan = unserved nominal demand U. RFP = [U(RF)-U(RE)] - [U(NF)-U(NE)].
Radius modes: real conformal radii, gradient-neutral uniform (same total radius), rank-flipped.
Set RADIUS=peak to build radii from daily-peak errors instead of hourly errors.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from common import OUT, default_radius_km, distance_matrix, zone_static

ALPHA = 0.10


def cq(scores, alpha=ALPHA):
    n = scores.shape[0]
    return np.quantile(scores, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0), axis=0, method="higher")


RADIUS = os.environ.get("RADIUS", "hourly")                              # hourly | peak


def demand_and_radius(split, model, radius=None):
    radius = radius or RADIUS
    files = sorted(glob.glob(str(OUT / f"preds_{split}_{model}_s*.npz")))
    dhat, rad, real = [], [], None
    for f in files:
        d = np.load(f)
        dhat.append(d["test_pred"].max(1).mean(0))                       # mean daily peak forecast
        if radius == "peak":                                             # error of the daily peak itself
            rad.append(cq(np.abs(d["cal_true"].max(1) - d["cal_pred"].max(1))))
        else:
            rad.append(cq(np.abs(d["cal_true"] - d["cal_pred"]).reshape(-1, d["cal_true"].shape[2])))
        real = d["test_true"].max(1)                                      # realised daily peaks (days x zones)
        zones = d["zones"]
    return zones, np.mean(dhat, 0), np.mean(rad, 0), real


def radius_variant(dhat, r, mode):
    if mode == "real":
        return r
    rho = r / np.maximum(dhat, 1e-6)
    if mode == "uniform":
        new = np.full_like(rho, 1.0)
    elif mode == "flipped":
        order = np.argsort(rho)
        new = np.empty_like(rho)
        new[order] = np.sort(rho)[::-1]
    else:
        raise ValueError(mode)
    return new * dhat * r.sum() / (new * dhat).sum()                    # keep total radius fixed


def grid_sensitivity(zones, st, mapping, n_feeders, vmin, base_scale, seed=0):
    """Map zones onto copies of the IEEE 33-bus feeder; return voltage-sensitivity rows (A c <= b, c in kW)."""
    import pandapower as pp
    import pandapower.networks as pn
    from sklearn.cluster import KMeans

    net = pn.case33bw()
    net.load["p_mw"] *= base_scale
    net.load["q_mvar"] *= base_scale
    pp.runpp(net)
    vb2 = net.res_bus.vm_pu.to_numpy() ** 2
    vn = net.bus.vn_kv.iloc[0] * 1e3
    nb = len(net.bus)
    parent, rline = {}, {}
    for _, ln in net.line.iterrows():
        parent[int(ln.to_bus)] = int(ln.from_bus)
        rline[int(ln.to_bus)] = ln.r_ohm_per_km * ln.length_km
    paths = {}
    for b in range(nb):
        path, x = set(), b
        while x in parent:
            path.add(x)
            x = parent[x]
        paths[b] = path
    S = np.array([[2 * sum(rline[l] for l in paths[n] & paths[m]) / vn ** 2 for m in range(nb)] for n in range(nb)])
    S_kw = S * 1e3                                                       # per kW of added load

    xy = st[["lon", "lat"]].to_numpy()
    lab = KMeans(n_feeders, n_init=10, random_state=seed).fit_predict(xy)
    load_buses = np.arange(1, nb)
    elec_depth = np.array([sum(rline[l] for l in paths[b]) for b in load_buses])
    bus_of = np.zeros(len(zones), dtype=int)
    rng = np.random.default_rng(seed)
    for k in range(n_feeders):
        idx = np.where(lab == k)[0]
        cen = xy[idx].mean(0)
        dist = np.linalg.norm(xy[idx] - cen, axis=1)
        if mapping == "M1":                     # zones near the cluster centre sit near the substation
            zorder = idx[np.argsort(dist)]
        elif mapping == "M2":
            zorder = rng.permutation(idx)
        else:                                   # M3: zones near the centre sit at the feeder end
            zorder = idx[np.argsort(-dist)]
        border = load_buses[np.argsort(elec_depth)]
        for t, z in enumerate(zorder):
            bus_of[z] = border[int(t * len(border) / len(zorder))]
    rows, rhs = [], []
    for k in range(n_feeders):
        idx = np.where(lab == k)[0]
        for n in load_buses:
            row = np.zeros(len(zones))
            row[idx] = S_kw[n, bus_of[idx]]
            rows.append(row)
            rhs.append(vb2[n] - vmin ** 2)
    return np.array(rows), np.array(rhs), lab, bus_of, float(net.res_bus.vm_pu.min())


LAST = {}


def solve(dhat, r, D, R_km, budget, F, kappa, gamma, tau, grid=None, time_limit=300, gap=1e-3, min_cost=False,
          relax=False, max_tau=False, feas_only=False):
    n = len(dhat)
    pairs = np.argwhere(D <= R_km * 1000)                                # (i, j)
    P = len(pairs)
    I, J = pairs[:, 0], pairs[:, 1]
    nv = 3 * n + 2 * P                                                   # y, c, p, a, q
    oy, oc, op, oa, oq = 0, n, 2 * n, 3 * n, 3 * n + P
    M = np.bincount(J, weights=dhat[I] + r[I], minlength=n)
    rows, lo, hi = [], [], []

    def add(mat, l, h):
        rows.append(sp.csr_matrix(mat))
        lo.append(np.atleast_1d(l))
        hi.append(np.atleast_1d(h))

    ar = np.arange(P)
    add(sp.coo_matrix((np.r_[np.ones(P), -np.ones(P)], (np.r_[ar, ar], np.r_[oa + ar, oy + J])), (P, nv)),
        np.full(P, -np.inf), np.zeros(P))                                # a_ij <= y_j
    data = np.r_[dhat[I], np.full(n, gamma), np.ones(P), -np.ones(n)]
    ri = np.r_[J, np.arange(n), J, np.arange(n)]
    ci = np.r_[oa + ar, op + np.arange(n), oq + ar, oc + np.arange(n)]
    add(sp.coo_matrix((data, (ri, ci)), (n, nv)), np.full(n, -np.inf), np.zeros(n))   # robust capacity
    add(sp.coo_matrix((np.r_[r[I], -np.ones(P), -np.ones(P)], (np.r_[ar, ar, ar], np.r_[oa + ar, op + J, oq + ar])),
                      (P, nv)), np.full(P, -np.inf), np.zeros(P))       # p_j + q_ij >= a_ij r_i
    add(sp.coo_matrix((np.ones(P), (I, oa + ar)), (n, nv)), np.full(n, tau), np.ones(n))  # tau <= s_i <= 1
    add(sp.coo_matrix((np.r_[np.ones(n), -M], (np.r_[np.arange(n), np.arange(n)], np.r_[oc + np.arange(n), oy + np.arange(n)])),
                      (n, nv)), np.full(n, -np.inf), np.zeros(n))       # c_j <= M_j y_j
    b = np.zeros(nv)
    b[oy:oy + n], b[oc:oc + n] = F, kappa
    add(b[None, :], -np.inf, budget)
    if grid is not None:
        G, g_rhs = grid
        add(sp.hstack([sp.csr_matrix((G.shape[0], oc)), sp.csr_matrix(G), sp.csr_matrix((G.shape[0], nv - oc - n))]),
            np.full(G.shape[0], -np.inf), g_rhs)
    A = sp.vstack(rows).tocsr()
    lo, hi = np.concatenate(lo), np.concatenate(hi)
    obj = b.copy() if min_cost else np.zeros(nv)
    if not (min_cost or feas_only):
        obj[oa:oa + P] = -dhat[I]
    ub = np.full(nv, np.inf)
    ub[oy:oy + n] = 1
    ub[oa:oa + P] = 1
    if not np.any(r > 0):
        ub[op:op + n], ub[oq:oq + P] = 0, 0
    if max_tau:                                                          # extra variable t: maximise min_i s_i
        A = sp.vstack([sp.hstack([A, sp.csr_matrix((A.shape[0], 1))]),
                       sp.hstack([sp.coo_matrix((np.ones(P), (I, oa + ar)), (n, nv)), -np.ones((n, 1))])]).tocsr()
        lo, hi = np.r_[lo, np.zeros(n)], np.r_[hi, np.full(n, np.inf)]
        obj = np.r_[1e-4 * obj / dhat.sum(), -1.0]
        ub = np.r_[ub, 1.0]
        nv += 1
    cons = LinearConstraint(A, lo, hi)
    integ = np.zeros(nv)
    if not relax:
        integ[oy:oy + n] = 1
    res = milp(obj, constraints=cons, integrality=integ, bounds=Bounds(np.zeros(nv), ub),
               options=dict(time_limit=time_limit, mip_rel_gap=gap, disp=False))
    LAST.update(status=res.status, message=res.message)
    if res.x is None:
        return None
    x = res.x
    a = sp.csr_matrix((x[oa:oa + P], (I, J)), (n, n))
    return dict(y=x[oy:oy + n].round(), c=x[oc:oc + n], a=a, s=np.asarray(a.sum(1)).ravel(), t=x[-1] if max_tau else np.nan,
                served=float(dhat[I] @ x[oa:oa + P]), status=res.status, gap=getattr(res, "mip_gap", np.nan))


def evaluate(plan, real, kp=5.0):
    """Proportional rationing at each station on every realised test day."""
    a, c = plan["a"], plan["c"]
    serv = []
    for d in real:
        load = np.asarray(a.multiply(d[:, None]).sum(0)).ravel()
        frac = np.where(load > 0, np.minimum(1, c / np.maximum(load, 1e-9)), 1)
        serv.append(np.asarray(a.multiply(frac[None, :]).sum(1)).ravel())
    serv = np.array(serv)
    unmet_ratio = 1 - serv.mean(0)
    unmet_kwh = float(((1 - serv) * real).sum(1).mean())
    ede = float(np.log(np.mean(np.exp(kp * unmet_ratio))) / kp)       # Kolm-Pollak EDE of unmet ratio
    return dict(real_unmet_kwh=unmet_kwh, real_unmet_ratio=float(unmet_kwh / real.sum(1).mean()),
                ede_unmet=ede, worst_decile_unmet=float(np.quantile(unmet_ratio, 0.9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="A")
    ap.add_argument("--model", default="iTransformer")
    ap.add_argument("--radius_km", type=float, default=None, help="default: common.default_radius_km()")
    ap.add_argument("--betas", nargs="+", type=float, default=[0.7])
    ap.add_argument("--gammas", nargs="+", type=float, default=[1, 2, 3])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5])
    ap.add_argument("--modes", nargs="+", default=["real", "uniform", "flipped"])
    ap.add_argument("--grids", nargs="+", default=["none"])
    ap.add_argument("--feeders", type=int, default=8)
    ap.add_argument("--vmin", type=float, default=0.93)
    ap.add_argument("--base_scale", type=float, default=0.6)
    ap.add_argument("--fmult", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.radius_km = args.radius_km or default_radius_km()

    zones, dhat, r0, real = demand_and_radius(args.split, args.model)
    st = zone_static(zones)
    D = distance_matrix()
    kappa = 1.0
    F = args.fmult * kappa * np.median(dhat)
    full = solve(dhat, np.zeros_like(dhat), D, args.radius_km, np.inf, F, kappa, 0, 1.0, min_cost=True)
    full_cost = F * full["y"].sum() + kappa * full["c"].sum()
    print(json.dumps(dict(zones=len(zones), total_peak_kw=float(dhat.sum()), total_radius=float(r0.sum()),
                          F=float(F), full_service_cost=float(full_cost), full_sites=int(full["y"].sum()))), flush=True)

    out = []
    for grid_name, beta, gamma, tau, mode in itertools.product(args.grids, args.betas, args.gammas, args.taus, args.modes):
        grid = None
        if grid_name != "none":
            G, g_rhs, lab, bus_of, vmin_base = grid_sensitivity(zones, st, grid_name, args.feeders, args.vmin, args.base_scale)
            grid = (G, g_rhs)
        r = radius_variant(dhat, r0, mode)
        B = beta * full_cost
        U, ev = {}, {}
        for key, rr, tt in [("NE", 0 * r, 0.0), ("NF", 0 * r, tau), ("RE", r, 0.0), ("RF", r, tau)]:
            plan = solve(dhat, rr, D, args.radius_km, B, F, kappa, gamma, tt, grid)
            if plan is None:
                U[key] = np.nan
                ev[key] = {}
                continue
            U[key] = float(dhat.sum() - plan["served"])
            ev[key] = evaluate(plan, real)
            ev[key]["sites"] = int(plan["y"].sum())
            ev[key]["status"] = int(plan["status"])
            ev[key]["gap"] = float(plan["gap"])
        por, pof = U["RE"] - U["NE"], U["NF"] - U["NE"]
        rfp = (U["RF"] - U["RE"]) - (U["NF"] - U["NE"])
        rec = dict(split=args.split, model=args.model, grid=grid_name, beta=beta, gamma=gamma, tau=tau, mode=mode,
                   U_NE=U["NE"], U_NF=U["NF"], U_RE=U["RE"], U_RF=U["RF"], PoR=por, PoF=pof, RFP=rfp,
                   RFP_rel=rfp / max(pof, 1e-9) if pof > 0 else np.nan)
        for k in ev:
            for m, val in ev[k].items():
                rec[f"{k}_{m}"] = val
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        out.append(rec)
    pd.DataFrame(out).to_csv(OUT / f"rfp_{args.split}_{args.model}{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
