"""Calibration and ex ante diagnostics (no new forecasts needed).

1. Hourly vs daily-peak conformal radii: rank agreement and per-zone test coverage of both constructions.
2. Ex ante dodgeable share for every permutation file named alignment_<split>_<model>_g<G>[_t<tau>_b<beta>].csv:
   rebuilds the random layouts with the same seeds, solves NE once per budget and correlates the index with the
   RFP and with the realised dodged share. Writes exante_<file> next to each input.
Usage: EVDATA=urbanev python diagnostics.py --splits A B ;  EVDATA=charged_SZH python diagnostics.py --splits C
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import OUT, default_radius_km, distance_matrix
from exp3_siting import cq, demand_and_radius, solve
from exp4_alignment import dodgeable_share


def coverage(split, model):
    rows = []
    for f in sorted(glob.glob(str(OUT / f"preds_{split}_{model}_s*.npz"))):
        d = np.load(f)
        e_cal = np.abs(d["cal_true"] - d["cal_pred"]).reshape(-1, d["cal_true"].shape[2])
        e_test = np.abs(d["test_true"] - d["test_pred"]).reshape(-1, d["test_true"].shape[2])
        p_cal = np.abs(d["cal_true"].max(1) - d["cal_pred"].max(1))
        p_test = np.abs(d["test_true"].max(1) - d["test_pred"].max(1))
        rh, rp = cq(e_cal), cq(p_cal)
        rows.append(dict(cov_hourly=(e_test <= rh).mean(0), cov_peak=(p_test <= rp).mean(0),
                         cov_hourly_on_peak=(p_test <= rh).mean(0), rh=rh, rp=rp))
    out = {k: np.mean([r[k] for r in rows], 0) for k in rows[0]}
    return out


def summarise_cov(c, name):
    v = c[name]
    return dict(mean=float(v.mean()), p10=float(np.quantile(v, 0.1)), median=float(np.median(v)),
                share_below_080=float((v < 0.8).mean()))


def exante(split, model, radius_km):
    zones, dhat, r0, _ = demand_and_radius(split, model)
    D = distance_matrix()
    rho = r0 / np.maximum(dhat, 1e-6)
    F = 2 * np.median(dhat)
    full = solve(dhat, np.zeros_like(dhat), D, radius_km, np.inf, F, 1.0, 0, 1.0, min_cost=True)
    full_cost = F * full["y"].sum() + full["c"].sum()
    qcache, res = {}, []
    for f in sorted(glob.glob(str(OUT / f"alignment_{split}_*_g*.csv"))):
        name = os.path.basename(f)
        m = re.match(rf"alignment_{split}_(\w+?)_g([\d.]+)(?:_t([\d.]+)_b([\d.]+))?\.csv$", name)
        if not m or m.group(1) != model:
            continue
        tau = float(m.group(3) or 0.5)
        beta = float(m.group(4) or 0.7)
        if beta not in qcache:
            ne = solve(dhat, 0 * dhat, D, radius_km, beta * full_cost, F, 1.0, 0, 0.0)
            qcache[beta] = 1 - ne["served"] / dhat.sum()
        q = qcache[beta]
        df = pd.read_csv(f)
        dq = []
        for k in df.perm:
            rk = rho if k == -1 else np.random.default_rng(1000 + int(k)).permutation(rho)
            r = rk * dhat * r0.sum() / (rk * dhat).sum()
            dq.append(dodgeable_share(rk, r, dhat, q))
        df["dodgeable"] = dq
        p = df[df.perm >= 0].dropna(subset=["RFP"])
        res.append(dict(file=name, gamma=float(m.group(2)), tau=tau, beta=beta, q=q, n=len(p),
                        rs_RFP_dodgeable=spearmanr(p.RFP, p.dodgeable)[0],
                        rs_dodgeable_dodgedRE=spearmanr(p.dodgeable, p.dodged_RE)[0],
                        rs_RFP_dodgedRE=spearmanr(p.RFP, p.dodged_RE)[0],
                        rs_PoR_dodgeable=spearmanr(p.PoR, p.dodgeable)[0],
                        real_dodgeable=float(df[df.perm == -1].dodgeable.iloc[0]), perm_dodgeable_mean=p.dodgeable.mean()))
        df.to_csv(OUT / f"exante_{name}", index=False)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[-1].items()}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["A", "B"])
    ap.add_argument("--models", nargs="+", default=["iTransformer", "DLinear"])
    ap.add_argument("--radius_km", type=float, default=None, help="default: common.default_radius_km()")
    args = ap.parse_args()
    args.radius_km = args.radius_km or default_radius_km()
    cov_rows, ex_rows = [], []
    for split in args.splits:
        for model in args.models:
            if not glob.glob(str(OUT / f"preds_{split}_{model}_s*.npz")):
                continue
            c = coverage(split, model)
            rho_h, rho_p = c["rh"], c["rp"]
            rec = dict(split=split, model=model, spearman_rh_rp=float(spearmanr(rho_h, rho_p)[0]),
                       ratio_rp_rh_median=float(np.median(rho_p / np.maximum(rho_h, 1e-9))))
            for k in ["cov_hourly", "cov_peak", "cov_hourly_on_peak"]:
                rec.update({f"{k}_{s}": v for s, v in summarise_cov(c, k).items()})
            cov_rows.append(rec)
            print(json.dumps(rec), flush=True)
            pd.DataFrame(dict(zone=np.arange(len(rho_h)), r_hourly=rho_h, r_peak=rho_p, cov_hourly=c["cov_hourly"],
                              cov_peak=c["cov_peak"])).to_csv(OUT / f"radius_compare_{split}_{model}.csv", index=False)
            ex_rows += [dict(split=split, model=model, **r) for r in exante(split, model, args.radius_km)]
    pd.DataFrame(cov_rows).to_csv(OUT / "review_coverage.csv", index=False)
    pd.DataFrame(ex_rows).to_csv(OUT / "review_exante.csv", index=False)


if __name__ == "__main__":
    main()
