"""Tightness of the dodging bound (Proposition 2) on every layout of the baseline permutation file.

bound_full = sum_j L_j pi_j / (L_j + pi_j),  pi_j = top-Gamma sum of r_i a0_ij;  bound_simple = sum_i r_i s0_i.
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import OUT, default_radius_km, distance_matrix
from exp3_siting import demand_and_radius, solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="A")
    ap.add_argument("--model", default="iTransformer")
    ap.add_argument("--radius_km", type=float, default=None, help="default: common.default_radius_km()")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--file", default=None, help="permutation file in outputs/ (default: baseline file)")
    args = ap.parse_args()
    args.radius_km = args.radius_km or default_radius_km()
    zones, dhat, r0, _ = demand_and_radius(args.split, args.model)
    D = distance_matrix()
    F = 2 * np.median(dhat)
    full = solve(dhat, np.zeros_like(dhat), D, args.radius_km, np.inf, F, 1.0, 0, 1.0, min_cost=True)
    B = 0.7 * (F * full["y"].sum() + full["c"].sum())
    ne = solve(dhat, 0 * dhat, D, args.radius_km, B, F, 1.0, 0, 0.0)
    a0 = ne["a"].tocsc()
    L = np.asarray(a0.T @ dhat).ravel()
    rho = r0 / np.maximum(dhat, 1e-6)
    path = args.file or f"alignment_{args.split}_{args.model}_g{args.gamma:g}.csv"
    df = pd.read_csv(OUT / path)
    g = int(args.gamma)
    full_b, simple_b = [], []
    for k in df.perm:
        rk = rho if k == -1 else np.random.default_rng(1000 + int(k)).permutation(rho)
        r = rk * dhat * r0.sum() / (rk * dhat).sum()
        pi = np.zeros(len(dhat))
        for j in np.where(L > 0)[0]:
            col = a0[:, j].toarray().ravel() * r
            pi[j] = np.sort(col)[::-1][:g].sum()
        full_b.append(float((L * pi / np.maximum(L + pi, 1e-9)).sum()))
        simple_b.append(float((r * ne["s"]).sum()))
    df["bound_full"], df["bound_simple"] = full_b, simple_b
    df["tight_full"] = df.PoR / df.bound_full
    df.to_csv(OUT / f"bound_{path}", index=False)
    p = df[df.perm >= 0]
    rec = dict(split=args.split, file=path, real_PoR=float(df.PoR.iloc[0]), real_bound_full=full_b[0],
               real_bound_simple=simple_b[0], tight_real=float(df.tight_full.iloc[0]),
               tight_perm_median=float(p.tight_full.median()), tight_perm_min=float(p.tight_full.min()),
               tight_perm_max=float(p.tight_full.max()), violations=int((df.PoR > df.bound_full + 1e-6).sum()))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
