"""Highest fairness floor tau* that a budget- and grid-feasible plan can guarantee, nominal vs robust.

tau* = max{tau : some plan within budget (and feeder voltage limits) serves at least tau of every zone}, found by
bisection on feasibility MILPs. If a candidate is neither solved nor proven infeasible within the time limit it is
treated as infeasible and counted in `unresolved`, so tau* is a lower bound in that case.
The robust ceiling drop  tau*_N - tau*_R  is the fairness that uncertainty protection costs.
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import OUT, default_radius_km, distance_matrix, zone_static
from exp3_siting import LAST, demand_and_radius, grid_sensitivity, radius_variant, solve


def ceiling(feasible, tol=0.02):
    lo, hi, unresolved = 0.0, 1.0, 0
    if feasible(1.0)[0]:
        return 1.0, 0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        ok, timed_out = feasible(mid)
        unresolved += timed_out
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["A", "B"])
    ap.add_argument("--model", default="iTransformer")
    ap.add_argument("--radius_km", type=float, default=None, help="default: common.default_radius_km()")
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--grids", nargs="+", default=["none", "M1", "M2", "M3"])
    ap.add_argument("--modes", nargs="+", default=["real", "uniform"])
    ap.add_argument("--time_limit", type=float, default=120)
    ap.add_argument("--vmin", type=float, default=0.93)
    ap.add_argument("--base_scale", type=float, default=0.4)
    ap.add_argument("--feeders", type=int, default=50)
    ap.add_argument("--fmult", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.radius_km = args.radius_km or default_radius_km()
    out = []
    for split in args.splits:
        zones, dhat, r0, _ = demand_and_radius(split, args.model)
        st = zone_static(zones)
        D = distance_matrix()
        F = args.fmult * np.median(dhat)
        full = solve(dhat, np.zeros_like(dhat), D, args.radius_km, np.inf, F, 1.0, 0, 1.0, min_cost=True)
        B = args.beta * (F * full["y"].sum() + full["c"].sum())
        for g in args.grids:
            grid = None if g == "none" else tuple(grid_sensitivity(zones, st, g, args.feeders, args.vmin,
                                                                   args.base_scale)[:2])
            for mode in ["nominal"] + args.modes:
                r = 0 * dhat if mode == "nominal" else radius_variant(dhat, r0, mode)
                gam = 0 if mode == "nominal" else args.gamma

                def feasible(tau):
                    plan = solve(dhat, r, D, args.radius_km, B, F, 1.0, gam, tau, grid,
                                 time_limit=args.time_limit, feas_only=True)
                    return plan is not None, int(plan is None and LAST["status"] == 1)

                tau_star, unresolved = ceiling(feasible)
                rec = dict(split=split, grid=g, mode=mode, tau_star=tau_star, unresolved=unresolved)
                print(json.dumps(rec), flush=True)
                out.append(rec)
                pd.DataFrame(out).to_csv(OUT / f"fair_ceiling_{args.model}{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
