"""Permutation test of what orders the robust fairness premium across error layouts.

Relative radii eta_i = r_i / d_i are randomly permuted across zones (total absolute radius held fixed; layout k uses
numpy.random.default_rng(1000 + k)). For each layout we solve RE and RF (NE and NF do not depend on radii) and record
  align_corr  = corr(eta_i, w_i), w_i = d_i * max(0, s_i^NF - s_i^NE)   (demand the fairness floor forces into service)
  align_share = share of total radius located in fairness-forced zones (w_i > 0)
  dodged_RE   = share of total radius in demand that the robust efficient plan leaves unserved
  dodgeable   = ex ante dodgeable share, computed from the layout alone (no robust solve)
Row perm = -1 is the observed layout.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import OUT, default_radius_km, distance_matrix
from exp3_siting import demand_and_radius, solve

CTX = {}


def dodgeable_share(rho, r, dhat, q):
    """Ex-ante dodgeable radius: the radius share an efficient plan could shed if the demand share q it leaves
    unserved anyway (U_NE / total demand) were taken from the zones with the largest relative radius."""
    order = np.argsort(-rho)
    cum = np.cumsum(dhat[order]) / dhat.sum()
    frac = np.clip((q - np.r_[0, cum[:-1]]) / (cum - np.r_[0, cum[:-1]]), 0, 1)
    return float((r[order] * frac).sum() / r.sum())


def run_perm(k):
    c = CTX
    if k == -1:
        rho = c["rho"]
    else:
        rho = np.random.default_rng(1000 + k).permutation(c["rho"])
    r = rho * c["dhat"] * c["r0"].sum() / (rho * c["dhat"]).sum()
    U, S, G = {}, {}, {}
    for key, tau in [("RE", 0.0), ("RF", c["tau"])]:
        plan = solve(c["dhat"], r, c["D"], c["R"], c["B"], c["F"], 1.0, c["gamma"], tau, time_limit=180)
        U[key] = np.nan if plan is None else c["dhat"].sum() - plan["served"]
        S[key] = np.full_like(r, np.nan) if plan is None else np.clip(plan["s"], 0, 1)
        G[f"status_{key}"] = -1 if plan is None else int(plan["status"])
        G[f"gap_{key}"] = np.nan if plan is None else float(plan["gap"])
    rfp = (U["RF"] - U["RE"]) - (c["U_NF"] - c["U_NE"])
    rho_eff = r / c["dhat"]
    w = c["w"]
    abandoned = c["s_NE"] < c["tau"]
    return dict(perm=k, RFP=rfp, PoR=U["RE"] - c["U_NE"],
                align_corr=float(np.corrcoef(rho_eff, w)[0, 1]),
                align_share=float(r[w > 0].sum() / r.sum()),
                abandon_share=float(r[abandoned].sum() / r.sum()),
                dodged_RE=float((r * (1 - S["RE"])).sum() / r.sum()),
                dodged_RF=float((r * (1 - S["RF"])).sum() / r.sum()),
                dodge_gap=float((r * (S["RF"] - S["RE"])).sum() / r.sum()),
                dodgeable=dodgeable_share(rho, r, c["dhat"], c["q"]),
                rho_demand_corr=float(spearmanr(rho, c["dhat"])[0]), **G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="A")
    ap.add_argument("--model", default="iTransformer")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--perms", type=int, default=30)
    ap.add_argument("--radius_km", type=float, default=None, help="default: common.default_radius_km()")
    ap.add_argument("--fmult", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    zones, dhat, r0, _ = demand_and_radius(args.split, args.model)
    D = distance_matrix()
    R = args.radius_km or default_radius_km()
    F = args.fmult * np.median(dhat)
    full = solve(dhat, np.zeros_like(dhat), D, R, np.inf, F, 1.0, 0, 1.0, min_cost=True)
    B = args.beta * (F * full["y"].sum() + full["c"].sum())
    ne = solve(dhat, 0 * dhat, D, R, B, F, 1.0, 0, 0.0)
    nf = solve(dhat, 0 * dhat, D, R, B, F, 1.0, 0, args.tau)
    CTX.update(dhat=dhat, r0=r0, rho=r0 / np.maximum(dhat, 1e-6), D=D, R=R, F=F, B=B, gamma=args.gamma, tau=args.tau,
               U_NE=dhat.sum() - ne["served"], U_NF=dhat.sum() - nf["served"],
               w=dhat * np.maximum(0, nf["s"] - ne["s"]), s_NE=ne["s"], q=1 - ne["served"] / dhat.sum())
    path = OUT / f"alignment_{args.split}_{args.model}_g{args.gamma:g}{args.tag}.csv"
    res = []
    for k in [-1] + list(range(args.perms)):
        res.append(run_perm(k))
        print(json.dumps(res[-1]), flush=True)
        pd.DataFrame(res).to_csv(path, index=False)
    df = pd.DataFrame(res)
    perm = df[df.perm >= 0].dropna()
    real = df[df.perm == -1].iloc[0]
    summ = dict(split=args.split, model=args.model, gamma=args.gamma, tau=args.tau, beta=args.beta, n=len(perm),
                spearman_RFP_corr=spearmanr(perm.RFP, perm.align_corr)[0],
                spearman_RFP_share=spearmanr(perm.RFP, perm.align_share)[0],
                spearman_RFP_abandon=spearmanr(perm.RFP, perm.abandon_share)[0],
                spearman_RFP_dodgeRE=spearmanr(perm.RFP, perm.dodged_RE)[0],
                spearman_RFP_dodgegap=spearmanr(perm.RFP, perm.dodge_gap)[0],
                spearman_RFP_dodgeable=spearmanr(perm.RFP, perm.dodgeable)[0],
                spearman_dodgeable_dodgeRE=spearmanr(perm.dodgeable, perm.dodged_RE)[0],
                real_abandon_share=real.abandon_share, real_dodge_gap=real.dodge_gap,
                perm_abandon_share_mean=perm.abandon_share.mean(), perm_dodge_gap_mean=perm.dodge_gap.mean(),
                real_RFP=real.RFP, real_align_corr=real.align_corr, real_align_share=real.align_share,
                perm_RFP_mean=perm.RFP.mean(), perm_RFP_p5=perm.RFP.quantile(.05), perm_RFP_p95=perm.RFP.quantile(.95),
                real_RFP_percentile=float((perm.RFP < real.RFP).mean()),
                share_perm_RFP_positive=float((perm.RFP > 0).mean()))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}), flush=True)


if __name__ == "__main__":
    main()
