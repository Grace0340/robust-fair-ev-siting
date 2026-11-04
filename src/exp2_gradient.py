"""Step 2: zone-level conformal radii and tests for a spatial gradient in forecast error.

Outputs (outputs/):
  zone_radii_{split}.csv     seed-averaged relative radius per zone and model + zone covariates
  gradient_tests.csv         Moran's I, regression elasticities, core/periphery ratios per split x model
  coverage_by_group.csv      test coverage of pooled vs zone-wise (Mondrian) calibration by distance quintile
  model_agreement_{split}.csv  Spearman correlation of relative radii across models
"""
import glob
import json
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from common import DATA, OUT, SPLITS, adjacency, distance_matrix, hour_index, load_volume, zone_static

ALPHA = 0.10
RNG = np.random.default_rng(0)


def cq(scores, alpha=ALPHA, axis=0):
    """Split-conformal quantile with finite-sample correction."""
    n = scores.shape[axis]
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, level, axis=axis, method="higher")


def morans_i(x, W, perms=9999):
    z = x - x.mean()
    s0 = W.sum()
    obs = len(x) / s0 * (z @ W @ z) / (z @ z)
    sims = np.empty(perms)
    for k in range(perms):
        zp = RNG.permutation(z)
        sims[k] = len(x) / s0 * (zp @ W @ zp) / (zp @ zp)
    p = (np.sum(np.abs(sims - sims.mean()) >= abs(obs - sims.mean())) + 1) / (perms + 1)
    return obs, p


def main():
    v = load_volume()
    zones = v.columns.to_numpy()
    st = zone_static(zones)
    occ = None
    if (DATA / "occupancy.csv").exists():
        occ = pd.read_csv(DATA / "occupancy.csv", index_col=0, parse_dates=True)
        occ.columns = occ.columns.astype(int)
        occ = occ[zones]
    A = adjacency(zones)
    np.fill_diagonal(A, 0)
    W = A / np.maximum(A.sum(1, keepdims=True), 1)
    isolated = int((A.sum(1) == 0).sum())
    Dm = distance_matrix()
    np.fill_diagonal(Dm, np.inf)
    k = min(6, len(zones) - 1)
    Wknn = np.zeros_like(Dm)
    np.put_along_axis(Wknn, np.argsort(Dm, 1)[:, :k], 1.0, 1)
    Wknn = Wknn / k

    files = sorted(glob.glob(str(OUT / "preds_*_s*.npz")))
    meta = [re.match(r".*preds_(\w)_(\w+)_s(\d+)\.npz", f).groups() for f in files]
    tests, cover = [], []
    for split in sorted({m[0] for m in meta}):
        fit = hour_index(v.index, *SPLITS[split]["fit"])
        scale = v.iloc[fit].mean().to_numpy()
        sat = (np.zeros(len(zones)) if occ is None
               else (occ.iloc[fit].to_numpy() >= 0.9 * st["piles"].to_numpy()).mean(0))
        cov = st.assign(mean_kwh=scale, sat_share=sat)
        rho_tab = {}
        for model in sorted({m[1] for m in meta if m[0] == split}):
            rhos, covs = [], []
            for f, (s, mm, seed) in zip(files, meta):
                if s != split or mm != model:
                    continue
                d = np.load(f)
                ecal = np.abs(d["cal_true"] - d["cal_pred"]).reshape(-1, len(zones))
                etest = np.abs(d["test_true"] - d["test_pred"]).reshape(-1, len(zones))
                r_zone = cq(ecal)                                  # Mondrian by zone
                q_pool = cq((ecal / scale).ravel())                # pooled relative score
                rhos.append(r_zone / scale)
                covs.append(pd.DataFrame({"zone_wise": (etest <= r_zone).mean(0),
                                          "pooled": (etest <= q_pool * scale).mean(0)}))
            rho = np.mean(rhos, 0)
            rho_tab[model] = rho
            cv = pd.concat(covs).groupby(level=0).mean()
            for gname, col in [("dist_cbd", "dist_cbd_km"), ("pile_density", "pile_density")]:
                q = pd.qcut(cov[col].to_numpy(), 5, labels=[f"Q{i}" for i in range(1, 6)])
                g = cv.assign(q=q).groupby("q", observed=True).mean()
                for dq, row in g.iterrows():
                    cover.append(dict(split=split, model=model, group_by=gname, quintile=dq, **row.to_dict()))

            lr = np.log(np.maximum(rho, 1e-3))
            I, p = morans_i(lr, W)
            I6, p6 = morans_i(lr, Wknn)
            X = pd.DataFrame({"log_dist_cbd": np.log(cov["dist_cbd_km"].to_numpy() + 0.5),
                              "log_pile_density": np.log(cov["pile_density"].to_numpy()),
                              "log_mean_kwh": np.log(np.maximum(scale, 1e-3)), "sat_share": sat})
            if occ is None:
                X = X.drop(columns="sat_share")
            ols = sm.OLS(lr, sm.add_constant(X)).fit(cov_type="HC3")
            ols_d = sm.OLS(lr, sm.add_constant(X[["log_dist_cbd"]])).fit(cov_type="HC3")
            core = cov["dist_cbd_km"] <= cov["dist_cbd_km"].quantile(0.2)
            peri = cov["dist_cbd_km"] >= cov["dist_cbd_km"].quantile(0.8)
            rec = dict(split=split, model=model, n_seeds=len(rhos), rho_median=float(np.median(rho)),
                       rho_p10=float(np.quantile(rho, .1)), rho_p90=float(np.quantile(rho, .9)),
                       moran_I=I, moran_p=p, moran_I_knn6=I6, moran_p_knn6=p6, periphery_core_ratio=float(rho[peri].mean() / rho[core].mean()),
                       dist_only_beta=ols_d.params["log_dist_cbd"], dist_only_p=ols_d.pvalues["log_dist_cbd"],
                       r2_full=ols.rsquared)
            for k in X.columns:
                rec[f"beta_{k}"] = ols.params[k]
                rec[f"p_{k}"] = ols.pvalues[k]
            tests.append(rec)
            print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items()}), flush=True)

        tab = cov.copy()
        for mname, r in rho_tab.items():
            tab[f"rho_{mname}"] = r
        tab.to_csv(OUT / f"zone_radii_{split}.csv")
        names = list(rho_tab)
        agree = pd.DataFrame([[spearmanr(rho_tab[a], rho_tab[b])[0] for b in names] for a in names],
                             index=names, columns=names)
        agree.to_csv(OUT / f"model_agreement_{split}.csv")
        print(f"split {split} Spearman agreement:\n{agree.round(2)}")
    pd.DataFrame(tests).to_csv(OUT / "gradient_tests.csv", index=False)
    pd.DataFrame(cover).to_csv(OUT / "coverage_by_group.csv", index=False)
    print("isolated zones in adjacency:", isolated)


if __name__ == "__main__":
    main()
