"""Print the tables and statistics reported in the paper, computed from outputs/ (run after scripts/run_all.sh)."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(os.environ.get("EV_RFP_ROOT", Path(__file__).resolve().parent.parent))
O = ROOT / "outputs"
LOG = O / "logs"
CH = {"SZ": O / "charged_SZH", "LA": O / "charged_LOA", "MEL": O / "charged_MEL"}
TOL = 200.0  # kW, tolerance for the ordering checks of Proposition 1
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def section(title):
    print(f"\n=== {title}")


def perm_summary(path):
    d = pd.read_csv(path)
    p = d[d.perm >= 0].dropna(subset=["RFP"])
    real = d[d.perm == -1].iloc[0]
    out = dict(n=len(p), n_all=int((d.perm >= 0).sum()), real_feasible=bool(np.isfinite(real.RFP)))
    if len(p) < 3:
        return out
    dF = p.RFP + p.PoR
    out.update(rs_dodged=spearmanr(p.RFP, p.dodged_RE)[0], rs_align=spearmanr(p.RFP, p.align_corr)[0],
               rs_PoR=spearmanr(p.RFP, p.PoR)[0], pct_real=(p.RFP < real.RFP).mean(),
               cv_fair=dF.std() / dF.mean(), cv_PoR=p.PoR.std() / p.PoR.mean(), real_RFP=real.RFP,
               real_dodged=real.dodged_RE, perm_dodged=p.dodged_RE.mean())
    if "dodgeable" in p:
        out.update(rs_dodgeable=spearmanr(p.RFP, p.dodgeable)[0], real_dodgeable=real.dodgeable,
                   perm_dodgeable=p.dodgeable.mean())
    return out


def permutation_tables():
    section("Permutation tests, hourly radii (Fig. 4, Fig. 5 and Section 5.5)")
    files = {"baseline": {"A": O / "alignment_A_iTransformer_g2.csv", "B": O / "alignment_B_iTransformer_g2.csv",
                          "SZ": CH["SZ"] / "alignment_C_iTransformer_g2.csv",
                          "LA": CH["LA"] / "alignment_C_iTransformer_g2.csv"},
             "p100": {s: O / f"alignment_{s}_iTransformer_g2_p100.csv" for s in "AB"}}
    rows = {}
    for f in sorted(O.glob("alignment_[AB]_*.csv")):
        m = re.fullmatch(r"alignment_([AB])_(iTransformer|DLinear)_g([\d.]+)(_t[\d.]+_b[\d.]+|_[RF]\d)?\.csv", f.name)
        if m:
            rows[(f"{m[2]} g{m[3]}{m[4] or ''}", m[1])] = perm_summary(f)
    for tag, fs in files.items():
        for k, f in fs.items():
            if f.exists():
                rows[(tag, k)] = perm_summary(f)
    s = pd.DataFrame(rows).T
    print(s.round(3).to_string())
    grid = s.drop(index=[i for i in s.index if i[0] in ("baseline", "p100")], errors="ignore")
    if len(grid):
        print(f"\nUrbanEV settings: {len(grid)}; Spearman with dodged share "
              f"[{grid.rs_dodged.min():.2f}, {grid.rs_dodged.max():.2f}], with dodgeable share "
              f"[{grid.rs_dodgeable.min():.2f}, {grid.rs_dodgeable.max():.2f}], with alignment "
              f"[{grid.rs_align.min():.2f}, {grid.rs_align.max():.2f}]; observed layout at or below the 10th "
              f"percentile in {int((grid.pct_real <= 0.1).sum())} settings")

    section("Bootstrap 95% intervals of the Spearman correlations (2,000 resamples)")
    rng = np.random.default_rng(0)
    for name, f in [("UrbanEV-A", files["p100"]["A"]), ("UrbanEV-B", files["p100"]["B"]),
                    ("CHARGED-SZ", files["baseline"]["SZ"]), ("CHARGED-LA", files["baseline"]["LA"])]:
        if not f.exists():
            continue
        p = pd.read_csv(f)
        p = p[p.perm >= 0].dropna(subset=["RFP"]).reset_index(drop=True)
        out = [name, f"n={len(p)}"]
        for col in ["dodged_RE", "dodgeable", "align_corr"]:
            bs = [spearmanr(*p.loc[rng.integers(0, len(p), len(p)), ["RFP", col]].values.T)[0] for _ in range(2000)]
            out.append(f"{col} {spearmanr(p.RFP, p[col])[0]:.3f} "
                       f"[{np.nanquantile(bs, .025):.3f}, {np.nanquantile(bs, .975):.3f}]")
        print(*out, sep=" | ")


def peak_and_model_tables():
    section("Peak-consistent radii (Section 5.7)")
    rows = {}
    for key, d in [("A", O), ("B", O), ("SZ", CH["SZ"]), ("LA", CH["LA"])]:
        split = key if key in "AB" else "C"
        for f in sorted(d.glob(f"alignment_{split}_iTransformer_g2_peak_t*.csv")):
            rows[(key, f.stem.rsplit("_t", 1)[1])] = perm_summary(f)
    if rows:
        print(pd.DataFrame(rows).T.round(3).to_string())

    section("Weaker forecasters (Section 5.5, forecast quality)")
    rows = {}
    for m in ["SNaive", "GRU", "GraphGRU"]:
        for s in "AB":
            f = O / f"alignment_{s}_{m}_g2_t0.4.csv"
            if f.exists():
                rows[(m, s)] = perm_summary(f)
    if rows:
        print(pd.DataFrame(rows).T.round(3).to_string())


def plan_tables():
    section("Four-plan tables: observed, uniform and rank-flipped radii")
    files = sorted(O.glob("rfp_*.csv")) + sorted(O.glob("charged_*/rfp_*.csv"))
    for f in files:
        if "_timing" in f.name:
            continue
        d = pd.read_csv(f)
        cols = [c for c in ["grid", "gamma", "tau", "beta", "mode", "U_NE", "PoR", "PoF", "RFP", "RFP_rel"] if c in d]
        print(f"-- {f.relative_to(O)}")
        print(d[cols].round(3).to_string(index=False))
        st = [c for c in d.columns if c.endswith("_status")]
        gp = [c for c in d.columns if c.endswith("_gap")]
        if gp:
            print("   max gap %.2e, time-limit hits %d of %d"
                  % (d[gp].max().max(), int((d[st] == 1).sum().sum()), int(d[st].notna().sum().sum())))

    section("Optimality-gap error bound on the RFP (baseline UrbanEV runs)")
    for s in "AB":
        log, f = LOG / f"exp3_{s}.txt", O / f"rfp_{s}_iTransformer.csv"
        if not (log.exists() and f.exists()):
            continue
        tot = json.loads(log.read_text().splitlines()[0])["total_peak_kw"]
        d = pd.read_csv(f)
        err = sum(d[f"{k}_gap"] * (tot - d[f"U_{k}"]) for k in ["NE", "NF", "RE", "RF"])
        print(s, "max RFP error bound %.2f%% of total demand" % (100 * err.max() / tot))


def expost_table():
    section("Ex post performance on test days (Fig. 7, Section 5.8)")
    e = pd.concat([pd.read_csv(f) for f in [O / "rfp_A_iTransformer.csv", O / "rfp_B_iTransformer.csv"] if f.exists()])
    cols = ["split", "gamma", "tau", "mode", "RFP"] + [f"{k}_{m}" for k in ["NE", "NF", "RE", "RF"]
                                                       for m in ["real_unmet_ratio", "ede_unmet", "worst_decile_unmet"]]
    e = e[[c for c in cols if c in e]].sort_values(["split", "gamma", "tau", "mode"])
    e.to_csv(O / "expost_summary.csv", index=False)
    e = e[(e["mode"] == "real") & e.RF_ede_unmet.notna()]
    print("settings with a feasible robust fair plan:", len(e), e.split.value_counts().to_dict())
    for p in ["NE", "NF", "RE", "RF"]:
        print(p, "unmet %% [%.1f, %.1f]" % (100 * e[f"{p}_real_unmet_ratio"].min(), 100 * e[f"{p}_real_unmet_ratio"].max()),
              "EDE [%.3f, %.3f]" % (e[f"{p}_ede_unmet"].min(), e[f"{p}_ede_unmet"].max()),
              "worst decile [%.3f, %.3f]" % (e[f"{p}_worst_decile_unmet"].min(), e[f"{p}_worst_decile_unmet"].max()))
    rel = 1 - e.RF_ede_unmet / e.NF_ede_unmet
    print("RF EDE below NF EDE in every setting:", bool((e.RF_ede_unmet < e.NF_ede_unmet).all()),
          "| EDE reduction NF->RF [%.1f, %.1f]%%, median %.1f%%" % (100 * rel.min(), 100 * rel.max(), 100 * rel.median()))
    print("RF worst decile below NF in every setting:", bool((e.RF_worst_decile_unmet < e.NF_worst_decile_unmet).all()))


def ceiling_tables():
    section("Fairness ceilings tau* (Section 5.6)")
    for f in sorted(O.glob("fair_ceiling_*.csv")) + sorted(O.glob("charged_*/fair_ceiling_*.csv")):
        d = pd.read_csv(f)
        pv = d.pivot_table(index=["split", "grid"], columns="mode", values="tau_star")
        if {"nominal", "real"} <= set(pv.columns):
            pv["cut_pct"] = 100 * (1 - pv["real"] / pv["nominal"])
        print(f"-- {f.relative_to(O)}  (unresolved checks: {int(d.unresolved.sum())})")
        print(pv.round(3).to_string())


def diagnostics_tables():
    section("Radius coverage and ex ante dodgeable share")
    for d in [O, CH["SZ"], CH["LA"]]:
        for name in ["review_coverage.csv", "review_exante.csv"]:
            if (d / name).exists():
                print(f"-- {(d / name).relative_to(O)}")
                print(pd.read_csv(d / name).round(3).to_string(index=False))
    for name in ["gradient_tests.csv", "coverage_by_group.csv"]:
        for d in [O, CH["SZ"], CH["LA"], CH["MEL"]]:
            if (d / name).exists():
                print(f"-- {(d / name).relative_to(O)}")
                print(pd.read_csv(d / name).round(3).to_string(index=False))


def solver_checks():
    section("Solver audit: ordering checks of Proposition 1 in the permutation runs")
    n_solves = n_tl = n_por = n_df = 0
    for f in sorted(O.glob("alignment_*.csv")) + sorted(O.glob("charged_*/alignment_*.csv")):
        d = pd.read_csv(f)
        if "status_RE" not in d:
            continue
        st = pd.concat([d.status_RE, d.status_RF]).dropna()
        ok = d.dropna(subset=["RFP"])
        n_solves += len(st)
        n_tl += int((st == 1).sum())
        n_por += int((ok.PoR < -TOL).sum())
        n_df += int((ok.RFP + ok.PoR < -TOL).sum())
    print(f"robust solves {n_solves}, time-limit hits {n_tl}, layouts with PoR < 0: {n_por}, with Delta_F < 0: {n_df}")

    section("Tightness of the dodging bound (Proposition 2)")
    for f in sorted(LOG.glob("bound_*.txt")):
        tight = [line for line in f.read_text().splitlines() if "tight_real" in line]
        print(f.name, tight[-1] if tight else "(no result)")
    if (O / "timing_summary.txt").exists():
        section("Wall time")
        print((O / "timing_summary.txt").read_text())


if __name__ == "__main__":
    for step in [permutation_tables, peak_and_model_tables, plan_tables, expost_table, ceiling_tables,
                 diagnostics_tables, solver_checks]:
        try:
            step()
        except FileNotFoundError as err:
            print(f"(skipped {step.__name__}: {err})")
