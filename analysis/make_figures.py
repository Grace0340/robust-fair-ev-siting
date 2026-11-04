"""Manuscript figures 1-7. Reads outputs/ (after scripts/run_all.sh) and writes outputs/figures/fig*.pdf and .png."""
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import norm, spearmanr

ROOT = Path(os.environ.get("EV_RFP_ROOT", Path(__file__).resolve().parent.parent))
O = ROOT / "outputs"
CH = {k: O / f"charged_{k}" for k in ["SZH", "LOA", "MEL"]}
FIG = O / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False, "lines.linewidth": 1.0,
})

GREY, DARK = "#9AA0A6", "#3C4043"
BLUE, LBLUE = "#3B6FB6", "#A9C4E8"
ORANGE, LORANGE = "#D9822B", "#F2C99B"
RED, GREEN = "#C0392B", "#2E8B57"
PLAN = {"NE": "#B8BEC6", "NF": LBLUE, "RE": "#5F6670", "RF": BLUE}
DATASETS = [("A", "UrbanEV-A"), ("B", "UrbanEV-B"), ("SZH", "CHARGED-SZ"), ("LOA", "CHARGED-LA")]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def label(ax, s, x=-0.18, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def note(ax, text, loc="best", fontsize=5.8):
    """Text box that matplotlib places where it covers the fewest data points."""
    leg = ax.legend([mpl.lines.Line2D([], [], ls="none")], [text], loc=loc, handlelength=0, handletextpad=0,
                    borderpad=0.3, borderaxespad=0.3, fontsize=fontsize, frameon=True, framealpha=0.85,
                    edgecolor="none", facecolor="white")
    ax.add_artist(leg)
    return leg


def headroom(ax, frac=0.3):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def minus(s):
    return s.replace("-", "\u2212")


def alignment(key):
    """Baseline permutation file; UrbanEV uses the 100-layout run."""
    if key in ("A", "B"):
        return pd.read_csv(O / f"alignment_{key}_iTransformer_g2_p100.csv")
    return pd.read_csv(CH[key] / "alignment_C_iTransformer_g2.csv")


def settings_table():
    """One row per UrbanEV permutation setting with hourly radii (model, Gamma, tau, beta, R and F variants)."""
    rows = []
    for f in sorted(O.glob("alignment_[AB]_*.csv")):
        m = re.fullmatch(r"alignment_([AB])_(iTransformer|DLinear)_g([\d.]+)(?:_t([\d.]+)_b([\d.]+))?(?:_([RF])(\d))?\.csv",
                         f.name)
        if not m:
            continue
        split, model, gamma, tau, beta, var, val = m.groups()
        d = pd.read_csv(f)
        p, r = d[d.perm >= 0].dropna(subset=["RFP"]), d[d.perm == -1].iloc[0]
        dF = p.RFP + p.PoR
        lab = (f"{split} {'DL' if model == 'DLinear' else 'iT'} $\\Gamma$={float(gamma):g} $\\tau$={float(tau or 0.5):g} "
               f"$\\beta$={float(beta or 0.7):g}")
        if var:
            lab += f" {'$R$=' + val + ' km' if var == 'R' else '$F$=' + val + '×med'}"
        rows.append(dict(split=split, name=lab, order=(split, model != "iTransformer", var or "", gamma, tau or "0.5",
                                                        beta or "0.7", val or ""),
                         rs_dodged=spearmanr(p.RFP, p.dodged_RE)[0], rs_dodgeable=spearmanr(p.RFP, p.dodgeable)[0],
                         rs_align=spearmanr(p.RFP, p.align_corr)[0], pct=(p.RFP < r.RFP).mean(),
                         cv_fair=dF.std() / dF.mean(), cv_PoR=p.PoR.std() / p.PoR.mean()))
    return pd.DataFrame(rows).sort_values("order").reset_index(drop=True)


def rfp_table(key):
    if key in ("A", "B"):
        d = pd.read_csv(O / f"rfp_{key}_iTransformer.csv")
        return d[(d.gamma == 2) & (d.tau == 0.5) & (d.beta == 0.7)].set_index("mode")
    return pd.read_csv(CH[key] / "rfp_C_iTransformer.csv").set_index("mode")


# ---------------------------------------------------------------- Fig 1: framework and dodging mechanism
def fig1():
    fig = plt.figure(figsize=(190 * MM, 84 * MM))
    ax = fig.add_axes([0, 0.334, 1, 0.594])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=DARK,
                                    lw=0.6))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7, fontweight="bold")
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=6.3, linespacing=1.35)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=DARK))

    box(1, 4, 19, 24, "1  Day-ahead forecasts",
        "Hourly charging energy\nper zone (kWh)\n\nSNaive, DLinear, GRU,\nGraph-GRU, iTransformer\n\n168-h lookback,\n24-h horizon", "#F1F3F4")
    box(23.5, 4, 19, 24, "2  Zone-wise conformal\nradii",
        "\n$r_i$ = $(1-\\alpha)$ split-conformal\nquantile of $|y-\\hat y|$ in zone $i$\n\n"
        "relative radius\n$\\rho_i = r_i/\\bar d_i$", "#F1F3F4")
    box(46, 4, 24, 24, "3  Four siting plans", "budgeted capacitated\ncoverage MILP", "#F1F3F4")
    for k, (lab, x, y) in enumerate([("NE", 49.6, 10.6), ("NF", 58.9, 10.6), ("RE", 49.6, 4.6), ("RF", 58.9, 4.6)]):
        ax.add_patch(Rectangle((x, y), 8.8, 5.4, fc=PLAN[lab], ec=DARK, lw=0.5))
        ax.text(x + 4.4, y + 2.7, lab, ha="center", va="center", fontsize=7, fontweight="bold",
                color="white" if lab in ("RE", "RF") else DARK)
    ax.text(48.9, 13.3, "nominal", rotation=90, ha="right", va="center", fontsize=5.6, color=DARK)
    ax.text(48.9, 7.3, "robust", rotation=90, ha="right", va="center", fontsize=5.6, color=DARK)
    ax.text(54, 16.6, "efficient", ha="center", fontsize=5.8, color=DARK)
    ax.text(63.3, 16.6, "fair ($s_i\\geq\\tau$)", ha="center", fontsize=5.8, color=DARK)
    box(73.5, 4, 25.5, 24, "4  Premium decomposition",
        "PoR = $U_{RE}-U_{NE}$\nPoF = $U_{NF}-U_{NE}$\n"
        "RFP = $(U_{RF}-U_{RE})-(U_{NF}-U_{NE})$\n\n"
        "RFP + PoR = $U_{RF}-U_{NF}$\n(robust cost of the fair plan)", "#F1F3F4")
    for x0, x1 in [(20.6, 23.0), (43.1, 45.5), (70.6, 73.0)]:
        arrow(x0, x1, 16)

    # mechanism panel: three zones, efficient plan abandons the high-error zone, fair plan cannot
    names = ["core\n(high demand,\nsmall $\\rho$)", "mid", "sparse\n(low pile density,\nlarge $\\rho$)"]
    d = np.array([1.0, 0.7, 0.45])
    rho = np.array([0.25, 0.45, 0.9])
    s_eff = np.array([1.0, 1.0, 0.0])
    s_fair = np.array([0.85, 0.75, 0.5])
    for k, (s, title) in enumerate([(s_eff, "Efficient plan: abandons the sparse zone\n(its uncertainty is dodged)"),
                                    (s_fair, "Fair plan: serves every zone at least $\\tau$\n(its uncertainty must be hedged)")]):
        a = fig.add_axes([0.06 + 0.48 * k, 0.046, 0.38, 0.232])
        x = np.arange(3)
        a.bar(x, d * s, 0.55, color=LBLUE if k else "#B8BEC6", lw=0, label="served demand")
        a.bar(x, d * rho * s, 0.55, bottom=d * s, color=ORANGE, alpha=0.8, lw=0,
              label="protected radius (costs capacity)")
        a.bar(x[s == 0], (d * rho)[s == 0], 0.55, bottom=0, color="none", ec=ORANGE, lw=0.8, hatch="////",
              label="dodged radius (costs nothing)")
        a.bar(x, d, 0.55, color="none", ec=DARK, lw=0.6, ls=":", label="forecast demand $\\bar d_i$")
        a.axhline(0, color=DARK, lw=0.4)
        if k:
            for i in x:
                a.plot([i - 0.3, i + 0.3], [0.5 * d[i]] * 2, color=RED, lw=0.8, ls="--")
            a.text(2.36, 0.5 * d[2], "$\\tau\\bar d_i$", color=RED, fontsize=6, va="center")
        a.set_xticks(x, names, fontsize=5.8)
        a.set_ylim(0, 1.35)
        a.set_yticks([])
        a.spines["left"].set_visible(False)
        a.set_title(title, fontsize=6.5, loc="left")
        if k == 0:
            a.legend(loc="upper right", fontsize=5.6, ncol=1, handlelength=1.2, bbox_to_anchor=(1.22, 1.05))
    fig.text(0.005, 0.995, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(0.005, 0.36, "b", fontsize=8, fontweight="bold", va="top")
    save(fig, "fig1_framework")


# ---------------------------------------------------------------- Fig 2: what drives error heterogeneity
def fig2():
    z = pd.read_csv(O / "zone_radii_A.csv")
    g = pd.read_csv(O / "gradient_tests.csv")
    cov = pd.read_csv(O / "coverage_by_group.csv")
    fig = plt.figure(figsize=(160 * MM, 118 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.42, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    rho = z["rho_iTransformer"].clip(upper=np.quantile(z["rho_iTransformer"], 0.98))
    sc = ax.scatter(z.lon, z.lat, c=rho, s=4 + 30 * z.mean_kwh / z.mean_kwh.max(), cmap="cividis_r", lw=0.2,
                    ec="white")
    ax.plot(114.0579, 22.5431, marker="*", color=RED, ms=6, mec="white", mew=0.4, zorder=4)
    ax.annotate("CBD", xy=(114.0579, 22.5431), xytext=(114.2, 22.47), color=RED, fontsize=6, va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.5, shrinkA=1, shrinkB=3), zorder=4)
    ax.set_aspect(1 / np.cos(np.deg2rad(22.6)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cb.set_label("relative radius $\\rho_i$", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_title("UrbanEV Shenzhen (size = demand)", loc="left")
    label(ax, "a", x=-0.2)

    ax = fig.add_subplot(gs[0, 1])
    x = np.log(z.pile_density)
    y = np.log(z.rho_iTransformer.clip(lower=1e-3))
    keep = y > -3
    ax.scatter(x[keep], y[keep], s=5, color=GREY, lw=0)
    b = np.polyfit(x[keep], y[keep], 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, np.polyval(b, xx), color=BLUE, lw=1.2)
    r = spearmanr(x, y)[0]
    txt = minus(f"Spearman {r:.2f} (n = {len(x)})")
    if (~keep).sum():
        txt += f"\n{(~keep).sum()} zones with $\\log\\rho_i<-3$ not shown"
    headroom(ax, 0.15)
    note(ax, txt, fontsize=5.8)
    ax.set_xlabel("log pile density (piles km$^{-2}$)")
    ax.set_ylabel("log $\\rho_i$")
    ax.set_title("Sparser supply, larger relative error", loc="left")
    label(ax, "b", x=-0.2)

    ax = fig.add_subplot(gs[1, 0])
    c = cov[(cov.model == "iTransformer") & (cov.group_by == "pile_density")]
    for split, mk in [("A", "o"), ("B", "s")]:
        cc = c[c.split == split]
        q = np.arange(1, 6)
        ax.plot(q, cc.pooled, marker=mk, ms=3, color=ORANGE, label=f"pooled ({split})")
        ax.plot(q, cc.zone_wise, marker=mk, ms=3, color=BLUE, label=f"zone-wise ({split})")
    ax.axhline(0.9, color=DARK, ls="--", lw=0.6)
    ax.set_xticks(q, ["Q1\nsparse", "Q2", "Q3", "Q4", "Q5\ndense"])
    ax.set_ylabel("test coverage")
    ax.set_ylim(min(c[["pooled", "zone_wise"]].min().min(), 0.9) - 0.02, 1.01)
    ax.legend(loc="upper left", fontsize=5.5, ncol=2, handlelength=1.4)
    ax.set_title("Pooled calibration under-covers sparse zones", loc="left")
    label(ax, "c", x=-0.2)

    ax = fig.add_subplot(gs[1, 1])
    rows = []
    for _, rr in g.iterrows():
        rows.append((f"UrbanEV-{rr.split} {rr.model}", rr))
    for key, name in [("SZH", "CHARGED-SZ"), ("LOA", "CHARGED-LA"), ("MEL", "CHARGED-MEL")]:
        gg = pd.read_csv(CH[key] / "gradient_tests.csv")
        rows.append((f"{name} iTransformer", gg[gg.model == "iTransformer"].iloc[0]))
    rows = [rw for rw in rows if "SNaive" not in rw[0]]
    for i, (name, rr) in enumerate(rows):
        for var, col, off in [("log_pile_density", BLUE, -0.15), ("log_dist_cbd", ORANGE, 0.15)]:
            bta, p = rr[f"beta_{var}"], rr[f"p_{var}"]
            se = abs(bta) / max(norm.ppf(1 - p / 2), 1e-6)
            ax.errorbar(bta, i + off, xerr=1.96 * se, fmt="o", ms=2.5, color=col, lw=0.7, capsize=0,
                        mfc=col if p < 0.05 else "white")
    ax.axvline(0, color=DARK, lw=0.5)
    ax.set_yticks(range(len(rows)), [n.replace("iTransformer", "iTrans.") for n, _ in rows], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("elasticity of $\\rho_i$ (HC3 95% CI)")
    ax.plot([], [], "o", color=BLUE, ms=3, label="pile density")
    ax.plot([], [], "o", color=ORANGE, ms=3, label="distance to CBD")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=5.8, handletextpad=0.2)
    ax.set_title("Drivers across models and cities", loc="left")
    label(ax, "d", x=-0.5)
    save(fig, "fig2_heterogeneity")


# ---------------------------------------------------------------- Fig 3: counterfactual error layouts
def fig3():
    fig, axs = plt.subplots(1, 2, figsize=(140 * MM, 52 * MM), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.35))
    ax = axs[0]
    modes = [("uniform", "uniform", "#D5D8DC"), ("real", "observed", BLUE), ("flipped", "rank-flipped", ORANGE)]
    vals = {}
    for i, (key, name) in enumerate(DATASETS):
        t = rfp_table(key)
        for k, (m, mn, col) in enumerate(modes):
            vals[(i, k)] = t.loc[m, "RFP"] / t.loc[m, "PoF"]
            ax.bar(i + (k - 1) * 0.26, vals[(i, k)], 0.24, color=col, ec=DARK, lw=0.4, label=mn if i == 0 else None)
    ax.axhline(0, color=DARK, lw=0.6)
    ax.set_xticks(range(4), [n.replace("-", "\n") for _, n in DATASETS])
    ax.set_ylabel("RFP / PoF")
    v = np.array([x for x in vals.values() if np.isfinite(x)])
    # bars taller than the 90th percentile are clipped and labelled so the other bars stay readable
    top = max(0.2, 1.15 * np.quantile(v, 0.9))
    for (i, k), x in vals.items():
        if x > top:
            ax.text(i + (k - 1) * 0.26, 0.97 * top, f"{x:.2f}", ha="center", va="top", fontsize=5.5, color="white",
                    rotation=90, fontweight="bold")
    ax.set_ylim(min(-0.5, 1.4 * v.min()), top)
    ax.legend(loc="lower center", ncol=3, handlelength=1, columnspacing=1.0, fontsize=6)
    ax.set_title("Heterogeneity lifts the robust fairness premium", loc="left")
    label(ax, "a", x=-0.14)

    ax = axs[1]
    for key, name in DATASETS[:2]:
        t = rfp_table(key)
        ax.plot([0, 1, 2], [t.loc[m, "PoR"] / t.loc["real", "U_NE"] * 100 for m in ["uniform", "real", "flipped"]],
                marker="o", ms=3, color=BLUE if key == "A" else ORANGE, label=f"{name}: PoR")
        ax.plot([0, 1, 2], [(t.loc[m, "U_RF"] - t.loc[m, "U_NF"]) / t.loc["real", "U_NE"] * 100
                            for m in ["uniform", "real", "flipped"]],
                marker="s", ms=3, ls="--", color=BLUE if key == "A" else ORANGE, label=f"{name}: RFP + PoR")
    ax.set_xticks([0, 1, 2], ["uniform", "observed", "flipped"])
    ax.set_ylabel("% of $U_{NE}$")
    ax.legend(loc="lower left", fontsize=5.5)
    ax.set_title("The efficient plan's robust cost\nmoves; the fair plan's barely does", loc="left")
    label(ax, "b", x=-0.24)
    save(fig, "fig3_counterfactual")


# ---------------------------------------------------------------- Fig 4: permutation test, the dodging rent
def fig4():
    fig, axs = plt.subplots(3, 4, figsize=(190 * MM, 128 * MM), gridspec_kw=dict(hspace=0.62, wspace=0.42))
    for j, (key, name) in enumerate(DATASETS):
        a = alignment(key)
        p, r = a[a.perm >= 0].dropna(subset=["RFP"]), a[a.perm == -1].iloc[0]
        scale = rfp_table(key).loc["real", "PoF"]
        ax = axs[0, j]
        ax.scatter(p.dodged_RE, p.RFP / scale, s=9, color=GREY, lw=0)
        ax.scatter(r.dodged_RE, r.RFP / scale, s=40, marker="*", color=RED, zorder=3)
        rs = spearmanr(p.RFP, p.dodged_RE)[0]
        ra = spearmanr(p.RFP, p.align_corr)[0]
        headroom(ax, 0.5)
        note(ax, minus(f"$r_s$(dodged) = {rs:.2f}\n$r_s$(alignment) = {ra:.2f}\n{len(p)} random layouts"))
        ax.set_xlabel("radius share dodged by RE, $\\delta_{RE}$")
        if j == 0:
            ax.set_ylabel("RFP / PoF")
        ax.set_title(name, loc="left", fontweight="bold")

        ax = axs[1, j]
        ax.scatter(p.dodgeable, p.RFP / scale, s=9, color=GREY, lw=0)
        ax.scatter(r.dodgeable, r.RFP / scale, s=40, marker="*", color=RED, zorder=3)
        headroom(ax)
        note(ax, minus(f"$r_s$ = {spearmanr(p.RFP, p.dodgeable)[0]:.2f}"))
        ax.set_xlabel("ex ante dodgeable share, $\\delta^{ex}$")
        if j == 0:
            ax.set_ylabel("RFP / PoF")

        ax = axs[2, j]
        ax.scatter(p.PoR / scale, p.RFP / scale, s=9, color=GREY, lw=0)
        ax.scatter(r.PoR / scale, r.RFP / scale, s=40, marker="*", color=RED, zorder=3)
        tot = ((p.RFP + p.PoR) / scale).mean()
        xx = np.linspace((p.PoR / scale).min(), (p.PoR / scale).max(), 10)
        ax.plot(xx, tot - xx, color=BLUE, ls="--", lw=0.8)
        cvf = (p.RFP + p.PoR).std() / (p.RFP + p.PoR).mean()
        cvp = p.PoR.std() / p.PoR.mean()
        headroom(ax, 0.45)
        note(ax, f"CV(RFP+PoR) = {cvf:.1%}\nCV(PoR) = {cvp:.1%}")
        ax.set_xlabel("PoR / PoF")
        if j == 0:
            ax.set_ylabel("RFP / PoF")
    fig.legend([mpl.lines.Line2D([], [], ls="none", marker="o", ms=3.5, color=GREY),
                mpl.lines.Line2D([], [], ls="none", marker="*", ms=7, color=RED),
                mpl.lines.Line2D([], [], color=BLUE, ls="--", lw=0.8)],
               ["random layouts", "observed layout", "RFP + PoR = mean (row c)"],
               loc="lower center", bbox_to_anchor=(0.5, 0.925), ncol=3, fontsize=6.5, handletextpad=0.3,
               columnspacing=1.5)
    for i, s in enumerate("abc"):
        label(axs[i, 0], s, x=-0.38)
    save(fig, "fig4_dodging")


# ---------------------------------------------------------------- Fig 5: robustness of the mechanism
def fig5():
    s = settings_table()
    fig, axs = plt.subplots(1, 2, figsize=(150 * MM, 92 * MM), gridspec_kw=dict(width_ratios=[1.4, 1], wspace=0.38))
    ax = axs[0]
    y = np.arange(len(s))
    for yy, a, b in zip(y, s[["rs_dodged", "rs_dodgeable"]].max(axis=1), s.rs_align):
        ax.plot([b, a], [yy, yy], color="#DADCE0", lw=0.8, zorder=1)
    ax.scatter(s.rs_dodged, y, color=BLUE, s=11, label="dodged share $\\delta_{RE}$", zorder=3)
    ax.scatter(s.rs_dodgeable, y, color=GREEN, s=13, marker="s", facecolor="none", lw=0.7,
               label="ex ante dodgeable share $\\delta^{ex}$", zorder=3)
    ax.scatter(s.rs_align, y, color=ORANGE, s=11, marker="D", label="alignment (original hypothesis)", zorder=3)
    ax.axvline(0, color=DARK, lw=0.5)
    ax.set_yticks(y, s.name, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, 1.05)
    ax.set_xlabel("Spearman correlation with RFP (30 layouts each)")
    ax.legend(loc="upper left", fontsize=5.3, bbox_to_anchor=(-0.55, -0.1), ncol=3, columnspacing=0.8,
              handletextpad=0.2)
    ax.set_title("Dodging, not alignment, orders RFP", loc="left")
    label(ax, "a", x=-0.55)

    ax = axs[1]
    for split, col in [("A", BLUE), ("B", ORANGE)]:
        ss = s[s.split == split]
        ax.scatter(ss.cv_PoR * 100, ss.cv_fair * 100, color=col, s=12, label=f"UrbanEV-{split}")
    for key, name in DATASETS[2:]:
        a = alignment(key)
        p = a[a.perm >= 0].dropna(subset=["RFP"])
        f = p.RFP + p.PoR
        ax.scatter(p.PoR.std() / p.PoR.mean() * 100, f.std() / f.mean() * 100, marker="^", s=18,
                   color=DARK if key == "SZH" else GREY, label=name)
    pts = np.concatenate([c.get_offsets() for c in ax.collections])
    lim = [0.5 * pts[pts > 0].min(), 2 * pts.max()]
    ax.plot(lim, lim, color=DARK, lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("CV of PoR across layouts (%)")
    ax.set_ylabel("CV of fair plan's robust cost (%)")
    ax.text(lim[1] / 4, lim[1] / 2.4, "1:1", fontsize=5.5, color=DARK)
    ax.legend(loc="upper left", fontsize=5.5)
    ax.set_title("Fair plan's robust cost is\nlayout-invariant in UrbanEV", loc="left")
    label(ax, "b", x=-0.3)
    save(fig, "fig5_sensitivity")


# ---------------------------------------------------------------- Fig 6: fairness ceiling under feeder limits
def fig6():
    t = pd.read_csv(O / "fair_ceiling_iTransformer.csv")
    fig, axs = plt.subplots(1, 3, figsize=(190 * MM, 55 * MM), gridspec_kw=dict(width_ratios=[1, 1, 0.62], wspace=0.3))
    series = [("nominal", "nominal", "#D5D8DC"), ("real", "robust, observed radii", BLUE),
              ("uniform", "robust, uniform radii", LBLUE)]
    for ax, split in zip(axs[:2], ["A", "B"]):
        tt = t[t.split == split]
        grids = ["none", "M1", "M2", "M3"]
        for k, (m, mn, col) in enumerate(series):
            v = [tt[(tt.grid == g) & (tt["mode"] == m)].tau_star.iloc[0] for g in grids]
            ax.bar(np.arange(4) + (k - 1) * 0.26, v, 0.24, color=col, ec=DARK, lw=0.4, label=mn)
        for i, g in enumerate(grids):
            n = tt[(tt.grid == g) & (tt["mode"] == "nominal")].tau_star.iloc[0]
            rb = tt[(tt.grid == g) & (tt["mode"] == "real")].tau_star.iloc[0]
            ax.text(i, n + 0.03, f"−{n - rb:.2f}", ha="center", fontsize=5.8, color=RED)
        ax.axhline(0.5, color=DARK, ls="--", lw=0.6)
        ax.set_xticks(range(4), ["no grid", "M1", "M2", "M3"])
        ax.set_ylim(0, 0.8)
        ax.set_ylabel("fairness ceiling $\\tau^*$")
        ax.set_title(f"UrbanEV-{split}", loc="left", fontweight="bold")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3, fontsize=6, handlelength=1.2)
    axs[0].text(3.45, 0.51, "$\\tau$ = 0.5", fontsize=5.5, ha="right", va="bottom")
    ax = axs[2]
    for i, (key, name) in enumerate([("LOA", "CHARGED-LA"), ("SZH", "CHARGED-SZ")]):
        tt = pd.read_csv(CH[key] / "fair_ceiling_iTransformer.csv").set_index("mode")
        for k, (m, mn, col) in enumerate(series):
            ax.bar(i + (k - 1) * 0.26, tt.loc[m, "tau_star"], 0.24, color=col, ec=DARK, lw=0.4)
        ax.text(i, tt.loc["nominal", "tau_star"] + 0.03, f"−{tt.loc['nominal', 'tau_star'] - tt.loc['real', 'tau_star']:.2f}",
                ha="center", fontsize=5.8, color=RED)
    ax.set_xticks([0, 1], ["CHARGED\nLA", "CHARGED\nSZ"])
    ax.set_ylim(0, 0.8)
    ax.set_title("No grid layer", loc="left", fontweight="bold")
    for a, l in zip(axs, "abc"):
        label(a, l, x=-0.2 if l != "c" else -0.3)
    save(fig, "fig6_ceiling")


# ---------------------------------------------------------------- Fig 7: ex post performance on test days
def fig7():
    e = pd.concat([pd.read_csv(O / f"rfp_{s}_iTransformer.csv") for s in "AB"])
    e = e[(e["mode"] == "real") & e.RF_ede_unmet.notna()]
    fig, axs = plt.subplots(1, 2, figsize=(140 * MM, 55 * MM), gridspec_kw=dict(wspace=0.38))
    ax = axs[0]
    for plan, mk in [("NE", "o"), ("RE", "o"), ("NF", "s"), ("RF", "s")]:
        ax.scatter(e[f"{plan}_real_unmet_ratio"] * 100, e[f"{plan}_ede_unmet"], s=14, marker=mk, color=PLAN[plan],
                   ec=DARK, lw=0.3, label=plan, zorder=3)
    for _, r in e.iterrows():
        ax.annotate("", xy=(r.RF_real_unmet_ratio * 100, r.RF_ede_unmet),
                    xytext=(r.NF_real_unmet_ratio * 100, r.NF_ede_unmet),
                    arrowprops=dict(arrowstyle="-|>", lw=0.4, color=BLUE, alpha=0.6, mutation_scale=5))
    ax.set_xlabel("realised unmet energy (% of demand)")
    ax.set_ylabel("Kolm–Pollak EDE of unmet ratio\n(lower = fairer)")
    ax.legend(loc="center right", fontsize=5.8, ncol=2, columnspacing=0.6, handletextpad=0.1)
    ax.set_title("Robustness buys fairness, not efficiency", loc="left")
    label(ax, "a", x=-0.26)

    ax = axs[1]
    ax.scatter(e.NF_worst_decile_unmet, e.RF_worst_decile_unmet, s=14, color=BLUE, ec=DARK, lw=0.3,
               label="NF → RF, each setting")
    w = np.concatenate([e.NF_worst_decile_unmet, e.RF_worst_decile_unmet])
    pad = 0.08 * (w.max() - w.min() + 1e-9)
    lim = [w.min() - pad, w.max() + pad]
    ax.plot(lim, lim, color=DARK, lw=0.5, ls=":")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("worst-decile unmet ratio, NF")
    ax.set_ylabel("worst-decile unmet ratio, RF")
    ax.text(lim[0] + 0.7 * (lim[1] - lim[0]), lim[0] + 0.86 * (lim[1] - lim[0]), "1:1", fontsize=5.5, color=DARK)
    ax.legend(loc="upper left", fontsize=5.8)
    ax.set_title("Worst-served zones gain from RF", loc="left")
    label(ax, "b", x=-0.26)
    save(fig, "fig7_expost")


if __name__ == "__main__":
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
        f()
        print("done", f.__name__)
