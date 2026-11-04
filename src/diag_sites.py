"""Site-level quality diagnostics for CHARGED volume series (used by preprocess_charged.py)."""
import numpy as np
import pandas as pd

from common import ROOT

CITIES = ["AMS", "SZH", "LOA", "MEL", "SPO", "JHB"]


def site_stats(v):
    x = v.to_numpy()
    days = x[: len(x) // 24 * 24].reshape(-1, 24, x.shape[1])
    dm, ds = days.mean(1), days.std(1)
    cv = np.where(dm > 0, ds / np.maximum(dm, 1e-9), np.nan)
    d = np.diff(x, axis=0)
    flat = ((d == 0) & (x[1:] > 0)).mean(0)
    up = ((d > 0) & (d < 0.05 * np.maximum(x[1:], 1e-9))).mean(0)
    active_days = (dm > 0).mean(0)
    first = np.argmax(dm > 0, axis=0)
    return pd.DataFrame(dict(cv=np.nanmedian(cv, 0), flat=flat, creep=up, active=active_days, first_day=first,
                             mean=x.mean(0)), index=v.columns)


def main():
    for city in CITIES:
        v = pd.read_csv(ROOT / "data" / "charged" / city / "volume.csv", index_col=0, parse_dates=True)
        st = site_stats(v)
        print(f"== {city} sites={len(st)}")
        print(st.quantile([.1, .25, .5, .75, .9]).round(3).to_string())
        for name, cond in [("cv<0.15", st.cv < 0.15), ("flat>=0.2", st.flat >= 0.2), ("creep>=0.3", st.creep >= 0.3),
                           ("active<0.8", st.active < 0.8)]:
            print(f"  {name}: n={cond.sum()} energy_share={st['mean'][cond].sum() / st['mean'].sum():.3f}")
        st.to_csv(ROOT / "outputs" / f"site_quality_{city}.csv")


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    main()
