"""Turn CHARGED *_remove_zero site data into zone-level inputs comparable with UrbanEV.

Sites with flat-lined, creeping or intermittent series (see diag_sites.py) are dropped first; cities left with fewer
than MIN_SITES sites are skipped. Cities with more than 300 remaining sites are aggregated into square grid cells
whose size gives an occupied-cell count closest to TARGET_ZONES (the size of the UrbanEV zone system); smaller
cities keep sites as zones. The coverage radius is the median distance from a zone to its tenth-nearest neighbour.
Writes data/proc/<CITY>/{volume.csv, zones.csv, distance.csv, meta.json}.
"""
import json

import numpy as np
import pandas as pd

from common import ROOT
from diag_sites import site_stats

CITIES = ["AMS", "JHB", "LOA", "MEL", "SPO", "SZH"]
TARGET_ZONES = 275
MIN_SITES = 30


def haversine_m(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    for city in CITIES:
        src = ROOT / "data" / "charged" / city
        dst = ROOT / "data" / "proc" / city
        dst.mkdir(parents=True, exist_ok=True)
        s = pd.read_csv(src / "sites.csv")
        s = s.rename(columns={s.columns[0]: "site"})
        v = pd.read_csv(src / "volume.csv", index_col=0, parse_dates=True)
        v.columns = v.columns.astype(int)
        q = site_stats(v)
        good = q.index[(q.cv >= 0.15) & (q.flat < 0.2) & (q.creep < 0.3) & (q.active >= 0.8)]
        energy_kept = float(q["mean"][good].sum() / q["mean"].sum())
        v = v[good]
        s = s[s["site"].isin(v.columns)].reset_index(drop=True)
        if len(s) < MIN_SITES:
            print(json.dumps(dict(city=city, skipped=True, good_sites=len(s), energy_kept=round(energy_kept, 3))),
                  flush=True)
            continue
        lat0 = s["latitude"].mean()
        x = (s["longitude"] - s["longitude"].min()) * 111.32 * np.cos(np.deg2rad(lat0))
        y = (s["latitude"] - s["latitude"].min()) * 110.57
        if len(s) > 300:
            best = None
            for size in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]:
                cells = (np.floor(x / size).astype(int).astype(str) + "_" + np.floor(y / size).astype(int).astype(str))
                n = cells.nunique()
                if best is None or abs(n - TARGET_ZONES) < abs(best[1] - TARGET_ZONES):
                    best = (size, n, cells)
            size, _, cells = best
            s["zone_key"] = cells
            mode = f"grid_{size}km"
        else:
            size = None
            s["zone_key"] = s["site"].astype(str)
            mode = "site"
        keys = sorted(s["zone_key"].unique())
        zid = {k: i for i, k in enumerate(keys)}
        s["zone"] = s["zone_key"].map(zid)
        w = s["charger_num"].clip(lower=1)
        z = s.assign(wlon=s["longitude"] * w, wlat=s["latitude"] * w, w=w).groupby("zone").agg(
            wlon=("wlon", "sum"), wlat=("wlat", "sum"), w=("w", "sum"), piles=("charger_num", "sum"),
            stations=("site", "count"))
        z["lon"], z["lat"] = z["wlon"] / z["w"], z["wlat"] / z["w"]
        D = haversine_m(z["lon"].to_numpy(), z["lat"].to_numpy())
        if size is not None:
            z["area"] = (size * 1000) ** 2
        else:
            Dn = D.copy()
            np.fill_diagonal(Dn, np.inf)
            rad = np.maximum(Dn.min(1) / 2, 250.0)
            z["area"] = np.pi * rad ** 2
        vz = v.T.groupby(s.set_index("site")["zone"].reindex(v.columns).to_numpy()).sum().T
        vz = vz[sorted(vz.columns)]
        vz.index.name = "time"
        vz.to_csv(dst / "volume.csv")
        z[["lon", "lat", "piles", "stations", "area"]].to_csv(dst / "zones.csv")
        pd.DataFrame(D, columns=vz.columns).to_csv(dst / "distance.csv", index=False)
        Dn = D.copy()
        np.fill_diagonal(Dn, np.inf)
        k = min(10, len(z) - 1)
        radius_km = float(np.median(np.sort(Dn, 1)[:, k - 1]) / 1000)
        meta = dict(city=city, mode=mode, sites=len(s), energy_kept=round(energy_kept, 3), zones=len(z),
                    radius_km=round(radius_km, 2),
                    hours=len(vz), mean_kwh_h=float(vz.values.mean()))
        (dst / "meta.json").write_text(json.dumps(meta))
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
