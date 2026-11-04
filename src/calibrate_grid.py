"""Pick the number of IEEE-33 feeder copies and base-load scale so that the voltage limit binds on a minority of feeders.

Usage: EVDATA=urbanev python calibrate_grid.py   (needs the split-A iTransformer forecasts)
"""
import itertools

import numpy as np
import pandas as pd

from common import OUT, default_radius_km, distance_matrix, zone_static
from exp3_siting import demand_and_radius, grid_sensitivity, solve

zones, dhat, r0, _ = demand_and_radius("A", "iTransformer")
st = zone_static(zones)
D = distance_matrix()
R = default_radius_km()
F = 2 * np.median(dhat)
full = solve(dhat, 0 * dhat, D, R, np.inf, F, 1.0, 0, 1.0, min_cost=True)
B = 0.7 * (F * full["y"].sum() + full["c"].sum())
plans = {"NE": solve(dhat, 0 * dhat, D, R, B, F, 1.0, 0, 0.0),
         "NF": solve(dhat, 0 * dhat, D, R, B, F, 1.0, 0, 0.5)}
print("NE capacity MW", plans["NE"]["c"].sum() / 1e3, "NF capacity MW", plans["NF"]["c"].sum() / 1e3)
rows = []
for K, bs, mp in itertools.product([30, 50, 80], [0.4, 0.6, 0.8], ["M1", "M2", "M3"]):
    G, rhs, lab, bus_of, vmin0 = grid_sensitivity(zones, st, mp, K, 0.93, bs)
    rec = dict(K=K, base_scale=bs, mapping=mp, base_vmin=round(vmin0, 4))
    for key, p in plans.items():
        ratio = (G @ p["c"]) / np.maximum(rhs, 1e-9)
        per_feeder = ratio.reshape(K, -1).max(1)
        rec[f"{key}_binding_share"] = float((per_feeder > 1).mean())
        rec[f"{key}_max_ratio"] = float(per_feeder.max())
    rows.append(rec)
    print(rec, flush=True)
pd.DataFrame(rows).to_csv(OUT / "grid_calibration.csv", index=False)
