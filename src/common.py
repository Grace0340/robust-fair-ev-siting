"""Shared data loading and chronological splits.

Select the dataset with the environment variable EVDATA:
  EVDATA=urbanev        (default) UrbanEV traffic zones, Shenzhen
  EVDATA=charged_<CITY> CHARGED city after preprocess_charged.py (AMS, JHB, LOA, MEL, SPO, SZH)
The project root defaults to the repository root and can be changed with EV_RFP_ROOT.
Raw data are expected in <root>/data and all outputs are written to <root>/outputs.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("EV_RFP_ROOT", Path(__file__).resolve().parent.parent))
DATASET = os.environ.get("EVDATA", "urbanev")
LOOKBACK, HORIZON = 168, 24

CENTRES = {"SZ": (114.0579, 22.5431), "AMS": (4.9003, 52.3728), "JHB": (28.0473, -26.2041),
           "LOA": (-118.2437, 34.0522), "MEL": (144.9631, -37.8136), "SPO": (-46.6339, -23.5505),
           "SZH": (114.0579, 22.5431)}

if DATASET == "urbanev":
    DATA = ROOT / "data" / "urbanev"
    OUT = ROOT / "outputs"
    CBD = CENTRES["SZ"]
    # The Spring Festival week of 2023 (21-27 Jan) is never an evaluation target. Split A calibrates in the
    # post-reopening COVID-19 wave; split B keeps calibration and test periods inside February 2023.
    SPLITS = {
        "A": dict(fit=("2022-09-01", "2022-11-30"), val=("2022-12-01", "2022-12-20"),
                  cal=("2022-12-21", "2023-01-20"), test=("2023-01-28", "2023-02-28")),
        "B": dict(fit=("2022-09-01", "2022-11-30"), val=("2022-12-01", "2022-12-20"),
                  cal=("2023-02-01", "2023-02-14"), test=("2023-02-15", "2023-02-28")),
    }
else:
    CITY = DATASET.split("_", 1)[1]
    DATA = ROOT / "data" / "proc" / CITY
    OUT = ROOT / "outputs" / DATASET
    CBD = CENTRES[CITY]
    SPLITS = {"C": dict(fit=("2023-04-01", "2023-07-10"), val=("2023-07-11", "2023-07-31"),
                        cal=("2023-08-01", "2023-08-31"), test=("2023-09-01", "2023-09-30"))}
OUT.mkdir(parents=True, exist_ok=True)


def default_radius_km():
    """Coverage radius: 3 km for UrbanEV; for CHARGED the value written by preprocess_charged.py."""
    if DATASET == "urbanev":
        return 3.0
    return float(json.loads((DATA / "meta.json").read_text())["radius_km"])


def load_volume():
    v = pd.read_csv(DATA / "volume.csv", index_col=0, parse_dates=True)
    v.columns = v.columns.astype(int)
    return v


def zone_static(zones):
    if DATASET == "urbanev":
        inf = pd.read_csv(DATA / "inf.csv")
        g = inf.groupby("TAZID").agg(lon=("longitude", "mean"), lat=("latitude", "mean"),
                                     piles=("charge_count", "sum"), stations=("station_id", "count"),
                                     area=("area", "first"))
    else:
        g = pd.read_csv(DATA / "zones.csv", index_col=0)
    g = g.reindex(zones)
    dx = (g["lon"] - CBD[0]) * 111.32 * np.cos(np.deg2rad(CBD[1]))
    dy = (g["lat"] - CBD[1]) * 110.57
    g["dist_cbd_km"] = np.sqrt(dx ** 2 + dy ** 2)
    g["pile_density"] = g["piles"] / (g["area"] / 1e6)
    return g


def distance_matrix():
    """Zone-to-zone distance in metres, ordered like the volume columns."""
    return pd.read_csv(DATA / "distance.csv").to_numpy(dtype=float)


def adjacency(zones):
    if DATASET == "urbanev":
        a = pd.read_csv(DATA / "adj.csv")
        a.columns = a.columns.astype(int)
        a.index = a.columns
        return a.reindex(index=zones, columns=zones).fillna(0).to_numpy(dtype=float)
    D = distance_matrix()
    np.fill_diagonal(D, np.inf)
    k = min(6, len(zones) - 1)
    A = np.zeros_like(D)
    np.put_along_axis(A, np.argsort(D, 1)[:, :k], 1.0, 1)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1)
    return A


def hour_index(index, start, end):
    """Integer positions of hours whose timestamp falls in [start 00:00, end 23:00]."""
    m = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end) + pd.Timedelta(hours=23))
    return np.where(m)[0]


def day_origins(index, start, end):
    """Forecast origins at each midnight in [start, end] that have a full lookback and horizon."""
    pos = hour_index(index, start, end)
    return [p for p in pos if index[p].hour == 0 and p >= LOOKBACK and p + HORIZON <= len(index)]
