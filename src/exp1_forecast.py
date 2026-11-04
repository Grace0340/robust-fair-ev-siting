"""Step 1: day-ahead zone-level charging-volume forecasts with several model families.

Saves calibration/test predictions (daily origins at 00:00, 24-h horizon) in kWh for every
model, split and seed to outputs/preds_{split}_{model}_s{seed}.npz.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import HORIZON, LOOKBACK, OUT, SPLITS, adjacency, day_origins, hour_index, load_volume

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def time_feats(index):
    h, d = index.hour.to_numpy(), index.dayofweek.to_numpy()
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)], 1).astype(np.float32)


class DLinear(nn.Module):
    def __init__(self, n, k=25):
        super().__init__()
        self.pool = nn.AvgPool1d(k, stride=1, padding=k // 2, count_include_pad=False)
        self.lin_s, self.lin_t = nn.Linear(LOOKBACK, HORIZON), nn.Linear(LOOKBACK, HORIZON)

    def forward(self, x, xm):
        z = x.transpose(1, 2)                       # B,N,L
        trend = self.pool(z)
        return (self.lin_s(z - trend) + self.lin_t(trend)).transpose(1, 2)


class GRUNet(nn.Module):
    """Channel-independent GRU; optional neighbour-mean input and node embedding (graph variant)."""

    def __init__(self, n, adj=None, hid=64, emb=8):
        super().__init__()
        self.graph = adj is not None
        if self.graph:
            a = adj.copy()
            np.fill_diagonal(a, 0)
            a = a / np.maximum(a.sum(1, keepdims=True), 1)
            self.register_buffer("A", torch.tensor(a, dtype=torch.float32))
            self.node = nn.Parameter(torch.randn(n, emb) * 0.1)
        fin = 1 + 4 + (1 + emb if self.graph else 0)
        self.gru = nn.GRU(fin, hid, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hid, HORIZON)

    def forward(self, x, xm):
        b, l, n = x.shape
        feats = [x.unsqueeze(-1), xm.unsqueeze(2).expand(b, l, n, 4)]
        if self.graph:
            feats.append(torch.einsum("ij,blj->bli", self.A, x).unsqueeze(-1))
            feats.append(self.node.view(1, 1, n, -1).expand(b, l, n, -1))
        z = torch.cat(feats, -1).permute(0, 2, 1, 3).reshape(b * n, l, -1)
        out, _ = self.gru(z)
        return self.head(out[:, -1]).view(b, n, HORIZON).transpose(1, 2)


class ITransformer(nn.Module):
    def __init__(self, n, d=128, layers=2, heads=8):
        super().__init__()
        self.emb = nn.Linear(LOOKBACK, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.proj = nn.Linear(d, HORIZON)

    def forward(self, x, xm):
        mu, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-5
        z = ((x - mu) / sd).transpose(1, 2)         # B,N,L  (variates as tokens)
        y = self.proj(self.enc(self.emb(z))).transpose(1, 2)
        return y * sd + mu


def build(name, n, adj):
    return {"DLinear": lambda: DLinear(n), "GRU": lambda: GRUNet(n),
            "GraphGRU": lambda: GRUNet(n, adj), "iTransformer": lambda: ITransformer(n)}[name]()


def windows(Z, TM, origins):
    x = np.stack([Z[p - LOOKBACK:p] for p in origins])
    xm = np.stack([TM[p - LOOKBACK:p] for p in origins])
    y = np.stack([Z[p:p + HORIZON] for p in origins])
    return [torch.tensor(a) for a in (x, xm, y)]


def predict(model, Z, TM, origins, bs=16):
    model.eval()
    x, xm, _ = windows(Z, TM, origins)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            outs.append(model(x[i:i + bs].to(DEV), xm[i:i + bs].to(DEV)).cpu().numpy())
    return np.concatenate(outs)


def train(name, Z, TM, adj, sp, seed, epochs=60, patience=8, bs=32, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = load_volume.index_cache
    fit_pos = hour_index(idx, *sp["fit"])
    fit_orig = [p for p in fit_pos if p >= LOOKBACK and p + HORIZON - 1 <= fit_pos[-1]]
    fit_orig = fit_orig[::3] if "GRU" in name else fit_orig
    val_orig = day_origins(idx, *sp["val"])
    model = build(name, Z.shape[1], adj).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x, xm, y = windows(Z, TM, fit_orig)
    xv, xmv, yv = windows(Z, TM, val_orig)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), bs):
            j = perm[i:i + bs]
            loss = nn.functional.mse_loss(model(x[j].to(DEV), xm[j].to(DEV)), y[j].to(DEV))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(xv.to(DEV), xmv.to(DEV)), yv.to(DEV)).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["SNaive", "DLinear", "GRU", "GraphGRU", "iTransformer"])
    ap.add_argument("--splits", nargs="+", default=["A", "B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    v = load_volume()
    load_volume.index_cache = v.index
    zones = v.columns.to_numpy()
    Y = v.to_numpy(dtype=np.float32)
    adj = adjacency(zones)
    TM = time_feats(v.index)
    log = []
    for s in args.splits:
        sp = SPLITS[s]
        fit_pos = hour_index(v.index, *sp["fit"])
        mu, sd = Y[fit_pos].mean(0), Y[fit_pos].std(0) + 1e-3
        Z = ((Y - mu) / sd).astype(np.float32)
        orig = {k: day_origins(v.index, *sp[k]) for k in ("cal", "test")}
        truth = {k: np.stack([Y[p:p + HORIZON] for p in o]) for k, o in orig.items()}
        for m in args.models:
            for seed in (args.seeds if m != "SNaive" else [0]):
                t0 = time.time()
                if m == "SNaive":
                    pred = {k: np.stack([Y[p - 168:p - 168 + HORIZON] for p in o]) for k, o in orig.items()}
                    info = {}
                else:
                    model, vl, ep = train(m, Z, TM, adj, sp, seed)
                    pred = {k: np.clip(predict(model, Z, TM, o) * sd + mu, 0, None) for k, o in orig.items()}
                    info = {"val_mse": vl, "epochs": ep}
                np.savez_compressed(OUT / f"preds_{s}_{m}_s{seed}.npz", zones=zones,
                                    cal_pred=pred["cal"], cal_true=truth["cal"],
                                    test_pred=pred["test"], test_true=truth["test"],
                                    cal_origins=np.array([str(v.index[p]) for p in orig["cal"]]),
                                    test_origins=np.array([str(v.index[p]) for p in orig["test"]]))
                e = pred["test"] - truth["test"]
                rec = dict(split=s, model=m, seed=seed, mae=float(np.abs(e).mean()),
                           rmse=float(np.sqrt((e ** 2).mean())),
                           wape=float(np.abs(e).sum() / truth["test"].sum()), sec=round(time.time() - t0, 1), **info)
                print(json.dumps(rec), flush=True)
                log.append(rec)
    tag = "_".join(args.splits + args.models)
    pd.DataFrame(log).to_csv(OUT / f"forecast_metrics_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
