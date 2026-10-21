"""Score wafer lots with the saved WAFER-GUARD model.

Usage:
    python inference.py <input.csv> [output.csv]

The input needs the 590 `sensor_*` columns (a `timestamp` column is optional).
Writes `defect_probability` and `flag` (1 = alarm) for each row.

Everything is loaded from plain files in ./artifacts (written by wafer_guard_fixed.ipynb):
    wafer_sae.pt            model weights (state_dict)
    preprocess_state.npz    kept columns + scaling statistics, frozen at the start of the test window
    config.json             column names, PS-L codes, model sizes, decision threshold
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


class WaferSAE(nn.Module):
    """Same architecture as in wafer_guard_fixed.ipynb (section 5)."""

    def __init__(self, d_cont, d_cat, d_hid, d_lat, p_drop):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 * d_cont + d_cat, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(d_hid, d_hid // 2), nn.LayerNorm(d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_lat),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_lat, d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_cont),
        )
        self.head = nn.Linear(d_lat, 1)

    def forward(self, z, mask, cat):
        h = self.encoder(torch.cat([z, mask, cat], dim=1))
        return self.decoder(h), self.head(h).squeeze(-1), h


def load(art=ART):
    with open(os.path.join(art, "config.json")) as fh:
        cfg = json.load(fh)
    state = dict(np.load(os.path.join(art, "preprocess_state.npz")))
    model = WaferSAE(cfg["d_cont"], cfg["d_cat"], cfg["hidden"], cfg["latent"], cfg["dropout"])
    model.load_state_dict(torch.load(os.path.join(art, "wafer_sae.pt"), map_location="cpu"))
    model.eval()
    return model, state, cfg


def preprocess(df, state, cfg):
    X = df[cfg["sensor_cols"]].to_numpy(dtype=np.float64)
    codes = cfg["legacy_codes"]

    # PS-L legacy probe station: sensor_0 == 0.0 is a sentinel, sensor_1 is a firmware code
    is_legacy = X[:, 0] == 0.0
    unknown = is_legacy & ~np.isin(X[:, 1], codes)
    if unknown.any():
        raise ValueError(f"unknown PS-L firmware code(s): {sorted(set(X[unknown, 1]))}")
    cat = np.zeros((len(X), 1 + len(codes)), dtype=np.float32)
    cat[:, 0] = is_legacy
    for j, c in enumerate(codes):
        cat[:, 1 + j] = is_legacy & (X[:, 1] == c)
    X[is_legacy, 0] = np.nan
    X[is_legacy, 1] = np.nan

    x = X[:, state["keep"]]
    mask = ~np.isnan(x) & state["usable"]
    with np.errstate(invalid="ignore"):
        z = np.where(mask, (x - state["median"]) / state["scale"], 0.0)
    z = np.clip(z, -cfg["clip_z"], cfg["clip_z"])
    return z.astype(np.float32), mask.astype(np.float32), cat


@torch.no_grad()
def score(df, art=ART):
    model, state, cfg = load(art)
    z, m, c = preprocess(df, state, cfg)
    _, logits, _ = model(torch.from_numpy(z), torch.from_numpy(m), torch.from_numpy(c))
    prob = torch.sigmoid(logits).numpy()
    return prob, (prob >= cfg["threshold"]).astype(int)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    torch.set_num_threads(1)
    data = pd.read_csv(sys.argv[1])
    prob, flag = score(data)
    out = pd.DataFrame({"defect_probability": prob, "flag": flag})
    if "timestamp" in data:
        out.insert(0, "timestamp", data["timestamp"])
    if len(sys.argv) > 2:
        out.to_csv(sys.argv[2], index=False)
        print(f"scored {len(out)} lots -> {sys.argv[2]} ({flag.sum()} flagged)")
    else:
        print(out.to_string(index=False))
