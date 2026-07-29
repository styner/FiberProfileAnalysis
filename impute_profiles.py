#!/usr/bin/env python3
"""
Impute missing values in along-tract profile tables with a **per-dataset**
SIREN implicit neural representation (INR).

For each dataset (one ``<subject>_<session>_<prefix>`` identifier) *all* of its
profiles -- every tract and every metric -- are imputed **jointly** by a single
SIREN, using only that dataset's own observed samples (never any
population/cross-subject information).

The network maps the 4-D coordinate ``(x, y, z, arc_length)`` of a sampled
location to a vector of the metric values at that location (one output channel
per metric: fa, md, rd, ad, NDI, ODI, FWF).  It is trained on all observed
samples pooled across every tract of the dataset (missing entries are masked
out of the loss), so tracts are distinguished by their 3-D position and the
metrics share a common coordinate backbone.  The trained network then predicts
the values at the missing locations.

The ``(x, y, z)`` coordinates come from the tract axis polydata in
``FiberAxis/<tract>_axis.vtk`` (point attribute ``SamplingDistance2Origin`` ==
arc length), interpolated to each profile arc-length.

SIREN defaults: 200 epochs, omega_0 = 10, 3 hidden layers, width 256,
Adam lr 1e-4.  Device selection: auto -> cuda > mps > cpu.

Output mirrors the input layout in a new folder (default ``Profiles_Imputed``),
in exactly the same CSV format (observed cells preserved verbatim; only blank
cells are filled).

Usage
-----
    python impute_profiles.py --profiles-dir Profiles --axis-dir FiberAxis \\
        --out-dir Profiles_Imputed
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

log = logging.getLogger("impute")

METRIC_ORDER = ["fa", "md", "rd", "ad", "NDI", "ODI", "FWF"]


# ---------------------------------------------------------------------------
# Tract axis (VTK legacy ASCII polydata)
# ---------------------------------------------------------------------------
def load_axis(path: str):
    """Return (points Nx3, distances N) from a legacy ASCII VTK polydata axis.

    ``distances`` is the ``SamplingDistance2Origin`` point scalar (== arc length).
    """
    toks = open(path).read().split()
    i = toks.index("POINTS")
    n = int(toks[i + 1])
    start = i + 3  # skip 'POINTS', count, dtype
    pts = np.asarray(toks[start:start + 3 * n], dtype=float).reshape(n, 3)
    si = next(
        k for k, t in enumerate(toks)
        if t == "SCALARS" and k + 1 < len(toks) and "SamplingDistance" in toks[k + 1]
    )
    lk = toks.index("LOOKUP_TABLE", si)
    dstart = lk + 2  # skip 'LOOKUP_TABLE', 'default'
    dist = np.asarray(toks[dstart:dstart + n], dtype=float)
    return pts, dist


def axis_coords_for(arc_lengths: np.ndarray, pts: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Interpolate axis (x,y,z) at each profile arc-length -> (M, 4) coords."""
    order = np.argsort(dist)
    d = dist[order]
    x = np.interp(arc_lengths, d, pts[order, 0])
    y = np.interp(arc_lengths, d, pts[order, 1])
    z = np.interp(arc_lengths, d, pts[order, 2])
    return np.column_stack([x, y, z, arc_lengths]).astype(np.float32)


# ---------------------------------------------------------------------------
# SIREN (multi-output)
# ---------------------------------------------------------------------------
def build_siren(in_features, hidden, hidden_layers, out_features, omega_0):
    import torch
    from torch import nn

    class SineLayer(nn.Module):
        def __init__(self, in_f, out_f, is_first=False):
            super().__init__()
            self.omega_0 = omega_0
            self.linear = nn.Linear(in_f, out_f)
            with torch.no_grad():
                if is_first:
                    self.linear.weight.uniform_(-1.0 / in_f, 1.0 / in_f)
                else:
                    b = np.sqrt(6.0 / in_f) / omega_0
                    self.linear.weight.uniform_(-b, b)

        def forward(self, x):
            return torch.sin(self.omega_0 * self.linear(x))

    layers = [SineLayer(in_features, hidden, is_first=True)]
    for _ in range(hidden_layers):
        layers.append(SineLayer(hidden, hidden))
    final = nn.Linear(hidden, out_features)
    with torch.no_grad():
        b = np.sqrt(6.0 / hidden) / omega_0
        final.weight.uniform_(-b, b)
    layers.append(final)
    return nn.Sequential(*layers)


def pick_device(choice: str):
    import torch

    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit_joint(X, Y_norm, cfg, device):
    """Train a multi-output SIREN on (coords -> metric-vector), masking NaN targets."""
    import torch

    mask = (~np.isnan(Y_norm)).astype(np.float32)
    Yf = np.nan_to_num(Y_norm, nan=0.0).astype(np.float32)
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Yf).to(device)
    Mt = torch.from_numpy(mask).to(device)
    denom = Mt.sum().clamp_min(1.0)

    model = build_siren(X.shape[1], cfg["hidden"], cfg["hidden_layers"], Y_norm.shape[1], cfg["omega0"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model.train()
    for _ in range(cfg["epochs"]):
        opt.zero_grad()
        pred = model(Xt)
        loss = (((pred - Yt) ** 2) * Mt).sum() / denom
        loss.backward()
        opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Format-preserving CSV write (only fill blank cells)
# ---------------------------------------------------------------------------
def write_filled(in_path, out_path, filled, value_fmt):
    """Copy *in_path* to *out_path*, substituting imputed values into blanks.

    *filled*: {data_row_index: {value_col_index: value_float}}.  Non-imputed
    cells are preserved byte-for-byte.
    """
    lines = open(in_path).read().splitlines()
    out = [lines[0]]  # header
    for r, line in enumerate(lines[1:]):
        row_fill = filled.get(r)
        if row_fill:
            parts = line.split(",")
            for j, val in row_fill.items():
                parts[j + 1] = value_fmt % val  # +1: column 0 is Arc_Length
            line = ",".join(parts)
        out.append(line)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profiles-dir", default="Profiles", help="Root with <tract>/<tract>_<metric>.csv tables")
    p.add_argument("--axis-dir", default="FiberAxis", help="Folder of <tract>_axis.vtk polydata")
    p.add_argument("--out-dir", default="Profiles_Imputed", help="Output root (mirrors input layout)")
    p.add_argument("--epochs", type=int, default=200, help="SIREN training epochs (default: 200)")
    p.add_argument("--omega0", type=float, default=10.0, help="SIREN omega_0 (default: 10)")
    p.add_argument("--hidden-layers", type=int, default=3, help="SIREN hidden sine layers (default: 3)")
    p.add_argument("--hidden-width", type=int, default=256, help="SIREN layer width (default: 256)")
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate (default: 1e-4)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Torch device")
    p.add_argument("--value-format", default="%.8g", help="printf format for imputed values")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s" if args.verbose else "%(message)s",
        stream=sys.stdout,
    )

    if not os.path.isdir(args.profiles_dir):
        p.error(f"profiles dir not found: {args.profiles_dir}")
    if not os.path.isdir(args.axis_dir):
        p.error(f"axis dir not found: {args.axis_dir}")

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    log.info("Torch device: %s", device)

    cfg = {
        "epochs": args.epochs, "omega0": args.omega0, "hidden": args.hidden_width,
        "hidden_layers": args.hidden_layers, "lr": args.lr,
    }
    log.info("SIREN: %d epochs, omega_0=%g, %d hidden layers x %d, Adam lr=%g (joint per subject-session)",
             args.epochs, args.omega0, args.hidden_layers, args.hidden_width, args.lr)

    # --- preload every profile table, grouped by tract ---------------------
    tables = sorted(
        f for f in glob.glob(os.path.join(args.profiles_dir, "**", "*.csv"), recursive=True)
        if not f.endswith("_agebinstats.csv")
    )
    log.info("Found %d profile tables", len(tables))

    tract_frames: dict = defaultdict(dict)   # tract -> {metric: DataFrame}
    tract_paths: dict = defaultdict(dict)    # tract -> {metric: csv path}
    for f in tables:
        tract = os.path.basename(os.path.dirname(f))
        metric = os.path.basename(f)[: -len(".csv")].rsplit("_", 1)[1]
        tract_frames[tract][metric] = pd.read_csv(f, index_col=0)
        tract_paths[tract][metric] = f

    # metrics actually present (in preferred order)
    metrics = [m for m in METRIC_ORDER if any(m in fr for fr in tract_frames.values())]
    metrics += sorted({m for fr in tract_frames.values() for m in fr} - set(metrics))
    log.info("Metrics (output channels): %s", metrics)

    # --- tract coordinates + global normalisation --------------------------
    tract_arc: dict = {}      # tract -> arc-length index (np array)
    tract_coords_raw: dict = {}
    for tract, frames in tract_frames.items():
        axis_path = os.path.join(args.axis_dir, f"{tract}_axis.vtk")
        if not os.path.isfile(axis_path):
            log.warning("no axis file for tract '%s'; its blanks cannot be imputed", tract)
            continue
        arc = next(iter(frames.values())).index.to_numpy(dtype=float)
        tract_arc[tract] = arc
        tract_coords_raw[tract] = axis_coords_for(arc, *load_axis(axis_path))

    all_coords = np.vstack(list(tract_coords_raw.values()))
    lo = all_coords.min(axis=0, keepdims=True)
    hi = all_coords.max(axis=0, keepdims=True)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    tract_coords_norm = {t: ((c - lo) / span * 2.0 - 1.0).astype(np.float32) for t, c in tract_coords_raw.items()}

    # --- identifiers (datasets) and which have missing cells ---------------
    ident_missing: dict = defaultdict(int)
    for tract, frames in tract_frames.items():
        for m, df in frames.items():
            na = df.isna().sum()
            for ident in df.columns:
                if na[ident] > 0:
                    ident_missing[ident] += int(na[ident])
    datasets = sorted(i for i, c in ident_missing.items() if c > 0)
    log.info("%d datasets (subject-sessions) need imputation", len(datasets))

    # --- per-dataset joint fit + fill --------------------------------------
    fills: dict = defaultdict(lambda: defaultdict(dict))  # csv_path -> {row -> {col_idx -> val}}
    imputable_tracts = [t for t in tract_frames if t in tract_coords_norm]
    n_cells = 0
    for di, ident in enumerate(datasets, 1):
        present = [t for t in imputable_tracts if ident in next(iter(tract_frames[t].values())).columns]
        if not present:
            continue

        # Pool observed samples across all tracts of this dataset.
        Xparts, Yparts = [], []
        for t in present:
            idx = tract_arc[t]
            cols = []
            for m in metrics:
                df = tract_frames[t].get(m)
                if df is not None and ident in df.columns:
                    cols.append(df[ident].reindex(idx).to_numpy(dtype=float))
                else:
                    cols.append(np.full(len(idx), np.nan))
            Xparts.append(tract_coords_norm[t])
            Yparts.append(np.column_stack(cols))
        X = np.vstack(Xparts).astype(np.float32)
        Y = np.vstack(Yparts)

        # Per-metric (per-channel) standardisation from this dataset's own values.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mu = np.nanmean(Y, axis=0)
            sd = np.nanstd(Y, axis=0)
        has_obs = np.sum(~np.isnan(Y), axis=0) > 0
        sd_safe = np.where((sd < 1e-12) | ~np.isfinite(sd), 1.0, sd)
        mu_safe = np.where(np.isfinite(mu), mu, 0.0)
        Y_norm = (Y - mu_safe) / sd_safe

        model = fit_joint(X, Y_norm.astype(np.float32), cfg, device)

        # Predict per tract and fill this dataset's blank cells.
        with torch.no_grad():
            for t in present:
                cN = torch.from_numpy(tract_coords_norm[t]).to(device)
                pred = model(cN).cpu().numpy() * sd_safe + mu_safe  # denormalise
                for mi, m in enumerate(metrics):
                    if not has_obs[mi]:
                        continue  # channel unobserved -> cannot impute without population data
                    df = tract_frames[t].get(m)
                    if df is None or ident not in df.columns:
                        continue
                    col = df[ident].to_numpy(dtype=float)
                    miss = np.where(np.isnan(col))[0]
                    if miss.size == 0:
                        continue
                    j = df.columns.get_loc(ident)
                    path = tract_paths[t][m]
                    for r in miss:
                        fills[path][int(r)][int(j)] = float(pred[r, mi])
                        n_cells += 1
        if di % 25 == 0:
            log.info("  ... %d/%d datasets fit (%d cells filled)", di, len(datasets), n_cells)

    # --- write every table to the output tree ------------------------------
    for f in tables:
        rel = os.path.relpath(f, args.profiles_dir)
        write_filled(f, os.path.join(args.out_dir, rel), fills.get(f, {}), args.value_format)

    log.info("Done. Wrote %d tables to %s/; imputed %d datasets (%d cells).",
             len(tables), args.out_dir, len(datasets), n_cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
