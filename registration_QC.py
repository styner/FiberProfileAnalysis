#!/usr/bin/env python3
"""
QC for the deformable alignment of subject diffusion tensors/metrics to a prior
diffusion-MRI atlas.

Each subject session has tensor-derived metric maps warped into atlas space
(``.../AtlasReg/*_Deformed<METRIC>.nii.gz`` and ``*_DeformedDTI.nrrd``).  The
atlas provides the same maps (``Atlas_*_<METRIC>.nii.gz`` + ``*_DTI.nrrd``).

The QC compares each subject to the atlas and (optionally) to an age-conditional
normative model, and flags registration outliers.

Metrics
-------
* Scalar maps (FA primary; MD/RD/AD optional) vs the atlas, within a brain mask:
  MAE, 3D-SSIM (+ map), NCC (zero-normalised cross-correlation).
* Tensor principal-direction **angular error** (deg) vs the atlas, over WM.

Age handling
------------
Subjects are age-diverse but registered into one age-specific atlas, so raw
similarity is age-confounded.  With ``--normative-dir`` an age-conditional
normative model (per age bin) provides voxelwise expectations, turning each
comparison into a standardised deviation (z):

* scalars: ``z = (subject - mean_age) / std_age``
* angular: principal directions are **axial** data (sign-ambiguous) on RP^2, so
  the normative "mean direction" is the leading eigenvector of the per-voxel
  dyadic mean ``T = mean(v vᵀ)`` (the spherical analogue of the circular mean),
  and the dispersion is ``sigma = sqrt(1 - tau1)`` (RMS ``sin`` of the reference
  angles, ``tau1`` = leading eigenvalue of the normalised ``T``).  The subject
  deviation is standardised as ``z = sin(angle(v_subj, mu)) / sigma`` -- valid
  for both small and large angles, degenerating to the tangent-space z-score
  when the spread is small.

Without a normative model, raw subject-vs-atlas scores are used (age-confounded).

Outlier decision is **one combined flag per subject-session** (a bad warp hits
all metrics/the tensor together).

Modes
-----
* ``--build-normative`` : compute the normative model from ``--reference-dir``
  into ``--normative-dir`` and exit.
* default (QC)          : score ``--data-dir`` subjects, write a table and outlier
  flags.  NIfTI disagreement maps + a per-subject preview PNG (atlas FA, DTI FA,
  FA diff, FA SSIM, angular error z) are written **only for flagged outliers**;
  ``--save-all-maps`` writes them for every session.

Usage
-----
    # build the age-conditional normative model from a reference cohort
    python registration_QC.py --build-normative --reference-dir RegistrationData \\
        --normative-dir RegNormative

    # QC (with normative if available, else raw)
    python registration_QC.py --data-dir RegistrationData --normative-dir RegNormative \\
        --out-dir RegistrationQC
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys

import numpy as np
import nibabel as nib
import nrrd

log = logging.getLogger("reg_qc")

AGE_RE = re.compile(r"ses-(\d+)m")
ALL_SCALARS = ["FA", "MD", "RD", "AD"]
# Candidate tensor-frame axis reflections (handedness / LPS-RAS sign conventions)
FLIP_VECS = {"none": (1, 1, 1), "x": (-1, 1, 1), "y": (1, -1, 1), "z": (1, 1, -1)}


# ---------------------------------------------------------------------------
# Bins / discovery
# ---------------------------------------------------------------------------
def parse_bins(spec):
    bins = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        lo, hi = (int(x) for x in tok.split("-"))
        bins.append([lo, hi, f"{lo}-{hi}m"])
    bins.sort(key=lambda b: b[0])
    if bins:
        bins[-1][1] = 10**9  # oldest bin open-ended
    return [tuple(b) for b in bins]


def bin_label_for_age(age, bins):
    for lo, hi, label in bins:
        if lo <= age <= hi:
            return label
    return None


def find_atlas(atlas_dir):
    """Return {metric: path} for scalar atlas maps and the atlas tensor path."""
    scalars = {}
    for m in ALL_SCALARS:
        hits = glob.glob(os.path.join(atlas_dir, f"*_{m}.nii.gz"))
        if hits:
            scalars[m] = sorted(hits)[0]
    tensor = sorted(glob.glob(os.path.join(atlas_dir, "*_DTI.nrrd")))
    return scalars, (tensor[0] if tensor else None)


DEFORMED_RE = re.compile(
    r"^(?P<sub>sub-[^_]+)_(?P<ses>ses-[^_]+)_(?P<mid>.+)_Deformed(?P<metric>FA|MD|RD|AD|DTI)\.(?:nii\.gz|nrrd)$")
PREFIX_RE = re.compile(r"^(?P<prefix>.+?_dwi)(?:_.*)?$")


def parse_deformed(basename):
    """(subject, session, prefix, metric) from a Deformed<METRIC> filename, or None.

    The prefix is the pipeline identifier (up to and including ``_dwi``), so it
    matches the profile/prep column names and distinguishes multiple
    acquisitions within one session.
    """
    m = DEFORMED_RE.match(basename)
    if m is None:
        return None
    pm = PREFIX_RE.match(m.group("mid"))
    prefix = pm.group("prefix") if pm else m.group("mid")
    return m.group("sub"), m.group("ses"), prefix, m.group("metric")


def find_sessions(root):
    """Discover scans (one per subject/session/prefix) with metric/tensor paths.

    Multiple acquisitions in the same session (different prefixes) become
    separate scan entries keyed by the full ``sub_ses_prefix`` identifier.
    """
    scans = {}  # (subject, session, prefix) -> entry
    for reg in sorted(glob.glob(os.path.join(root, "sub-*", "ses-*", "AtlasReg"))):
        ses_dir = os.path.dirname(reg)
        session = os.path.basename(ses_dir)
        am = AGE_RE.search(session)
        if am is None:
            log.warning("no age in %s; skipping", ses_dir)
            continue
        age = int(am.group(1))
        for f in sorted(glob.glob(os.path.join(reg, "*_Deformed*.nii.gz"))
                        + glob.glob(os.path.join(reg, "*_DeformedDTI.nrrd"))):
            parsed = parse_deformed(os.path.basename(f))
            if parsed is None:
                continue
            subject, ses, prefix, metric = parsed
            key = (subject, session, prefix)
            entry = scans.setdefault(key, {
                "id": f"{subject}_{session}_{prefix}", "subject": subject, "session": session,
                "prefix": prefix, "age": age, "scalars": {}, "tensor": None,
            })
            if metric == "DTI":
                entry["tensor"] = f
            elif metric in ALL_SCALARS:
                entry["scalars"][metric] = f
    return [scans[k] for k in sorted(scans)]


# ---------------------------------------------------------------------------
# Tensor / directions
# ---------------------------------------------------------------------------
def load_scalar(path):
    img = nib.load(path)
    return np.asanyarray(img.dataobj, dtype=np.float32), img.affine


def principal_directions(tensor_path, mask, flip=(1, 1, 1)):
    """Leading eigenvector per masked voxel from a 6-component NRRD tensor.

    Returns (X,Y,Z,3) float32 (zeros outside the mask).  NRRD symmetric-matrix
    order is [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz].  *flip* reflects the direction axes
    to correct a tensor-frame handedness/convention mismatch with the atlas.
    """
    comp, _ = nrrd.read(tensor_path)  # (6, X, Y, Z)
    idx = np.where(mask)
    c = comp[:, idx[0], idx[1], idx[2]].astype(np.float64)  # (6, N)
    n = c.shape[1]
    D = np.empty((n, 3, 3), dtype=np.float64)
    D[:, 0, 0] = c[0]; D[:, 0, 1] = D[:, 1, 0] = c[1]; D[:, 0, 2] = D[:, 2, 0] = c[2]
    D[:, 1, 1] = c[3]; D[:, 1, 2] = D[:, 2, 1] = c[4]; D[:, 2, 2] = c[5]
    _, vecs = np.linalg.eigh(D)          # ascending eigenvalues
    pd = (vecs[:, :, -1] * np.asarray(flip, dtype=np.float64)).astype(np.float32)
    out = np.zeros(mask.shape + (3,), dtype=np.float32)
    out[idx] = pd
    return out


def detect_tensor_flip(sessions, atlas_pd, atlas_fa, mask_thr, angular_fa_min, n_probe=3):
    """Detect the axis reflection aligning subject tensors to the atlas frame.

    Probes a few subjects; for each, picks the reflection minimising the median
    core-WM angular error to the atlas; returns the majority choice.
    """
    core_thr = max(angular_fa_min, 0.3)
    votes, medians, used = {}, {}, 0
    for s in sessions:
        if not s["tensor"] or "FA" not in s["scalars"]:
            continue
        sfa, _ = load_scalar(s["scalars"]["FA"])
        wm = (sfa > mask_thr) & (atlas_fa > core_thr)
        if wm.sum() < 1000:
            continue
        best = None
        for name, fv in FLIP_VECS.items():
            pd = principal_directions(s["tensor"], wm, fv)
            dot = np.clip(np.abs(np.sum(pd[wm] * atlas_pd[wm], axis=-1)), 0, 1)
            med = float(np.median(np.degrees(np.arccos(dot))))
            if best is None or med < best[1]:
                best = (name, med)
        votes[best[0]] = votes.get(best[0], 0) + 1
        medians.setdefault(best[0], []).append(best[1])
        used += 1
        if used >= n_probe:
            break
    if not votes:
        return "none", FLIP_VECS["none"], np.nan
    choice = max(votes, key=votes.get)
    return choice, FLIP_VECS[choice], float(np.median(medians[choice]))


def angular_error_deg(pd_a, pd_b, mask):
    """Undirected angle (deg) between two principal-direction fields, over mask."""
    dot = np.abs(np.sum(pd_a * pd_b, axis=-1))
    np.clip(dot, 0.0, 1.0, out=dot)
    ang = np.zeros(mask.shape, dtype=np.float32)
    ang[mask] = np.degrees(np.arccos(dot[mask]))
    return ang


# ---------------------------------------------------------------------------
# Similarity (masked)
# ---------------------------------------------------------------------------
def masked_mae(a, b, mask):
    return float(np.mean(np.abs(a[mask] - b[mask])))


def masked_ncc(a, b, mask):
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def masked_ssim(a, b, mask, data_range):
    from skimage.metrics import structural_similarity

    _, smap = structural_similarity(
        a.astype(np.float64), b.astype(np.float64),
        data_range=data_range, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False, full=True,
    )
    return float(np.mean(smap[mask])), smap.astype(np.float32)


def brain_mask(subj_fa, atlas_fa, thr):
    return (subj_fa > thr) & (atlas_fa > thr)


# ---------------------------------------------------------------------------
# Normative model
# ---------------------------------------------------------------------------
def build_normative(sessions, atlas_scalars, atlas_fa, bins, scalar_metrics, do_angular,
                    mask_thr, angular_fa_min, min_count, normative_dir, flip=(1, 1, 1), flip_name="none"):
    """Compute and save the age-conditional normative model, per bin."""
    os.makedirs(normative_dir, exist_ok=True)
    ref_affine = nib.load(atlas_scalars["FA"]).affine
    shape = atlas_fa.shape
    atlas_wm = atlas_fa > angular_fa_min

    manifest = {"bins": [b[2] for b in bins], "scalar_metrics": scalar_metrics,
                "angular": do_angular, "mask_threshold": mask_thr,
                "angular_fa_min": angular_fa_min, "min_count": min_count, "tensor_flip": flip_name}

    for lo, hi, label in bins:
        subs = [s for s in sessions if lo <= s["age"] <= hi]
        if not subs:
            log.info("bin %s: no reference subjects", label)
            continue
        log.info("bin %s: %d reference subjects", label, len(subs))
        bin_dir = os.path.join(normative_dir, label)
        os.makedirs(bin_dir, exist_ok=True)

        # scalar accumulators
        acc = {m: {"sum": np.zeros(shape, np.float64), "sqsum": np.zeros(shape, np.float64),
                   "cnt": np.zeros(shape, np.float64)} for m in scalar_metrics}
        # angular dyadic accumulator T = sum(v vᵀ): 6 unique comps + count
        if do_angular:
            T = np.zeros((6,) + shape, np.float64)
            Tcnt = np.zeros(shape, np.float64)

        for s in subs:
            sfa, _ = load_scalar(s["scalars"]["FA"])
            mask = brain_mask(sfa, atlas_fa, mask_thr)
            for m in scalar_metrics:
                if m not in s["scalars"]:
                    continue
                v, _ = load_scalar(s["scalars"][m])
                acc[m]["sum"][mask] += v[mask]
                acc[m]["sqsum"][mask] += v[mask].astype(np.float64) ** 2
                acc[m]["cnt"][mask] += 1.0
            if do_angular and s["tensor"]:
                wm = mask & atlas_wm
                pd = principal_directions(s["tensor"], wm, flip)[wm]  # (N,3)
                ii = np.where(wm)
                T[0][ii] += pd[:, 0] * pd[:, 0]; T[1][ii] += pd[:, 0] * pd[:, 1]
                T[2][ii] += pd[:, 0] * pd[:, 2]; T[3][ii] += pd[:, 1] * pd[:, 1]
                T[4][ii] += pd[:, 1] * pd[:, 2]; T[5][ii] += pd[:, 2] * pd[:, 2]
                Tcnt[ii] += 1.0

        # scalar mean/std
        for m in scalar_metrics:
            cnt = acc[m]["cnt"]
            ok = cnt >= min_count
            mean = np.full(shape, np.nan, np.float32)
            std = np.full(shape, np.nan, np.float32)
            mean[ok] = (acc[m]["sum"][ok] / cnt[ok]).astype(np.float32)
            var = acc[m]["sqsum"][ok] / cnt[ok] - (acc[m]["sum"][ok] / cnt[ok]) ** 2
            std[ok] = np.sqrt(np.clip(var * cnt[ok] / np.maximum(cnt[ok] - 1, 1), 0, None)).astype(np.float32)
            nib.save(nib.Nifti1Image(mean, ref_affine), os.path.join(bin_dir, f"{m}_mean.nii.gz"))
            nib.save(nib.Nifti1Image(std, ref_affine), os.path.join(bin_dir, f"{m}_std.nii.gz"))
            nib.save(nib.Nifti1Image(cnt.astype(np.float32), ref_affine),
                     os.path.join(bin_dir, f"{m}_count.nii.gz"))

        # angular mean axis (dyadic) + dispersion
        if do_angular:
            mu = np.zeros(shape + (3,), np.float32)
            sigma = np.full(shape, np.nan, np.float32)
            tau1 = np.full(shape, np.nan, np.float32)
            ok = np.where(Tcnt >= min_count)
            if ok[0].size:
                M = np.empty((ok[0].size, 3, 3), np.float64)
                for a, (r, cc) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
                    M[:, r, cc] = M[:, cc, r] = T[a][ok] / Tcnt[ok]
                w, v = np.linalg.eigh(M)  # ascending
                mu[ok] = v[:, :, -1].astype(np.float32)
                tau1[ok] = w[:, -1].astype(np.float32)
                sigma[ok] = np.sqrt(np.clip(1.0 - w[:, -1], 0, None)).astype(np.float32)
            nib.save(nib.Nifti1Image(mu, ref_affine), os.path.join(bin_dir, "angular_mu.nii.gz"))
            nib.save(nib.Nifti1Image(sigma, ref_affine), os.path.join(bin_dir, "angular_sigma.nii.gz"))
            nib.save(nib.Nifti1Image(tau1, ref_affine), os.path.join(bin_dir, "angular_coherence.nii.gz"))
            nib.save(nib.Nifti1Image(Tcnt.astype(np.float32), ref_affine),
                     os.path.join(bin_dir, "angular_count.nii.gz"))

    with open(os.path.join(normative_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Wrote normative model to %s/", normative_dir)


def load_normative_scalar(normative_dir, label, metric):
    base = os.path.join(normative_dir, label, metric)
    if not os.path.isfile(base + "_mean.nii.gz"):
        return None
    mean = np.asanyarray(nib.load(base + "_mean.nii.gz").dataobj, dtype=np.float32)
    std = np.asanyarray(nib.load(base + "_std.nii.gz").dataobj, dtype=np.float32)
    return mean, std


def load_normative_angular(normative_dir, label):
    d = os.path.join(normative_dir, label)
    if not os.path.isfile(os.path.join(d, "angular_mu.nii.gz")):
        return None
    mu = np.asanyarray(nib.load(os.path.join(d, "angular_mu.nii.gz")).dataobj, dtype=np.float32)
    sigma = np.asanyarray(nib.load(os.path.join(d, "angular_sigma.nii.gz")).dataobj, dtype=np.float32)
    return mu, sigma


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------
def make_preview(sid, atlas_fa, subj_fa, maps, out_dir):
    """One PNG per subject: atlas FA, DTI FA, FA diff, FA SSIM, angular error z."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = int(np.argmax([(atlas_fa[:, :, z] > 0.2).sum() for z in range(atlas_fa.shape[2])]))  # most-WM slice
    sl = lambda a: np.rot90(a[:, :, k])
    panels = [("Atlas FA", sl(atlas_fa), "gray", (0, 1)),
              ("DTI FA", sl(subj_fa), "gray", (0, 1))]
    if "FA_diff" in maps:
        panels.append(("FA diff (subj−atlas)", sl(maps["FA_diff"]), "RdBu_r", (-0.4, 0.4)))
    if "FA_ssim" in maps:
        panels.append(("FA SSIM", sl(maps["FA_ssim"]), "viridis", (0, 1)))
    if "angular_z" in maps:
        panels.append(("Angular error z", sl(maps["angular_z"]), "inferno", (0, 3)))
    elif "angular_deg" in maps:
        panels.append(("Angular error (deg)", sl(maps["angular_deg"]), "hot", (0, 60)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.6))
    for ax, (title, img, cm, (vlo, vhi)) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(img, cmap=cm, vmin=vlo, vmax=vhi)
        ax.set_title(title, fontsize=11); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Registration QC — {sid} (axial z={k})", fontsize=13)
    fig.tight_layout()
    prev_dir = os.path.join(out_dir, "previews")
    os.makedirs(prev_dir, exist_ok=True)
    fig.savefig(os.path.join(prev_dir, f"{sid}_preview.png"), dpi=110)
    plt.close(fig)


def qc_subject(s, atlas, bins, normative_dir, cfg, out_dir=None):
    """Score one session vs atlas (+ normative); return a row dict.

    When *out_dir* is given, also save the NIfTI disagreement maps and a preview
    PNG (used only for the subjects we choose to write out).
    """
    atlas_scalars, atlas_fa, atlas_pd, atlas_wm, ref_affine = atlas
    sfa, _ = load_scalar(s["scalars"]["FA"])
    mask = brain_mask(sfa, atlas_fa, cfg["mask_thr"])
    label = bin_label_for_age(s["age"], bins)
    row = {"id": s["id"], "subject": s["subject"], "session": s["session"],
           "prefix": s.get("prefix", ""), "age": s["age"], "bin": label}
    save_maps = out_dir is not None
    maps = {}  # name -> full-volume array (only populated when saving)

    # --- scalar metrics ---
    for m in cfg["scalar_metrics"]:
        if m not in s["scalars"]:
            continue
        v, _ = load_scalar(s["scalars"][m])
        dr = float(atlas_scalars[m][mask].max() - atlas_scalars[m][mask].min()) or 1.0
        row[f"{m}_MAE"] = masked_mae(v, atlas_scalars[m], mask)
        row[f"{m}_NCC"] = masked_ncc(v, atlas_scalars[m], mask)
        ssim_mean, ssim_map = masked_ssim(v, atlas_scalars[m], mask, dr)
        row[f"{m}_SSIM"] = ssim_mean

        znorm = load_normative_scalar(normative_dir, label, m) if normative_dir else None
        z = None
        if znorm is not None:
            mean, std = znorm
            valid = mask & np.isfinite(mean) & np.isfinite(std) & (std > 0)
            z = np.zeros(mask.shape, np.float32)
            z[valid] = (v[valid] - mean[valid]) / std[valid]
            row[f"{m}_meanAbsZ"] = float(np.mean(np.abs(z[valid]))) if valid.any() else np.nan
            row[f"{m}_fracZgt"] = float(np.mean(np.abs(z[valid]) > cfg["z_thresh"])) if valid.any() else np.nan
        if save_maps and m == "FA":
            diff = np.zeros(mask.shape, np.float32); diff[mask] = v[mask] - atlas_scalars[m][mask]
            smap = np.zeros(mask.shape, np.float32); smap[mask] = ssim_map[mask]
            maps["FA_diff"] = diff
            maps["FA_ssim"] = smap
            if z is not None:
                maps["FA_z"] = z

    # --- angular ---
    if cfg["do_angular"] and s["tensor"] is not None and atlas_pd is not None:
        wm = mask & atlas_wm
        pd = principal_directions(s["tensor"], wm, cfg["flip"])
        ang = angular_error_deg(pd, atlas_pd, wm)
        row["ANG_meanDeg"] = float(np.mean(ang[wm])) if wm.any() else np.nan
        if save_maps:
            maps["angular_deg"] = ang

        na = load_normative_angular(normative_dir, label) if normative_dir else None
        if na is not None:
            mu, sigma = na
            valid = wm & np.isfinite(sigma) & (np.linalg.norm(mu, axis=-1) > 0)
            zang = np.zeros(mask.shape, np.float32)
            if valid.any():
                dot = np.abs(np.sum(pd[valid] * mu[valid], axis=-1))
                np.clip(dot, 0.0, 1.0, out=dot)
                sin_d = np.sqrt(np.clip(1.0 - dot ** 2, 0, None))
                zang[valid] = sin_d / np.maximum(sigma[valid], cfg["angular_sigma_floor"])
            row["ANG_meanZ"] = float(np.mean(zang[valid])) if valid.any() else np.nan
            row["ANG_fracZgt"] = float(np.mean(zang[valid] > cfg["z_thresh"])) if valid.any() else np.nan
            if save_maps:
                maps["angular_z"] = zang

    # --- write maps + preview ---
    if save_maps and maps:
        sub_out = os.path.join(out_dir, "maps")
        os.makedirs(sub_out, exist_ok=True)
        for name, arr in maps.items():
            nib.save(nib.Nifti1Image(arr, ref_affine), os.path.join(sub_out, f"{s['id']}_{name}.nii.gz"))
        make_preview(s["id"], atlas_fa, sfa, maps, out_dir)
    return row


def robust_z(x):
    """MAD-based z-score (median/1.4826·MAD); 0 where scale is degenerate."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-12:
        return np.zeros_like(x)
    return (x - med) / scale


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="RegistrationData", help="Root with sub-*/ses-*/AtlasReg + Atlas/")
    p.add_argument("--atlas-dir", default=None, help="Atlas folder (default: <data-dir>/Atlas)")
    p.add_argument("--out-dir", default="RegistrationQC", help="QC output folder")
    p.add_argument("--reference-dir", default=None, help="Reference cohort for --build-normative (default: --data-dir)")
    p.add_argument("--normative-dir", default=None, help="Age-conditional normative model folder (read or write)")
    p.add_argument("--build-normative", action="store_true", help="Build the normative model and exit")
    p.add_argument("--bins", default="0-3,4-9,10-60", help="Age bins (months, inclusive; oldest open-ended)")
    p.add_argument("--scalar-metrics", default="FA", help="Comma list of scalar metrics to QC (default: FA)")
    p.add_argument("--no-angular", action="store_true", help="Skip the tensor angular-error metric")
    p.add_argument("--mask-threshold", type=float, default=1e-3, help="FA threshold for the brain mask (default: 1e-3)")
    p.add_argument("--angular-fa-min", type=float, default=0.2, help="Atlas FA floor for WM angular region (default: 0.2)")
    p.add_argument("--tensor-flip", default="auto", choices=["auto", "none", "x", "y", "z"],
                   help="Correct a subject-vs-atlas tensor-frame axis reflection (default: auto-detect)")
    p.add_argument("--angular-sigma-floor", type=float, default=0.035, help="Floor on angular dispersion (~sin 2°)")
    p.add_argument("--min-count", type=int, default=2, help="Min reference subjects per voxel for a valid normative")
    p.add_argument("--z-thresh", type=float, default=3.0, help="|z| threshold for extreme-voxel fractions (default: 3)")
    p.add_argument("--outlier-mad", type=float, default=3.5, help="Robust-z cutoff on the combined score (default: 3.5)")
    p.add_argument("--save-all-maps", action="store_true",
                   help="Write disagreement maps + previews for every session (default: only flagged outliers)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s | %(message)s" if args.verbose else "%(message)s",
                        stream=sys.stdout)

    atlas_dir = args.atlas_dir or os.path.join(args.data_dir, "Atlas")
    atlas_scalar_paths, atlas_tensor = find_atlas(atlas_dir)
    if "FA" not in atlas_scalar_paths:
        p.error(f"atlas FA not found in {atlas_dir}")
    scalar_metrics = [m.strip() for m in args.scalar_metrics.split(",") if m.strip()]
    scalar_metrics = [m for m in scalar_metrics if m in atlas_scalar_paths]
    do_angular = (not args.no_angular) and atlas_tensor is not None
    bins = parse_bins(args.bins)

    atlas_fa, _ = load_scalar(atlas_scalar_paths["FA"])
    atlas_wm = atlas_fa > args.angular_fa_min
    atlas_pd = principal_directions(atlas_tensor, atlas_wm) if do_angular else None

    def resolve_flip(sessions):
        if not do_angular or atlas_pd is None:
            return "none", FLIP_VECS["none"]
        if args.tensor_flip != "auto":
            return args.tensor_flip, FLIP_VECS[args.tensor_flip]
        name, fv, med = detect_tensor_flip(sessions, atlas_pd, atlas_fa,
                                           args.mask_threshold, args.angular_fa_min)
        log.info("Auto-detected tensor-frame flip: '%s' (median core-WM angular error %.1f°)", name, med)
        if name != "none":
            log.info("  -> applying axis reflection %s to subject tensors (LPS/RAS convention mismatch)", name)
        return name, fv

    if args.build_normative:
        ref_dir = args.reference_dir or args.data_dir
        if not args.normative_dir:
            p.error("--build-normative requires --normative-dir")
        sessions = find_sessions(ref_dir)
        log.info("Building normative from %d reference sessions in %s", len(sessions), ref_dir)
        flip_name, flip = resolve_flip(sessions)
        build_normative(sessions, atlas_scalar_paths, atlas_fa, bins, scalar_metrics, do_angular,
                        args.mask_threshold, args.angular_fa_min, args.min_count, args.normative_dir,
                        flip, flip_name)
        return 0

    # --- QC mode ---
    normative_dir = args.normative_dir if (args.normative_dir and
                                           os.path.isfile(os.path.join(args.normative_dir, "manifest.json"))) else None
    log.info("Metrics: scalars=%s angular=%s | normative=%s",
             scalar_metrics, do_angular, normative_dir or "(none -> raw, age-confounded)")

    atlas_scalars = {m: load_scalar(atlas_scalar_paths[m])[0] for m in scalar_metrics}
    ref_affine = nib.load(atlas_scalar_paths["FA"]).affine
    atlas = (atlas_scalars, atlas_fa, atlas_pd, atlas_wm, ref_affine)

    sessions = find_sessions(args.data_dir)
    flip_name, flip = resolve_flip(sessions)
    if normative_dir:
        with open(os.path.join(normative_dir, "manifest.json")) as fh:
            nf = json.load(fh).get("tensor_flip", "none")
        if nf != flip_name:
            log.warning("normative was built with tensor_flip='%s' but QC uses '%s'", nf, flip_name)

    cfg = {"scalar_metrics": scalar_metrics, "do_angular": do_angular, "mask_thr": args.mask_threshold,
           "angular_fa_min": args.angular_fa_min, "angular_sigma_floor": args.angular_sigma_floor,
           "z_thresh": args.z_thresh, "flip": flip}

    log.info("QC on %d sessions", len(sessions))
    os.makedirs(args.out_dir, exist_ok=True)

    save_all = args.save_all_maps
    rows, scored = [], []
    for s in sessions:
        if "FA" not in s["scalars"]:
            log.warning("%s: no FA; skipping", s["id"])
            continue
        rows.append(qc_subject(s, atlas, bins, normative_dir, cfg, args.out_dir if save_all else None))
        scored.append(s)
        log.info("  scored %s (age %dm, bin %s)%s", s["id"], s["age"], rows[-1]["bin"],
                 " [maps saved]" if save_all else "")

    import pandas as pd
    df = pd.DataFrame(rows)

    # --- combined per-subject score + outlier flag ---
    if normative_dir:
        prim = [f"{m}_meanAbsZ" for m in scalar_metrics if f"{m}_meanAbsZ" in df]
        if do_angular and "ANG_meanZ" in df:
            prim.append("ANG_meanZ")
        df["combined_score"] = df[prim].mean(axis=1)
    else:
        parts = []
        for m in scalar_metrics:
            if f"{m}_MAE" in df:
                parts.append(robust_z(df[f"{m}_MAE"].to_numpy()))
                parts.append(robust_z(1.0 - df[f"{m}_SSIM"].to_numpy()))
                parts.append(robust_z(1.0 - df[f"{m}_NCC"].to_numpy()))
        if do_angular and "ANG_meanDeg" in df:
            parts.append(robust_z(df["ANG_meanDeg"].to_numpy()))
        df["combined_score"] = np.mean(np.vstack(parts), axis=0) if parts else np.nan

    df["combined_robust_z"] = robust_z(df["combined_score"].to_numpy())
    df["is_outlier"] = df["combined_robust_z"] > args.outlier_mad

    csv = os.path.join(args.out_dir, "registration_qc.csv")
    df.to_csv(csv, index=False)
    n_out = int(df["is_outlier"].sum())
    log.info("Wrote %s | %d/%d sessions flagged (combined robust-z > %.1f)",
             csv, n_out, len(df), args.outlier_mad)
    if n_out:
        log.info("Outliers: %s", ", ".join(df.loc[df["is_outlier"], "id"]))

    # --- maps + previews: flagged only (default) or all (already done above) ---
    if save_all:
        log.info("Saved maps + previews for all %d sessions (--save-all-maps)", len(scored))
    else:
        for i in np.where(df["is_outlier"].to_numpy())[0]:
            qc_subject(scored[i], atlas, bins, normative_dir, cfg, args.out_dir)
        log.info("Saved maps + previews for %d flagged session(s) under %s/{maps,previews}", n_out, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
