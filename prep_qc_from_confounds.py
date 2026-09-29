#!/usr/bin/env python3
"""
Derive prep-QC motion numbers from ABCD / HBCD diffusion confounds files.

The NIRAL prep-QC report (``PrepQC/DWIQC_report.csv``) carries one row per
scan::

    sub,ses,prefix,remaining_number_of_gradients,number_of_excluded_gradients,
    rms_larger_than_1,rms_larger_than_2,rms_larger_than_3

ABCD and HBCD are preprocessed with QSIPrep instead, which writes a per-scan
``*_confounds.tsv`` (one row per diffusion volume) rather than that report.  This
script reads those confounds files and fills the same columns from the volume's
motion, so QSIPrep-preprocessed cohorts can be reported and pooled alongside the
DTIPrep-processed ones.

Motion measures
---------------
Two measures of the same frame-to-frame rigid-body motion are used here.  Write
the change between consecutive volumes as a six-vector with the rotations
converted to millimetres of arc::

    v = (dtx, dty, dtz, r*drx, r*dry, r*drz)          r = 50 mm

* **FD** -- Power's framewise displacement, ``|v|_1`` (the sum of the absolute
  components).  QSIPrep writes it into the ``framewise_displacement`` column,
  and it can be recomputed from ``trans_*`` / ``rot_*`` (``--recompute-fd``; the
  two agree exactly at the default ``--fd-radius`` of 50 mm).
* **RMS** -- the Jenkinson displacement used by FSL (``eddy_movement_rms``,
  mcflirt's ``rmsrel``): the root-mean-square displacement of brain voxels
  induced by the motion,

      rms = sqrt( R^2/5 * trace(A'A) + |dt|^2 )      A = M - I

  for the relative rotation matrix ``M`` over a ball of radius ``R``
  (``--rms-radius``, FSL's default 80 mm).  It is computed from ``trans_*`` /
  ``rot_*``, which QSIPrep always writes.

Translating an RMS threshold into an FD threshold
-------------------------------------------------
The report's ``rms_larger_than_<T>`` columns count volumes whose **RMS** exceeds
T mm, so a threshold applied to FD is not the same cut.  The two are, however,
the L1 and L2 norms of the very same vector: for small angles
``trace(A'A) = 2|dr|^2``, so

    rms ~= sqrt( |dt|^2 + (R*sqrt(2/5))^2 * |dr|^2 ) = |v|_2

and FSL's default ``R = 80 mm`` puts the effective rotation radius at
``80*sqrt(2/5) = 50.6 mm`` -- essentially Power's 50 mm.  (That agreement is not
a coincidence: both stand in for the typical distance of brain tissue from the
rotation centre.  Checked numerically on the HBCD example file, the exact
Jenkinson formula and ``|v|_2`` agree to 4e-3 mm.)

Being the L1 and L2 norms of one six-vector, the two are bounded against each
other::

    rms <= fd <= sqrt(6) * rms        i.e.  k = fd/rms in [1.00, 2.45]

``k = 1`` for motion along a single axis, ``k = sqrt(6)`` for motion spread
evenly over all six components; real head motion is dominated by two or three
coupled components and lands in between.  (The bound is the ideal case, where
both measures are read off the same vector.  FD differences the Euler angles
directly while the RMS composes the true relative rotation ``R_i+1 R_i'``, so
once a scan has drifted to a non-trivial cumulative rotation the two part
company at order ``|r|*|dr|`` and individual volumes can sit slightly outside --
measured at k up to 2.50 for a drift of 0.34 rad.  Treat the interval as a
sanity range, not a hard constraint.)

On ``sub-G0MXBE3C_ses-00A`` the ratio has median **1.62** (p5-p95 1.23-2.19,
full range 1.14-2.26), giving

    rms 1 mm -> fd ~1.6 mm      rms 2 mm -> fd ~3.2 mm      rms 3 mm -> fd ~4.9 mm

Hence ``--rms-fd-factor`` (default 1.6), applied when thresholds are given in
RMS units but motion is measured as FD.  ``--measure-factor`` replaces the fixed
value with the ratio measured per scan from its own motion parameters, which is
markedly more faithful whenever a cohort's motion has a different *shape* from
the one the default was calibrated on: on a simulated scan whose motion is
spread evenly over all six axes (k = 1.99), counting volumes above an RMS of
1.5 mm gives 124 by the direct RMS, 122 through the measured factor, but 207
through the fixed 1.6.

Two caveats on that conversion:

* It assumes DTIPrep's column is a *displacement* RMS in mm (as eddy and mcflirt
  report).  Were it instead the plain RMS of the six parameter values, that
  equals ``|v|_2/sqrt(6)`` and every factor above would scale by ``sqrt(6)``
  (rms 1 -> fd ~4).  The 1/2/3 mm thresholds argue for the displacement reading,
  but the pipeline's own ``00_SLICE_Check/report.md`` settles it.
* ``k`` is a per-volume quantity and varies (1.1-2.3 within this one subject), so
  a converted threshold reproduces the typical volume, not an exact count.  With
  a subject processed through both pipelines, matching the *counts* is the more
  faithful calibration -- and it absorbs the ambiguity above.

``--motion-measure rms`` sidesteps the factor altogether: the RMS is computed
directly from the motion parameters and compared against the report's thresholds
as-is.  It is the more direct route whenever ``trans_*`` / ``rot_*`` are present,
which for QSIPrep output they always are.

Other column mapping
--------------------
* ``number_of_excluded_gradients`` -- volumes above ``--exclude-threshold``.
  Note this is a **motion-only** substitute: DTIPrep also excludes gradients on
  slice-wise / interlace / gradient-wise intensity checks, which a confounds
  file cannot reproduce, so expect this count to sit below the report's.
* ``remaining_number_of_gradients`` -- all remaining volumes, baselines included
  (as in the report, where remaining + excluded is the full gradient list).

AP / PA pairs
-------------
A confounds file often holds both phase-encoding directions of one acquisition
(``original_file`` = ``..._dir-AP_..._dwi`` and ``..._dir-PA_..._dwi``).  By
default each direction is its own row, as ``original_file`` names two scans.
With ``--sum-ap-pa`` the rows that differ only in their ``dir-`` entity are
merged into one by summation -- remaining, excluded and every
``*_larger_than_*`` count are added, which is exactly what counting over the
pooled volumes of both directions gives -- and the merged row's prefix drops the
``dir-`` entity (``acq-dir79select_dir-AP_run-001_dwi`` ->
``acq-dir79select_run-001_dwi``).  Rows that differ in anything else (a
different ``run-``, say) stay separate; ``--combine`` is the blunter tool that
merges every run in a file regardless.

The first volume of a scan has no predecessor, so its motion is undefined
(``n/a``) and it is never counted as high-motion.  Where one confounds file
concatenates several source runs (``original_file``), the value spanning a run
boundary is likewise dropped -- it is a difference between two unrelated
acquisitions.

Usage
-----
    python prep_qc_from_confounds.py sub-G0MXBE3C_ses-00A_confounds.tsv
    python prep_qc_from_confounds.py QSIPrepOutput/ -o PrepQC/DWIQC_report_hbcd.csv
    python prep_qc_from_confounds.py QSIPrepOutput/ --motion-measure rms
    python prep_qc_from_confounds.py QSIPrepOutput/ --motion-measure rms --sum-ap-pa
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

log = logging.getLogger("prep_qc_confounds")

FD_COLUMN = "framewise_displacement"
TRANS_COLS = ["trans_x", "trans_y", "trans_z"]
ROT_COLS = ["rot_x", "rot_y", "rot_z"]
RUN_COLUMN = "original_file"

# Power's FD radius (mm) and FSL's Jenkinson-RMS ball radius (mm).
FD_RADIUS = 50.0
RMS_RADIUS = 80.0
# fd/rms for typical head motion; see "Translating an RMS threshold" above.
DEFAULT_RMS_FD_FACTOR = 1.6
FACTOR_BOUNDS = (1.0, np.sqrt(6.0))  # L1 vs L2 norm of a six-vector (the ideal case)
# Generous range for a *measured* ratio: outside this something is wrong (unit
# confusion, a degenerate scan) rather than merely unusual motion.
FACTOR_SANITY = (0.5, 3.5)

# sub-<id>_ses-<id>_<prefix>, as in prep_qc_reformat.py (the sub-/ses- tags are
# stripped in the report, so they are stripped here too).
ID_RE = re.compile(r"sub-(?P<sub>[^_/]+)_ses-(?P<ses>[^_/]+)(?:_(?P<prefix>.+))?$")

BASE_COLUMNS = ["sub", "ses", "prefix",
                "remaining_number_of_gradients", "number_of_excluded_gradients"]
EXTRA_COLUMNS = ["n_volumes", "n_motion_defined", "motion_mean", "motion_median", "motion_max",
                 "rms_fd_factor"]


def strip_nifti_ext(name):
    """'sub-x_ses-y_dwi.nii.gz' -> 'sub-x_ses-y_dwi' (also .nii, .nrrd, .tsv)."""
    for ext in (".nii.gz", ".nii", ".nrrd", ".tsv", ".csv"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def parse_identity(name):
    """(sub, ses, prefix) from a ``sub-X_ses-Y_<prefix>`` name; None if no match.

    *prefix* is None when the name carries no entities beyond subject/session.
    """
    m = ID_RE.match(strip_nifti_ext(os.path.basename(str(name))))
    if m is None:
        return None
    return m.group("sub"), m.group("ses"), m.group("prefix")


def scan_identity(confounds_path, original_file):
    """(sub, ses, prefix) for one scan, preferring the source image's entities.

    ``original_file`` names the acquisition (``..._acq-dir79select_run-001_dwi``)
    where the confounds filename often only says ``_confounds``, so it is the
    better source of the report's *prefix*; subject/session fall back to the
    confounds path (its parent folders included, for BIDS-style layouts).
    """
    ident = parse_identity(original_file) if original_file else None
    prefix = ident[2] if ident else None

    if ident is None:  # no identity in original_file -> the confounds path
        parts = [os.path.basename(confounds_path)] + os.path.abspath(confounds_path).split(os.sep)[::-1]
        ident = next((p for p in (parse_identity(x) for x in parts) if p), None)
    if ident is None:
        return None
    sub, ses, own_prefix = ident
    if prefix is None:  # fall back to the confounds name, minus the suffix
        prefix = re.sub(r"_?(confounds|desc-confounds|dwi_confounds)$", "", own_prefix or "")
    return sub, ses, prefix or "dwi"


def motion_params(df):
    """(translations mm, rotations rad) as (N,3) arrays; raises if absent."""
    missing = [c for c in TRANS_COLS + ROT_COLS if c not in df]
    if missing:
        raise ValueError(f"missing motion parameter column(s) {missing}")
    trans = df[TRANS_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    rot = df[ROT_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return trans, rot


def rotation_matrices(rot):
    """(N,3,3) rotation matrices from (N,3) Euler angles (radians), Rz @ Ry @ Rx."""
    a, b, c = rot.T
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    z, o = np.zeros_like(a), np.ones_like(a)
    rx = np.stack([o, z, z, z, ca, -sa, z, sa, ca], -1).reshape(-1, 3, 3)
    ry = np.stack([cb, z, sb, z, o, z, -sb, z, cb], -1).reshape(-1, 3, 3)
    rz = np.stack([cc, -sc, z, sc, cc, z, z, z, o], -1).reshape(-1, 3, 3)
    return rz @ ry @ rx


def framewise_displacement(df, radius=FD_RADIUS, recompute=False):
    """Power FD per volume in mm (NaN for the first), from the column or params."""
    if not recompute and FD_COLUMN in df:
        return pd.to_numeric(df[FD_COLUMN], errors="coerce").to_numpy(dtype=float)

    try:
        trans, rot = motion_params(df)
    except ValueError as exc:
        raise ValueError(f"no '{FD_COLUMN}' column and cannot recompute it: {exc}") from None
    # L1 norm of (translations, rotations scaled to mm of arc)
    steps = np.abs(np.diff(np.hstack([trans, rot * radius]), axis=0)).sum(axis=1)
    return np.concatenate([[np.nan], steps])  # first volume has no predecessor


def rms_displacement(df, radius=RMS_RADIUS):
    """Jenkinson RMS displacement per volume in mm (NaN for the first).

    ``sqrt(R^2/5 * trace(A'A) + |dt|^2)`` with ``A = M - I`` for the rotation
    between consecutive volumes: the RMS displacement of the points of a
    uniform ball of radius *R*, whose covariance is ``(R^2/5) I``.  The exact
    rotation matrices are used rather than the small-angle form, so the measure
    stays right for the large-motion volumes the thresholds are about.
    """
    trans, rot = motion_params(df)
    mats = rotation_matrices(rot)
    rel = mats[1:] @ np.swapaxes(mats[:-1], 1, 2)  # rotation from volume i to i+1
    a = rel - np.eye(3)
    trace = np.einsum("nij,nij->n", a, a)
    dt = np.diff(trans, axis=0)
    steps = np.sqrt(radius ** 2 / 5.0 * trace + (dt * dt).sum(axis=1))
    return np.concatenate([[np.nan], steps])


def measured_factor(fd, rms, min_volumes=5, min_rms=1e-3):
    """Per-scan median fd/rms, or None when the scan cannot support the estimate."""
    ok = np.isfinite(fd) & np.isfinite(rms) & (rms > min_rms)
    if ok.sum() < min_volumes:
        return None
    return float(np.median(fd[ok] / rms[ok]))


def motion_row(motion, n_volumes, exclude_cut, cuts, count_among):
    """The report's motion columns for one scan.

    *motion* holds only the volumes whose motion is defined (the first volume of
    every run is already dropped), while *n_volumes* counts every volume in the
    scan.  *cuts* maps each output column to the threshold applied to *motion*.
    """
    excluded = int(np.sum(motion > exclude_cut))
    row = {"remaining_number_of_gradients": n_volumes - excluded,
           "number_of_excluded_gradients": excluded}
    counted = motion[motion <= exclude_cut] if count_among == "remaining" else motion
    for name, cut in cuts.items():
        row[name] = int(np.sum(counted > cut))
    return row


DIR_ENTITY_RE = re.compile(r"(?:^|_)dir-[^_]+")


def pair_key(prefix):
    """The prefix with its phase-encoding ``dir-`` entity removed, or None if it has none."""
    stripped = DIR_ENTITY_RE.sub("", prefix, count=1).lstrip("_")
    return stripped if stripped != prefix else None


def merge_ap_pa(path, groups):
    """Merge runs of *groups* that differ only in their ``dir-`` entity.

    Returns ``[(original_file, index array, prefix override or None)]``; a merged
    group pools the volumes of its runs (so each count is the sum over the
    directions) and carries the direction-free prefix.
    """
    buckets, order = {}, []
    for original_file, idx in groups:
        ident = scan_identity(path, original_file)
        key = pair_key(ident[2]) if ident else None
        bucket_id = (ident[0], ident[1], key) if key else ("__single__", original_file, len(order))
        if bucket_id not in buckets:
            buckets[bucket_id] = []
            order.append(bucket_id)
        buckets[bucket_id].append((original_file, idx))

    merged = []
    for bucket_id in order:
        members = buckets[bucket_id]
        if len(members) == 1:
            merged.append((members[0][0], members[0][1], None))
            continue
        log.info("%s: summing %d phase-encoding directions (%s) into %s",
                 os.path.basename(path), len(members),
                 ", ".join(DIR_ENTITY_RE.search(scan_identity(path, of)[2]).group().lstrip("_")
                           for of, _ in members), bucket_id[2])
        merged.append((members[0][0], np.concatenate([idx for _, idx in members]), bucket_id[2]))
    return merged


def run_groups(df):
    """[(original_file, row index array)] in order of appearance; one group if absent."""
    if RUN_COLUMN not in df:
        return [(None, np.arange(len(df)))]
    runs = df[RUN_COLUMN].astype(str).to_numpy()
    starts = np.concatenate([[0], np.where(runs[1:] != runs[:-1])[0] + 1])
    ends = np.concatenate([starts[1:], [len(runs)]])
    return [(runs[a], np.arange(a, b)) for a, b in zip(starts, ends)]


def scan_rows(path, args):
    """One row dict per scan in a confounds file (per ``original_file`` run)."""
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        log.warning("%s has no volumes; skipping", path)
        return []

    if args.motion_measure == "rms":
        motion_all = rms_displacement(df, args.rms_radius)
    else:
        motion_all = framewise_displacement(df, args.fd_radius, args.recompute_fd)

    # fd/rms measured on this scan, when the conversion is both needed and asked for
    factor = args.rms_fd_factor
    if args.measure_factor and args.needs_factor:
        try:
            measured = measured_factor(framewise_displacement(df, args.fd_radius, args.recompute_fd),
                                       rms_displacement(df, args.rms_radius))
        except ValueError as exc:
            measured = None
            log.debug("%s: cannot measure fd/rms (%s)", path, exc)
        if measured is None:
            log.warning("%s: too little motion to measure fd/rms; using %.2f",
                        os.path.basename(path), factor)
        else:
            lo, hi = FACTOR_SANITY
            if not lo <= measured <= hi:
                log.warning("%s: measured fd/rms %.2f is implausible (outside [%.1f, %.1f]); "
                            "using %.2f", os.path.basename(path), measured, lo, hi, factor)
            else:
                factor = measured
                if not FACTOR_BOUNDS[0] <= measured <= FACTOR_BOUNDS[1]:
                    log.debug("%s: measured fd/rms %.2f sits outside the ideal L1/L2 band "
                              "(large cumulative rotation)", os.path.basename(path), measured)
                log.debug("%s: measured fd/rms = %.2f", os.path.basename(path), factor)

    scale = factor if args.needs_factor else 1.0
    cuts = {f"{args.threshold_units}_larger_than_{t:g}": t * scale for t in args.thresholds}
    exclude_cut = args.exclude_threshold * scale

    runs = run_groups(df)
    # The first volume of every run has no valid predecessor: its motion is
    # either undefined (the very first) or straddles two unrelated acquisitions.
    usable = np.isfinite(motion_all)
    for _, idx in runs:
        usable[idx[0]] = False

    groups = [(of, idx, None) for of, idx in runs]
    if len(runs) > 1:
        if args.combine:
            groups = [(runs[0][0], np.arange(len(df)), None)]
        elif args.sum_ap_pa:
            groups = merge_ap_pa(path, runs)
        if len(groups) > 1:
            log.info("%s: %d source runs -> %d row(s)", os.path.basename(path), len(runs), len(groups))

    rows = []
    for original_file, idx, prefix_override in groups:
        ident = scan_identity(path, original_file)
        if ident is None:
            log.warning("no sub-/ses- identity for %s; skipping", path)
            continue
        sub, ses, prefix = ident
        prefix = prefix_override or prefix
        motion = motion_all[idx][usable[idx]]

        row = {"sub": sub, "ses": ses, "prefix": prefix}
        row.update(motion_row(motion, len(idx), exclude_cut, cuts, args.count_among))
        if args.extra:
            row.update({
                "n_volumes": len(idx), "n_motion_defined": len(motion),
                "motion_mean": round(float(np.mean(motion)), 4) if motion.size else np.nan,
                "motion_median": round(float(np.median(motion)), 4) if motion.size else np.nan,
                "motion_max": round(float(np.max(motion)), 4) if motion.size else np.nan,
                "rms_fd_factor": round(scale, 4) if args.needs_factor else np.nan,
            })
        rows.append(row)
    return rows


def collect_inputs(paths, pattern):
    """Expand files, globs and directories (searched recursively) into TSV paths."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", pattern), recursive=True))
        elif any(ch in p for ch in "*?["):
            out += sorted(glob.glob(p, recursive=True))
        else:
            out.append(p)
    seen = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("confounds", nargs="+",
                   help="QSIPrep *_confounds.tsv file(s), a glob, or a folder to search")
    p.add_argument("-o", "--output", default=None,
                   help="Output CSV in DWIQC_report.csv format (default: write to stdout)")
    p.add_argument("--pattern", default="*confounds.tsv",
                   help="Filename pattern used when a folder is given (default: *confounds.tsv)")

    p.add_argument("--motion-measure", choices=["fd", "rms"], default="fd",
                   help="Per-volume motion: 'fd' (Power framewise displacement, QSIPrep's column) "
                        "or 'rms' (Jenkinson displacement from trans_*/rot_*, as FSL reports). "
                        "'rms' matches the report's own measure and needs no conversion factor "
                        "(default: fd)")
    p.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 2.0, 3.0], metavar="MM",
                   help="Thresholds for the *_larger_than_* columns, in --threshold-units "
                        "(default: 1 2 3, as in the report)")
    p.add_argument("--threshold-units", choices=["rms", "fd"], default="rms",
                   help="Units the thresholds are given in, and the name of the output columns "
                        "(default: rms, matching DWIQC_report.csv)")
    p.add_argument("--exclude-threshold", type=float, default=2.0, metavar="MM",
                   help="Motion (in --threshold-units) above which a volume counts as excluded "
                        "(default: 2.0); motion-only, so below DTIPrep's count, which also uses "
                        "intensity checks")
    p.add_argument("--count-among", choices=["all", "remaining"], default="all",
                   help="Count the *_larger_than_* volumes among all volumes or only the "
                        "remaining (non-excluded) ones (default: all)")

    p.add_argument("--rms-fd-factor", type=float, default=DEFAULT_RMS_FD_FACTOR, metavar="K",
                   help=f"fd/rms ratio converting RMS thresholds to FD cuts (default: "
                        f"{DEFAULT_RMS_FD_FACTOR}); used only with --motion-measure fd and "
                        f"--threshold-units rms. The L1/L2 norm inequality puts it in "
                        f"[1.00, {FACTOR_BOUNDS[1]:.2f}] for typical motion")
    p.add_argument("--measure-factor", action="store_true",
                   help="Measure fd/rms per scan from its own motion parameters instead of using "
                        "the fixed --rms-fd-factor")
    p.add_argument("--fd-radius", type=float, default=FD_RADIUS, metavar="MM",
                   help=f"Sphere radius converting rotations to mm for FD (default: {FD_RADIUS:g}, "
                        f"QSIPrep's)")
    p.add_argument("--rms-radius", type=float, default=RMS_RADIUS, metavar="MM",
                   help=f"Ball radius for the Jenkinson RMS (default: {RMS_RADIUS:g}, FSL's)")
    p.add_argument("--recompute-fd", action="store_true",
                   help="Recompute FD from trans_*/rot_* instead of using the "
                        f"'{FD_COLUMN}' column (needed for a non-default --fd-radius)")

    p.add_argument("--sum-ap-pa", action="store_true",
                   help="Merge rows of one confounds file that differ only in their dir- "
                        "(phase-encoding) entity, e.g. dir-AP and dir-PA, by summing their counts; "
                        "the merged prefix drops the dir- entity")
    p.add_argument("--combine", action="store_true",
                   help="One row per confounds file even when it concatenates several source "
                        "runs (default: one row per original_file, as the report is per-scan)")
    p.add_argument("--extra", action="store_true",
                   help="Add diagnostic columns (volume counts, mean/median/max motion, factor)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)
    if args.combine and args.sum_ap_pa:
        p.error("--combine already merges every run of a file; use one or the other")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s | %(message)s", stream=sys.stderr)

    # the RMS->FD conversion is needed only when the two disagree
    args.needs_factor = (args.motion_measure == "fd" and args.threshold_units == "rms")
    lo, hi = FACTOR_BOUNDS
    if args.rms_fd_factor <= 0:
        p.error("--rms-fd-factor must be positive")
    if not lo <= args.rms_fd_factor <= hi:
        log.warning("--rms-fd-factor %g sits outside the L1/L2 band [%.2f, %.2f] expected for "
                    "fd/rms; that is only plausible for heavily drifted rotations",
                    args.rms_fd_factor, lo, hi)
    if args.measure_factor and not args.needs_factor:
        log.warning("--measure-factor does nothing with --motion-measure %s and "
                    "--threshold-units %s: no conversion is applied",
                    args.motion_measure, args.threshold_units)
    if args.motion_measure == "fd" and args.threshold_units == "fd":
        log.info("thresholds are FD cuts; columns are named fd_larger_than_* rather than the "
                 "report's rms_larger_than_*")
    if not args.recompute_fd and args.fd_radius != FD_RADIUS:
        log.warning("--fd-radius %g applies only where FD is recomputed; files that carry a "
                    "'%s' column keep it -- add --recompute-fd to use the radius everywhere",
                    args.fd_radius, FD_COLUMN)
    if args.needs_factor:
        log.info("Thresholds in RMS mm -> FD cuts via fd/rms = %s: %s",
                 "measured per scan" if args.measure_factor else f"{args.rms_fd_factor:g}",
                 ", ".join(f"{t:g}->{t * args.rms_fd_factor:.2f}" for t in args.thresholds))

    files = collect_inputs(args.confounds, args.pattern)
    if not files:
        p.error(f"no confounds files found (pattern {args.pattern!r})")
    log.info("Reading %d confounds file(s)", len(files))

    rows, n_bad = [], 0
    for f in files:
        try:
            rows += scan_rows(f, args)
        except (OSError, ValueError, KeyError) as exc:
            log.warning("%s: %s; skipping", f, exc)
            n_bad += 1
    if not rows:
        log.error("no scans could be read from %d file(s)", len(files))
        return 1

    columns = BASE_COLUMNS + [f"{args.threshold_units}_larger_than_{t:g}" for t in args.thresholds]
    if args.extra:
        columns += EXTRA_COLUMNS
    df = pd.DataFrame(rows)[columns].sort_values(["sub", "ses", "prefix"])

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        df.to_csv(args.output, index=False)
        log.info("Wrote %d scan(s) to %s%s", len(df), args.output,
                 f" ({n_bad} file(s) skipped)" if n_bad else "")
    else:
        df.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
