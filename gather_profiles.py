#!/usr/bin/env python3
"""
Gather along-tract FVP profiles across subjects/sessions into per-(tract, metric)
CSV tables.

Input
-----
FVP files produced per subject/session under a profiles tree, e.g.::

    Output_Profiles/<subject>/<session>/Profiles/
        <subject>_<session>_<prefix>_<tract>_<metric>.fvp

Each FVP file has a small text header followed by a CSV-style block whose header
line starts with ``Arc_Length,``.  The per-position metric value is taken from
the ``Parameter_Value`` column (configurable via --value-column).

Output
------
One CSV per (tract, metric), grouped into a folder per tract::

    <out>/<tract>/<tract>_<metric>.csv

Rows are the arc-length sample positions; columns are one per
``<subject>_<session>_<prefix>`` identifier (so multiple acquisitions/prefixes
in the same session each get their own column).  The column set is shared
across all metric tables of a tract: if an identifier has, say, ``fa`` but not
``NDI``, its column still appears in the ``NDI`` table, left blank.  Cells are
the metric value; genuinely missing values (e.g. source ``-nan``) are blank.

By default every tract and metric found in the FVP tree is gathered.
``--metrics`` limits the run to a given list of metrics (names are the final
filename token, e.g. ``fa``, ``md``, ``NDI``, ``FWF``) and ``--tracts`` to a
given list of tracts (names as in the fiber VTK files, e.g. ``Fornix_L``).
Both match case-insensitively.

Usage
-----
    python gather_profiles.py --profiles-dir Output_Profiles --out-dir Profiles_CSV
    python gather_profiles.py --metrics fa md NDI
    python gather_profiles.py --tracts Fornix_L Fornix_R --metrics fa
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from collections import defaultdict

import pandas as pd

log = logging.getLogger("profiles")

FVP_HEADER_PREFIX = "Arc_Length,"


def load_tract_vocabulary(fibers_dir: str) -> list:
    """Tract names from the fiber VTK files, longest first for suffix matching."""
    if not fibers_dir or not os.path.isdir(fibers_dir):
        log.warning("fibers dir '%s' not found; falling back to '_dwi_' name splitting", fibers_dir)
        return []
    tracts = [os.path.splitext(f)[0] for f in os.listdir(fibers_dir) if f.lower().endswith(".vtk")]
    return sorted(set(tracts), key=len, reverse=True)


def parse_fvp_name(path: str, tracts: list):
    """Extract (subject, session, prefix, tract, metric) from an FVP filename.

    subject/session are the first two BIDS tokens; the metric is the final
    token; the tract is the longest known tract name that is a suffix of the
    middle (prefix + tract) span; the prefix is whatever precedes the tract in
    that span (e.g. ``acq-dir79select_dir_run-001_dwi``).  Falls back to
    splitting on the last ``_dwi_`` when no tract vocabulary matches.
    """
    base = os.path.basename(path)
    if base.endswith(".fvp"):
        base = base[:-4]
    toks = base.split("_")
    if len(toks) < 4:
        return None
    subject, session, metric = toks[0], toks[1], toks[-1]
    middle = "_".join(toks[2:-1])  # prefix + tract

    tract = None
    for t in tracts:  # longest-first -> unambiguous (Fornix_L vs Fornix_narrow_L)
        if middle == t or middle.endswith("_" + t):
            tract = t
            break
    if tract is not None:
        prefix = middle[: -len(tract)].rstrip("_") if middle != tract else ""
    elif "_dwi_" in base:  # fallback when the tract vocabulary is unavailable
        prefix, tail = base.split("_dwi_", 1)
        prefix = "_".join(prefix.split("_")[2:] + ["dwi"])  # drop subject/session tokens
        tract = tail.rsplit("_", 1)[0]
    else:
        return None
    return subject, session, prefix, tract, metric


def read_fvp_profile(path: str, value_col: str, arc_precision: int = 4):
    """Return {arc_length_str: value} for one FVP file, or None if unparseable.

    Arc-length keys are formatted strings (rounded to *arc_precision* decimals)
    so that columns align exactly across files -- float keys are unreliable for
    alignment / CSV round-tripping.
    """
    try:
        with open(path) as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        log.warning("could not read %s (%s)", path, exc)
        return None

    hidx = next((i for i, l in enumerate(lines) if l.startswith(FVP_HEADER_PREFIX)), None)
    if hidx is None:
        log.warning("no '%s' header in %s; skipping", FVP_HEADER_PREFIX, path)
        return None

    header = lines[hidx].split(",")
    try:
        ai = header.index("Arc_Length")
        vi = header.index(value_col)
    except ValueError:
        log.warning("column '%s' not found in %s (has %s); skipping", value_col, path, header)
        return None

    profile = {}
    for row in lines[hidx + 1:]:
        if not row.strip():
            continue
        parts = row.split(",")
        if len(parts) <= max(ai, vi):
            continue
        try:
            arc = round(float(parts[ai]), arc_precision) + 0.0  # +0.0 normalises -0.0
            profile[f"%.{arc_precision}f" % arc] = float(parts[vi])
        except ValueError:
            continue
    return profile or None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profiles-dir", default="Output_Profiles", help="Root of the per-subject FVP tree")
    p.add_argument("--fibers-dir", default="Fibers", help="Folder of tract VTK files (tract-name vocabulary)")
    p.add_argument("--out-dir", default="Profiles_CSV", help="Output root (one subfolder per tract)")
    p.add_argument(
        "--value-column",
        default="Parameter_Value",
        help="FVP column to tabulate (default: Parameter_Value)",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        metavar="METRIC",
        help="Only gather these metrics (e.g. fa md NDI); default: every metric found",
    )
    p.add_argument(
        "--tracts",
        nargs="+",
        metavar="TRACT",
        help="Only gather these tracts (e.g. Fornix_L Fornix_R); default: every tract found",
    )
    p.add_argument("--arc-precision", type=int, default=4, help="Decimals for arc-length column alignment")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s" if args.verbose else "%(message)s",
        stream=sys.stdout,
    )

    if not os.path.isdir(args.profiles_dir):
        p.error(f"profiles dir not found: {args.profiles_dir}")

    wanted_metrics = {m.lower() for m in args.metrics} if args.metrics else None
    if wanted_metrics:
        log.info("Restricting to metrics: %s", ", ".join(sorted(wanted_metrics)))

    wanted_tracts = {t.lower() for t in args.tracts} if args.tracts else None
    if wanted_tracts:
        log.info("Restricting to tracts: %s", ", ".join(sorted(wanted_tracts)))

    tracts = load_tract_vocabulary(args.fibers_dir)
    log.info("Tract vocabulary: %d names from %s", len(tracts), args.fibers_dir)

    if wanted_tracts and tracts:  # typo check against the known tract names
        unknown = sorted(wanted_tracts - {t.lower() for t in tracts})
        if unknown:
            log.warning("requested tract(s) not in %s: %s", args.fibers_dir, ", ".join(unknown))

    files = sorted(glob.glob(os.path.join(args.profiles_dir, "**", "*.fvp"), recursive=True))
    log.info("Found %d FVP files under %s", len(files), args.profiles_dir)
    if not files:
        return 0

    # (tract, metric) -> { identifier -> { arc_length -> value } }
    # identifier = "<subject>_<session>_<prefix>" (prefix kept so multiple
    # acquisitions in one session become separate columns).
    groups: dict = defaultdict(dict)
    tract_idents: dict = defaultdict(set)  # tract -> all identifiers seen (any metric)
    n_ok = n_skip = n_filtered = 0
    for f in files:
        parsed = parse_fvp_name(f, tracts)
        if parsed is None:
            log.warning("could not parse tract/metric from %s; skipping", os.path.basename(f))
            n_skip += 1
            continue
        subject, session, prefix, tract, metric = parsed
        if wanted_tracts is not None and tract.lower() not in wanted_tracts:
            n_filtered += 1
            continue
        if wanted_metrics is not None and metric.lower() not in wanted_metrics:
            n_filtered += 1
            continue
        profile = read_fvp_profile(f, args.value_column, args.arc_precision)
        if profile is None:
            n_skip += 1
            continue
        ident = "_".join(x for x in (subject, session, prefix) if x)
        tract_idents[tract].add(ident)
        if ident in groups[(tract, metric)]:
            log.warning("duplicate identifier %s for (%s, %s); overwriting", ident, tract, metric)
        groups[(tract, metric)][ident] = profile
        n_ok += 1

    active_filters = [
        flag for flag, on in (("--tracts", wanted_tracts), ("--metrics", wanted_metrics)) if on is not None
    ]
    all_idents = sorted(set().union(*tract_idents.values())) if tract_idents else []
    log.info(
        "Parsed %d profiles (%d skipped%s); %d identifiers, %d (tract,metric) tables",
        n_ok, n_skip,
        f", {n_filtered} excluded by {'/'.join(active_filters)}" if active_filters else "",
        len(all_idents), len(groups),
    )
    if active_filters and not groups and files:
        log.warning("no FVP files matched the requested %s", " / ".join(active_filters))

    os.makedirs(args.out_dir, exist_ok=True)
    n_written = 0
    for (tract, metric), per_ident in sorted(groups.items()):
        # Columns = every identifier seen for this tract (blank where this
        # metric is absent for that identifier). Rows = arc-length positions.
        columns = sorted(tract_idents[tract])
        arc_rows = sorted(
            {arc for prof in per_ident.values() for arc in prof}, key=float
        )
        data = {
            ident: [per_ident.get(ident, {}).get(arc, float("nan")) for arc in arc_rows]
            for ident in columns
        }
        df = pd.DataFrame(data, index=arc_rows, columns=columns)
        df.index.name = "Arc_Length"

        tract_dir = os.path.join(args.out_dir, tract)
        os.makedirs(tract_dir, exist_ok=True)
        out_csv = os.path.join(tract_dir, f"{tract}_{metric}.csv")
        df.to_csv(out_csv)
        n_written += 1
        n_blank_cols = sum(1 for c in columns if c not in per_ident)
        log.debug(
            "wrote %s [%d arc rows x %d ident cols, %d blank]",
            out_csv, df.shape[0], df.shape[1], n_blank_cols,
        )

    log.info("Wrote %d CSV files under %s/", n_written, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
