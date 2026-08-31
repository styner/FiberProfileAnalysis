#!/usr/bin/env python3
"""
Cohort-level sanity check / pruning of along-tract FVP profile files.

Python port of ``CheckProfile_Cohort.script`` (tcsh).  For every tract in the
atlas fiber set, it scans each subject/session's ``Profiles`` folder for the FVP
files of the standard metrics and:

  * removes an FVP with fewer than ``--min-lines`` lines (too short);
  * for non-FWF metrics, removes an FVP whose profile is essentially all zero
    (fewer than ``--min-lines`` non ``,0,0,0,0`` lines) -- disable with
    ``--no-zero-check``;
  * reports (but does NOT remove) FVPs that contain ``nan`` -- these are left
    for later imputation.

Actions are appended to ``<proc_dir>/check_Fibers_<YYYY-MM-DD>.txt``.

Usage
-----
    python CheckProfile_Cohort.py <base_dir> <proc_dir> [--atlas-dir DIR]
                                  [--no-zero-check] [--dry-run]

``<base_dir>`` holds ``sub*/ses*/Profiles/*.fvp``; ``<proc_dir>`` receives the
report.  Removal is real by default (as in the original); use ``--dry-run`` to
only log what would be removed.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import sys

# Metrics scanned, matching the original brace expansion (case-sensitive).
METRICS = ["fa", "md", "rd", "ad", "NDI", "ODI", "FWF", "FW_FA", "FW_AD", "FW_RD", "FW_MD"]

DEFAULT_ATLAS_DIR = "/proj/NIRAL/studies/ORIGIN/atlas/DTI_IBISEP_Feb26/FibersParam"


def tract_name(vtk_path):
    """basename, drop extension, drop a '_parametrized' substring (as in tcsh)."""
    name = os.path.splitext(os.path.basename(vtk_path))[0]
    return name.replace("_parametrized", "")


def count_lines(text):
    """Number of lines, matching ``wc -l`` (counts newline characters)."""
    return text.count("\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base_dir", help="Cohort root with sub*/ses*/Profiles/*.fvp")
    p.add_argument("proc_dir", help="Folder for the report file")
    p.add_argument("--atlas-dir", default=DEFAULT_ATLAS_DIR,
                   help="Folder of atlas fiber *.vtk defining the tract list")
    p.add_argument("--min-lines", type=int, default=10, help="Minimum FVP line count to keep (default: 10)")
    p.add_argument("--no-zero-check", dest="zero_check", action="store_false",
                   help="Skip the all-zero profile check (only the line-count and nan checks are done)")
    p.add_argument("--dry-run", action="store_true", help="Only log what would be removed; delete nothing")
    args = p.parse_args(argv)

    date = datetime.date.today().isoformat()
    os.makedirs(args.proc_dir, exist_ok=True)
    report_path = os.path.join(args.proc_dir, f"check_Fibers_{date}.txt")
    report = open(report_path, "a")

    def log(msg):
        report.write(msg + "\n")
        report.flush()

    print(f"checking profile FVP data {args.base_dir}")

    tracts = sorted(glob.glob(os.path.join(args.atlas_dir, "*.vtk")))
    if not tracts:
        print(f"WARNING: no atlas tracts found in {args.atlas_dir}", file=sys.stderr)

    n_removed = n_nan = 0
    for vtk in tracts:
        tract = tract_name(vtk)
        print(tract)

        for metric in METRICS:
            # files containing the tract name and ending in <metric>.fvp
            hits = glob.glob(os.path.join(args.base_dir, "sub*", "ses*", "Profiles", f"*{tract}*.fvp"))
            for fvp in sorted(hits):
                base = os.path.basename(fvp)
                if not base.endswith(f"{metric}.fvp"):   # case-sensitive, disjoint per metric
                    continue
                if not os.path.exists(fvp):               # may have been removed under another tract
                    continue

                with open(fvp, errors="replace") as fh:
                    text = fh.read()
                lines = text.split("\n")
                total = count_lines(text)

                # (1) too few lines
                if total < args.min_lines:
                    log(f"bad fvp, less than {args.min_lines} in profile - removing {fvp}")
                    if args.dry_run:
                        log(f"dryrun - no removal {fvp}")
                    else:
                        os.remove(fvp)
                    n_removed += 1
                    continue

                # (2) all-zero profile (non-FWF metrics only)
                if args.zero_check and "_FWF" not in base:
                    n_zero = sum(1 for ln in lines if ",0,0,0,0" in ln)
                    n_nonzero = total - n_zero
                    if n_nonzero < args.min_lines:
                        log(f"bad fvp, all 0 in profile - removing {fvp}")
                        if args.dry_run:
                            log(f"dryrun - no removal {fvp}")
                        else:
                            os.remove(fvp)
                        n_removed += 1
                        continue

                # (3) nan data -> report only, keep for later imputation
                if any("nan" in ln for ln in lines):
                    log(f"FYI fvp with nan data {fvp} - NOT removing {fvp} , impute nan data later")
                    n_nan += 1

    report.close()
    print(f"done: {'would remove' if args.dry_run else 'removed'} {n_removed} FVP, "
          f"{n_nan} with nan (kept); report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
