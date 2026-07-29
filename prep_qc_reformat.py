#!/usr/bin/env python3
"""
Reformat the raw-DWI prep-QC report (motion / artifact info) into a form
compatible with the rest of the reporting.

The report comes in two styles: one with **two** image-name columns
(``image_name_1`` = a stringified list of the AP/PA NIfTIs, ``image_name_2`` =
the processed NRRD path), and one with a **single** ``image_name`` column.  Only
the ``image_name*`` column(s) are removed -- one or two, whichever are present --
and replaced by three columns ``sub``, ``ses``, ``prefix`` (matching the
``summary_*.csv`` convention; the ``sub-`` / ``ses-`` tags stripped).  All other
columns are kept unchanged.

Identity is read from a ``sub-<sub>_ses-<ses>_<prefix>`` path component.  The
**last** image column is preferred, as its processed path carries the canonical
pipeline prefix (e.g. ``acq-dir79select_dir_run-001_dwi``), whereas the raw NIfTI
filenames use the phase-encode ``dir-AP`` / ``dir-PA`` variant.

Usage
-----
    python prep_qc_reformat.py --input PrepQC/DWIQC_report.csv \\
        --output PrepQC/DWIQC_report_reformatted.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

ID_RE = re.compile(r"^sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_(?P<prefix>.+)$")


def extract_identity(*paths):
    """Return (sub, ses, prefix) from the first matching path component."""
    for path in paths:
        for comp in str(path).replace("\\", "/").split("/"):
            m = ID_RE.match(comp)
            if m:
                return m.group("sub"), m.group("ses"), m.group("prefix")
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="PrepQC/DWIQC_report.csv", help="Prep-QC CSV to reformat")
    p.add_argument("--output", default=None, help="Output CSV (default: <input>_reformatted.csv)")
    args = p.parse_args(argv)

    out_path = args.output or (os.path.splitext(args.input)[0] + "_reformatted.csv")

    df = pd.read_csv(args.input)
    # Only the image-name column(s) are replaced -- one style has two
    # (image_name_1/image_name_2), the other a single (image_name).
    image_cols = [c for c in df.columns if str(c).lower().startswith("image_name")]
    if not image_cols:
        p.error(f"no 'image_name' column(s) found to reformat; columns are {list(df.columns)}")
    # prefer the last image column (the processed path with the canonical prefix)
    pref = list(reversed(image_cols))

    subs, sess, prefixes, failed = [], [], [], []
    for i in range(len(df)):
        ident = extract_identity(*[df.iloc[i][c] for c in pref])
        if ident is None:
            failed.append(i)
            subs.append(""); sess.append(""); prefixes.append("")
        else:
            subs.append(ident[0]); sess.append(ident[1]); prefixes.append(ident[2])
    if failed:
        print(f"WARNING: could not extract identity for {len(failed)} row(s): {failed[:10]}", file=sys.stderr)

    rest = df.drop(columns=image_cols).reset_index(drop=True)
    out = pd.concat([pd.DataFrame({"sub": subs, "ses": sess, "prefix": prefixes}), rest], axis=1)
    out.to_csv(out_path, index=False)
    print(f"Reformatted {len(df)} rows ({len(image_cols)} image column(s) removed) -> {out_path} "
          f"(columns: {list(out.columns)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
