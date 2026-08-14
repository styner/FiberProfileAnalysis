#!/usr/bin/env python3
"""
Strip all non-baseline (diffusion-weighted) volumes from a DWI and keep only the
baseline (b~=0) volumes.

Supports both formats:
  * NIfTI (``.nii`` / ``.nii.gz``) -- b-values come from the FSL ``.bval`` sidecar
    (inferred from the image name or given with ``--bval``); the matching
    ``.bvec`` is filtered too when present.
  * NRRD (``.nrrd`` / ``.nhdr``) -- b-values come from the embedded gradient keys
    (``DWMRI_b-value`` and ``DWMRI_gradient_XXXX``), where the per-volume b-value
    is ``b_ref * ||gradient||^2``; kept gradient keys are renumbered in the output
    header.

A volume is "baseline" when its b-value is <= ``--threshold`` (default 10).

Usage
-----
    python strip_to_baseline.py input_dwi.nii.gz
    python strip_to_baseline.py input_dwi.nrrd --output baseline.nrrd --threshold 50
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _default_output(inp):
    for ext in (".nii.gz", ".nii", ".nrrd", ".nhdr"):
        if inp.endswith(ext):
            return inp[: -len(ext)] + "_baseline" + ext
    root, ext = os.path.splitext(inp)
    return root + "_baseline" + ext


def _infer_grad(path, ext):
    for suffix in (".nii.gz", ".nii"):
        if path.endswith(suffix):
            return path[: -len(suffix)] + ext
    return os.path.splitext(path)[0] + ext


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------
def strip_nifti(inp, out, threshold, bval_path, bvec_path):
    import nibabel as nib

    bval_path = bval_path or _infer_grad(inp, ".bval")
    if not os.path.isfile(bval_path):
        sys.exit(f"bval file not found: {bval_path} (give it with --bval)")
    bvals = np.atleast_1d(np.loadtxt(bval_path).astype(float))

    img = nib.load(inp)
    if img.ndim < 4:
        print(f"{inp} is 3D (no volumes to strip); copying as-is")
        keep = np.array([0])
    else:
        n_vol = img.shape[3]
        if len(bvals) != n_vol:
            sys.exit(f"bval count {len(bvals)} != number of volumes {n_vol}")
        keep = np.where(bvals <= threshold)[0]
        if keep.size == 0:
            sys.exit(f"no baseline volumes (b<={threshold}); min b-value is {bvals.min():g}")

    data = np.asanyarray(img.dataobj)
    out_data = data[..., keep] if img.ndim >= 4 else data
    nib.save(nib.Nifti1Image(out_data, img.affine, img.header), out)

    # filtered bval/bvec alongside the output (all ~baseline)
    bval_out = _infer_grad(out, ".bval")
    np.savetxt(bval_out, bvals[keep][None, :], fmt="%g")
    written = [out, bval_out]
    bvec_path = bvec_path or _infer_grad(inp, ".bvec")
    if os.path.isfile(bvec_path):
        bvecs = np.loadtxt(bvec_path)
        if bvecs.ndim == 2 and bvecs.shape[0] == 3:            # 3 x N (FSL)
            bvecs_out = bvecs[:, keep]
        elif bvecs.ndim == 2 and bvecs.shape[1] == 3:          # N x 3
            bvecs_out = bvecs[keep, :]
        else:
            bvecs_out = None
        if bvecs_out is not None:
            bvec_out = _infer_grad(out, ".bvec")
            np.savetxt(bvec_out, bvecs_out, fmt="%.8f")
            written.append(bvec_out)
    return keep, written


# ---------------------------------------------------------------------------
# NRRD
# ---------------------------------------------------------------------------
def _grad_axis(data, n_grad, header):
    """Index of the gradient (list) axis in the NRRD data array."""
    kinds = header.get("kinds")
    if kinds:
        for ax, k in enumerate(kinds):
            if str(k).lower() in ("list", "vector"):
                return ax
    matches = [ax for ax, s in enumerate(data.shape) if s == n_grad]
    if len(matches) != 1:
        sys.exit(f"cannot identify the gradient axis (shape {data.shape}, {n_grad} gradients)")
    return matches[0]


def strip_nrrd(inp, out, threshold):
    import nrrd

    data, header = nrrd.read(inp)
    grad_items = sorted(
        ((int(k.rsplit("_", 1)[1]), k) for k in header if k.startswith("DWMRI_gradient_")),
        key=lambda t: t[0],
    )
    if not grad_items:
        sys.exit(f"{inp} has no DWMRI_gradient_* header keys (not a NRRD DWI); use a NIfTI + bval instead")
    if "DWMRI_b-value" not in header:
        sys.exit(f"{inp} has no DWMRI_b-value header key")
    b_ref = float(str(header["DWMRI_b-value"]).split()[0])
    grads = [np.asarray(str(header[k]).split(), dtype=float) for _, k in grad_items]
    bvals = np.array([b_ref * float(g @ g) for g in grads])   # b_i = b_ref * ||g_i||^2
    n_grad = len(bvals)

    axis = _grad_axis(data, n_grad, header)
    keep = np.where(bvals <= threshold)[0]
    if keep.size == 0:
        sys.exit(f"no baseline volumes (b<={threshold}); min b-value is {bvals.min():g}")

    out_data = np.ascontiguousarray(np.take(data, keep, axis=axis))

    new_header = dict(header)
    new_header.pop("sizes", None)  # let pynrrd recompute from data
    for _, k in grad_items:        # drop old gradient keys
        new_header.pop(k, None)
    for new_i, old_i in enumerate(keep):            # re-add kept gradients, renumbered
        new_header[f"DWMRI_gradient_{new_i:04d}"] = header[grad_items[old_i][1]]
    nrrd.write(out, out_data, new_header)
    return keep, [out]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="DWI file (.nii/.nii.gz or .nrrd/.nhdr)")
    p.add_argument("--output", default=None, help="Output path (default: <input>_baseline.<ext>)")
    p.add_argument("--threshold", type=float, default=10.0,
                   help="A volume is baseline when its b-value <= this (default: 10)")
    p.add_argument("--bval", default=None, help="FSL bval for NIfTI (default: inferred)")
    p.add_argument("--bvec", default=None, help="FSL bvec for NIfTI (default: inferred)")
    args = p.parse_args(argv)

    if not os.path.isfile(args.input):
        p.error(f"input not found: {args.input}")
    out = args.output or _default_output(args.input)

    if args.input.endswith((".nrrd", ".nhdr")):
        keep, written = strip_nrrd(args.input, out, args.threshold)
    elif args.input.endswith((".nii", ".nii.gz")):
        keep, written = strip_nifti(args.input, out, args.threshold, args.bval, args.bvec)
    else:
        p.error(f"unsupported extension: {args.input}")

    print(f"kept {len(keep)} baseline volume(s) (b<={args.threshold:g}); wrote: {', '.join(written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
