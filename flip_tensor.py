#!/usr/bin/env python3
"""
Flip (reflect) the orientation frame of a diffusion tensor NRRD along one or
more axes and write it back out.

This reflects the *tensor directions*, not the spatial voxel layout -- i.e. it
applies the reflection ``R = diag(sx, sy, sz)`` to every voxel's tensor,
``D' = R D Rᵀ``.  For a symmetric tensor stored as the 6 unique components
``[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]`` this simply multiplies each component
``D_ij`` by ``s_i · s_j`` -- so a single-axis flip negates that axis's two
off-diagonal terms and leaves the diagonal (and hence FA/MD/eigenvalues)
unchanged.  It is the operation that corrects a tensor-frame handedness /
LPS-RAS sign-convention mismatch (e.g. the x-flip found by ``registration_QC``).

Usage
-----
    python flip_tensor.py input_DTI.nrrd output_DTI.nrrd --axes x
    python flip_tensor.py in.nrrd out.nrrd --axes x,z
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import nrrd

# 6-component symmetric-matrix order -> (row, col) index pairs
COMP_IJ = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
AXIS = {"x": 0, "y": 1, "z": 2}


def flip_tensor(data: np.ndarray, axes) -> np.ndarray:
    """Reflect the tensor frame along *axes* (list of 'x'/'y'/'z').

    *data* is the 6-component symmetric tensor field with the component axis
    first (shape ``(6, ...)``).
    """
    signs = np.ones(3, dtype=data.dtype)
    for a in axes:
        signs[AXIS[a]] = -1.0
    out = data.copy()
    for c, (i, j) in enumerate(COMP_IJ):
        factor = signs[i] * signs[j]
        if factor != 1.0:
            out[c] *= factor
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Input tensor NRRD (6-component symmetric matrix, component axis first)")
    p.add_argument("output", help="Output tensor NRRD")
    p.add_argument("--axes", required=True,
                   help="Axes to flip: comma-separated among x,y,z (e.g. 'x' or 'x,z')")
    args = p.parse_args(argv)

    axes = [a.strip().lower() for a in args.axes.split(",") if a.strip()]
    bad = [a for a in axes if a not in AXIS]
    if bad:
        p.error(f"invalid axes {bad}; choose from x,y,z")

    data, header = nrrd.read(args.input)
    if data.ndim != 4 or data.shape[0] != 6:
        p.error(f"expected a 6-component tensor (6, X, Y, Z); got shape {data.shape}")

    flipped = flip_tensor(data, axes)
    nrrd.write(args.output, flipped, header)  # header preserved verbatim
    print(f"Flipped tensor axes {axes}; wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
