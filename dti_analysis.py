#!/usr/bin/env python3
"""
Diffusion tensor estimation and scalar map computation for a diffusion-MRI
dataset (NIfTI + FSL bval/bvec).

Pipeline
--------
1.  Load the DWI volume and the FSL gradient table (``.bval`` / ``.bvec``).
2.  Compute a brain mask (otsu / SynthSeg / SynthStrip) from a structural
    input image -- either the mean b=0 or, optionally, the axial-diffusivity
    (AD) map of an unmasked fit -- with optional N4 (ITK) bias correction.
3.  Fit the diffusion tensor with *weighted least squares* (WLS) using DIPY.
4.  Derive the standard scalar maps: FA, MD, RD, AD.
5.  Write the tensor to NRRD (teem / 3D-Slicer compatible symmetric-matrix
    format) and the four scalar maps to NIfTI.
6.  If >=2 shells are present, fit NODDI with AMICO (using all shells) and
    write NDI / ODI / FWF (+ direction) maps to NIfTI.
7.  If >=2 shells are present, fit free-water-corrected DTI (fwDTI, DIPY) using
    all shells and write FWFA / FWMD / FWRD / FWAD / FWf (+ tensor NRRD).

Note
----
The task asked for "FA, MR, RD, AD". There is no standard tensor metric named
"MR"; the canonical fourth scalar alongside FA/RD/AD is MD (mean diffusivity),
so MD is what is computed here.

Usage
-----
    python dti_analysis.py --dwi Example/<subject>_dwi_QCed.nii.gz --out results

If ``--bval`` / ``--bvec`` are omitted they are inferred from the DWI path by
swapping the extension.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext

import numpy as np
import nibabel as nib
import nrrd

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
import dipy.reconst.dti as dti

log = logging.getLogger("dti")


def setup_logging(verbose: bool) -> None:
    """Route logging to stdout; DEBUG (with timestamps) when *verbose*."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s" if verbose else "%(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def thread_limits(n: int | None):
    """Context manager capping BLAS/OpenMP threads for the DIPY (numpy) fit.

    Returns a no-op context when *n* is None/<=0, so the numeric libraries use
    their own defaults (typically all cores).
    """
    if n and n > 0:
        import threadpoolctl

        return threadpoolctl.threadpool_limits(limits=n)
    return nullcontext()


def _img_stats(a) -> str:
    """Compact one-line summary of an array for debug logging."""
    a = np.asarray(a)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return f"shape={tuple(a.shape)} dtype={a.dtype} (no finite values)"
    nz = int(np.count_nonzero(finite))
    return (
        f"shape={tuple(a.shape)} dtype={a.dtype} min={finite.min():.4g} "
        f"max={finite.max():.4g} mean={finite.mean():.4g} nonzero={nz} "
        f"nonfinite={a.size - finite.size}"
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def infer_grad_path(dwi_path: str, ext: str) -> str:
    """Replace the (possibly double) extension of *dwi_path* with *ext*."""
    base = dwi_path
    for suffix in (".nii.gz", ".nii"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    else:
        base = os.path.splitext(base)[0]
    return base + ext


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Tidy a binary mask (fill holes, open, keep largest component)."""
    try:  # scipy ships with dipy
        from scipy import ndimage as ndi

        raw = int(mask.sum())
        mask = ndi.binary_fill_holes(mask)
        mask = ndi.binary_opening(mask, iterations=1)
        lbl, n = ndi.label(mask)
        log.debug("cleanup: %d raw voxels, %d connected components", raw, n)
        if n > 1:
            sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
            mask = lbl == (int(np.argmax(sizes)) + 1)
            log.debug("cleanup: kept largest component (%d voxels)", int(mask.sum()))
    except Exception as exc:  # pragma: no cover
        log.debug("cleanup skipped (%s)", exc)
    return mask.astype(bool)


def mean_b0(data: np.ndarray, b0s_mask: np.ndarray) -> np.ndarray:
    """Mean of the b=0 volumes as a 3D float32 image."""
    return data[..., b0s_mask].mean(axis=-1).astype(np.float32)


def n4_bias_correct(image: np.ndarray, voxel_sizes) -> np.ndarray:
    """N4 bias-field correction (ITK, via SimpleITK) of a 3D image.

    A rough Otsu foreground mask restricts the bias-field estimation to the
    object.  Returns the corrected image in the same array layout and scale.
    """
    import SimpleITK as sitk

    t0 = time.perf_counter()
    arr = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    arr = np.clip(arr, 0.0, None)  # N4 works in the log domain -> non-negative
    log.debug("N4 input: %s; spacing=%s mm", _img_stats(arr), [round(float(v), 3) for v in voxel_sizes])
    # numpy is (X, Y, Z); SimpleITK expects (Z, Y, X)
    sitk_img = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))
    sitk_img.SetSpacing([float(v) for v in voxel_sizes])

    fg = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
    fg_voxels = int(np.count_nonzero(sitk.GetArrayFromImage(fg)))
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    log.debug(
        "N4 foreground=%d voxels, max iterations=%s",
        fg_voxels,
        list(corrector.GetMaximumNumberOfIterations()),
    )
    corrected = corrector.Execute(sitk_img, fg)

    out = sitk.GetArrayFromImage(corrected).transpose(2, 1, 0)
    out = np.ascontiguousarray(out).astype(np.float32)
    log.debug("N4 output: %s (%.1fs)", _img_stats(out), time.perf_counter() - t0)
    return out


def brain_mask_otsu(image: np.ndarray) -> np.ndarray:
    """Otsu brain mask from a 3D structural image, with hole filling."""
    from dipy.segment.threshold import otsu

    thr = otsu(image)
    log.debug("otsu threshold = %.4g", thr)
    return _largest_component(image > thr)


def brain_mask_synthseg(
    image: np.ndarray, affine: np.ndarray, use_cuda: bool = False, threads: int | None = None
) -> np.ndarray:
    """Brain mask via DIPY's SynthSeg deep-learning segmentation.

    SynthSeg is contrast-agnostic, so it segments the provided 3D image
    directly.  *threads* caps PyTorch's intra-op CPU threads (ignored on GPU).
    Requires PyTorch and downloads pretrained weights on first use.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    from dipy.nn.torch.synthseg import SynthSeg

    if threads and threads > 0 and not use_cuda:
        torch.set_num_threads(threads)
        log.debug("SynthSeg: torch intra-op threads set to %d", threads)
    log.debug("SynthSeg: use_cuda=%s, input %s", use_cuda, _img_stats(image))
    t0 = time.perf_counter()
    seg = SynthSeg(use_cuda=use_cuda)
    seg.fetch_default_weights()
    _labels, _label_dict, mask = seg.predict(image.astype(np.float32), affine)
    mask = np.asarray(mask) > 0
    log.debug("SynthSeg: %d voxels (%.1fs)", int(mask.sum()), time.perf_counter() - t0)
    return mask


def brain_mask_synthstrip(
    image: np.ndarray,
    affine: np.ndarray,
    exec_path: str = "mri_synthstrip",
    threads: int | None = None,
) -> np.ndarray:
    """Brain mask via FreeSurfer's SynthStrip, called as an external process.

    The provided 3D image is written to a temporary NIfTI and passed to
    ``mri_synthstrip -i <in> -o <stripped> -m <mask> [-t <threads>]``.
    SynthStrip is contrast-agnostic, so it strips the DWI b0 (or AD) image
    directly.  *threads* maps to SynthStrip's ``-t/--threads`` (None -> let
    SynthStrip use its own default).  Requires a working ``mri_synthstrip`` on
    the system (native FreeSurfer install); this code does not install
    FreeSurfer.
    """
    if shutil.which(exec_path) is None and not os.path.isfile(exec_path):
        raise FileNotFoundError(
            f"SynthStrip executable not found: '{exec_path}'. Install FreeSurfer's "
            "mri_synthstrip and ensure it is on PATH, or pass --synthstrip-exec."
        )

    workdir = tempfile.mkdtemp(prefix="synthstrip_")
    in_path = os.path.join(workdir, "input.nii.gz")
    stripped_path = os.path.join(workdir, "stripped.nii.gz")
    mask_path = os.path.join(workdir, "mask.nii.gz")
    log.debug("SynthStrip: exec=%s, workdir=%s", shutil.which(exec_path) or exec_path, workdir)
    log.debug("SynthStrip: input %s", _img_stats(image))
    try:
        nib.save(nib.Nifti1Image(image.astype(np.float32), affine), in_path)
        cmd = [exec_path, "-i", in_path, "-o", stripped_path, "-m", mask_path]
        if threads is not None:
            cmd += ["-t", str(threads)]
        log.info("      running: %s", " ".join(cmd))
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log.debug("SynthStrip: exit=%d in %.1fs", proc.returncode, time.perf_counter() - t0)
        if proc.stdout.strip():
            log.debug("SynthStrip stdout:\n%s", proc.stdout.strip())
        if proc.stderr.strip():
            log.debug("SynthStrip stderr:\n%s", proc.stderr.strip())
        if proc.returncode != 0:
            raise RuntimeError(
                f"mri_synthstrip failed (exit {proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        mask = np.asarray(nib.load(mask_path).dataobj) > 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return mask


def compute_brain_mask(
    image: np.ndarray,
    affine: np.ndarray,
    method: str,
    synthstrip_exec: str,
    use_cuda: bool,
    threads: int | None = None,
) -> np.ndarray:
    """Dispatch to the requested masking backend for a 3D input image."""
    log.debug("dispatch masking: method=%s, input %s", method, _img_stats(image))
    if method == "synthseg":
        return brain_mask_synthseg(image, affine, use_cuda=use_cuda, threads=threads)
    if method == "synthstrip":
        return brain_mask_synthstrip(image, affine, exec_path=synthstrip_exec, threads=threads)
    return brain_mask_otsu(image)


def write_tensor_nrrd(path: str, tensor_quad: np.ndarray, affine: np.ndarray) -> None:
    """Write a per-voxel 3x3 tensor field to NRRD as a symmetric matrix.

    The teem / 3D-Slicer convention stores the six unique components first,
    ordered ``[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]``, with ``kinds`` flagging the
    leading axis as a 3D symmetric matrix.  Spatial geometry (orientation,
    spacing, origin) is taken from the NIfTI affine, which is RAS by
    definition, so the NRRD ``space`` is right-anterior-superior.
    """
    # tensor_quad: (X, Y, Z, 3, 3) -> (6, X, Y, Z)
    comps = np.stack(
        [
            tensor_quad[..., 0, 0],  # Dxx
            tensor_quad[..., 0, 1],  # Dxy
            tensor_quad[..., 0, 2],  # Dxz
            tensor_quad[..., 1, 1],  # Dyy
            tensor_quad[..., 1, 2],  # Dyz
            tensor_quad[..., 2, 2],  # Dzz
        ],
        axis=0,
    ).astype(np.float32)

    # Per-axis world-space direction vectors; the component axis has no
    # spatial direction ("none" -> NaN row for pynrrd).
    space_directions = np.full((4, 3), np.nan, dtype=float)
    space_directions[1] = affine[:3, 0]
    space_directions[2] = affine[:3, 1]
    space_directions[3] = affine[:3, 2]

    header = {
        "space": "right-anterior-superior",
        "space directions": space_directions,
        "space origin": affine[:3, 3].astype(float),
        "kinds": ["3D-symmetric-matrix", "domain", "domain", "domain"],
        "measurement frame": np.eye(3),
        "space units": ["mm", "mm", "mm"],
    }
    log.debug("NRRD: components %s, kinds=%s, space=%s", comps.shape, header["kinds"], header["space"])
    log.debug("NRRD space directions:\n%s", np.array2string(space_directions, precision=4))
    nrrd.write(path, comps, header)


# ---------------------------------------------------------------------------
# NODDI (AMICO)
# ---------------------------------------------------------------------------
# AMICO map name -> (output suffix, description). NDI==ICVF, FWF==ISOVF.
_NODDI_MAPS = {
    "NDI": ("NDI", "Neurite Density Index (intra-cellular volume fraction)"),
    "ODI": ("ODI", "Orientation Dispersion Index"),
    "FWF": ("FWF", "Free Water Fraction (isotropic volume fraction)"),
}


def compute_noddi(
    dwi_path: str,
    bval_path: str,
    bvec_path: str,
    mask_path: str | None,
    affine: np.ndarray,
    out_dir: str,
    prefix: str,
    threads: int | None = None,
    ndirs: int = 2000,
    bstep: float = 50.0,
    b0_thr: float = 0.0,
) -> list:
    """Fit the NODDI model with AMICO and write geometry-corrected maps.

    Uses **all** shells (NODDI is a multi-shell model).  The gradient table is
    re-written to AMICO's strict 3xN scheme format (DIPY normalises whatever
    orientation the input bvec uses, so an Nx3 file is transposed here).  AMICO
    writes its maps with a header that loses/garbles the true geometry, so each
    map is re-wrapped with the reference *affine* (AMICO keeps the original
    voxel grid, so this is an exact correction).  Returns the written paths.
    """
    import amico
    from dipy.io.gradients import read_bvals_bvecs

    work = tempfile.mkdtemp(prefix="amico_")
    amico_out = os.path.join(work, "amico_out")
    try:
        # --- gradient table -> AMICO scheme (handles Nx3 vs 3xN) -----------
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        bt = os.path.join(work, "grad.bval")
        vt = os.path.join(work, "grad.bvec")
        np.savetxt(bt, bvals.reshape(1, -1), fmt="%g")       # 1 x N
        np.savetxt(vt, bvecs.T, fmt="%.8f")                  # 3 x N
        scheme = os.path.join(work, "grad.scheme")
        amico.util.fsl2scheme(bt, vt, scheme, bStep=bstep)
        log.debug("NODDI: scheme written (bStep=%g) from %d volumes", bstep, len(bvals))

        # Rotation LUT for the requested number of kernels (cached in ~/.dipy).
        amico.lut.precompute_rotation_matrices(12, ndirs)

        ae = amico.Evaluation(study_path=work, subject=".", output_path=amico_out)
        ae.set_config("nthreads", threads if threads and threads > 0 else -1)
        log.debug("NODDI: AMICO nthreads=%s, ndirs=%d", ae.get_config("nthreads"), ndirs)
        ae.load_data(
            dwi_filename=os.path.abspath(dwi_path),
            scheme_filename=scheme,
            mask_filename=os.path.abspath(mask_path) if mask_path else None,
            b0_thr=b0_thr,
        )
        ae.set_model("NODDI")
        ae.generate_kernels(regenerate=True, ndirs=ndirs)
        ae.load_kernels()
        ae.fit()
        ae.save_results()

        # --- copy maps out with the corrected affine -----------------------
        written = []
        for amico_name, (suffix, _descr) in _NODDI_MAPS.items():
            src = os.path.join(amico_out, f"fit_{amico_name}.nii.gz")
            if not os.path.isfile(src):
                log.warning("NODDI: expected map missing: %s", src)
                continue
            vol = np.asanyarray(nib.load(src).dataobj).astype(np.float32)
            dst = os.path.join(out_dir, f"{prefix}_NODDI_{suffix}.nii.gz")
            nib.save(nib.Nifti1Image(vol, affine), dst)
            written.append(dst)
            log.info("      wrote %s", dst)
            log.debug("      NODDI %s stats: %s", suffix, _img_stats(vol[vol != 0]))

        # Fibre orientation map (4D vector), same geometry correction.
        src_dir = os.path.join(amico_out, "fit_dir.nii.gz")
        if os.path.isfile(src_dir):
            vol = np.asanyarray(nib.load(src_dir).dataobj).astype(np.float32)
            dst = os.path.join(out_dir, f"{prefix}_NODDI_dir.nii.gz")
            nib.save(nib.Nifti1Image(vol, affine), dst)
            written.append(dst)
            log.info("      wrote %s", dst)
        return written
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Free-water-eliminated DTI (fwDTI)
# ---------------------------------------------------------------------------
def compute_fwdti(dwi_path, bval_path, bvec_path, mask, affine, out_dir, prefix,
                  fit_method="NLS", b0_threshold=50.0, threads=None):
    """Free-water-eliminated DTI (Hoy et al. 2014) via DIPY, using ALL shells.

    The bi-tensor model separates a tissue tensor from an isotropic free-water
    compartment, so it needs multi-shell data (>=2 non-zero b-values) to be
    well-posed.  Writes free-water-corrected FA/MD/RD/AD, the free-water volume
    fraction ``f``, and the corrected tensor (NRRD).  Re-reads the DWI so all
    shells are used regardless of the DTI ``--max-bval`` filter.
    """
    from dipy.reconst.fwdti import FreeWaterTensorModel

    data = np.asarray(nib.load(dwi_path).dataobj, dtype=np.float32)
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold)
    log.debug("fwDTI: %d volumes, fit_method=%s", len(bvals), fit_method)

    model = FreeWaterTensorModel(gtab, fit_method=fit_method)
    with thread_limits(threads):
        fit = model.fit(data, mask=mask)

    maps = {"FA": fit.fa, "MD": fit.md, "RD": fit.rd, "AD": fit.ad, "f": fit.f}
    written = []
    for key, vol in maps.items():
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if mask is not None:
            vol = vol * mask
        dst = os.path.join(out_dir, f"{prefix}_FW{key}.nii.gz")
        nib.save(nib.Nifti1Image(vol, affine), dst)
        written.append(dst)
        log.info("      wrote %s", dst)
        log.debug("      FW%s stats: %s", key, _img_stats(vol[vol != 0] if mask is not None else vol))

    quad = fit.quadratic_form.astype(np.float32)  # (X, Y, Z, 3, 3) tissue tensor
    if mask is not None:
        quad = quad * mask[..., None, None]
    dst = os.path.join(out_dir, f"{prefix}_FWtensor.nrrd")
    write_tensor_nrrd(dst, quad, affine)
    written.append(dst)
    log.info("      wrote %s", dst)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dwi", required=True, help="DWI volume (*.nii.gz / *.nii)")
    p.add_argument("--bval", help="FSL b-values (default: inferred from --dwi)")
    p.add_argument("--bvec", help="FSL b-vectors (default: inferred from --dwi)")
    p.add_argument("--out", default="results", help="Output directory")
    p.add_argument("--prefix", help="Output filename prefix (default: from DWI name)")
    p.add_argument(
        "--fit-method",
        default="WLS",
        choices=sorted(dti.common_fit_methods.keys()),
        help="DIPY tensor fit method (default: WLS). e.g. WLS (weighted LS), "
        "OLS/LS (ordinary LS), NLLS (non-linear), RESTORE (robust to outliers)",
    )
    p.add_argument("--b0-threshold", type=float, default=50.0, help="b-value at/below which a volume is a b0")
    p.add_argument(
        "--max-bval",
        type=float,
        default=1600.0,
        help="Drop DWI volumes whose b-value exceeds this before fitting (default: 1600). "
        "Use a large value (e.g. 1e9) to keep all shells",
    )
    p.add_argument("--mask", help="Existing brain mask (*.nii.gz / *.nii); non-zero voxels are kept")
    p.add_argument(
        "--mask-method",
        default="otsu",
        choices=["otsu", "synthseg", "synthstrip"],
        help="How to compute the brain mask when --mask is not given (default: otsu). "
        "'synthseg' uses DIPY's deep-learning segmentation (needs PyTorch + weight download); "
        "'synthstrip' calls FreeSurfer's mri_synthstrip as an external process",
    )
    p.add_argument(
        "--mask-input",
        default="b0",
        choices=["b0", "ad"],
        help="Image fed to the masking method (default: b0 = mean b=0). "
        "'ad' first fits an unmasked tensor and uses its axial-diffusivity map, "
        "which tends to give SynthStrip a cleaner brain outline",
    )
    p.add_argument(
        "--bias-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="N4 (ITK) bias-field correction of the masking-input image (default: on)",
    )
    p.add_argument(
        "--synthstrip-exec",
        default="mri_synthstrip",
        help="Path to the mri_synthstrip executable (default: found on PATH)",
    )
    p.add_argument(
        "-j",
        "--threads",
        type=int,
        default=None,
        help="CPU threads for the DIPY tensor fit (BLAS/OpenMP) and for SynthStrip (-t). "
        "Default: library defaults (typically all available cores)",
    )
    p.add_argument("--cuda", action="store_true", help="Use GPU for SynthSeg if available")
    p.add_argument("--no-mask", action="store_true", help="Fit every voxel instead of masking the brain")
    p.add_argument(
        "--noddi",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute NODDI maps with AMICO (default: auto -- on when >=2 shells are present, "
        "using ALL shells regardless of --max-bval). Use --no-noddi to disable",
    )
    p.add_argument("--noddi-ndirs", type=int, default=2000, help="AMICO rotation kernels for NODDI (default: 2000)")
    p.add_argument("--noddi-bstep", type=float, default=50.0, help="AMICO scheme b-value rounding step (default: 50)")
    p.add_argument(
        "--fwdti",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute free-water-corrected DTI (fwDTI) with DIPY (default: auto -- on when >=2 shells are "
        "present, using ALL shells regardless of --max-bval). Use --no-fwdti to disable",
    )
    p.add_argument("--fwdti-fit-method", default="NLS", choices=["NLS", "WLS"],
                   help="fwDTI fit method (default: NLS; WLS is faster)")
    p.add_argument("-v", "--verbose", action="store_true", help="Emit detailed debug output throughout")
    args = p.parse_args(argv)

    setup_logging(args.verbose)
    t_start = time.perf_counter()
    log.debug("parsed arguments: %s", vars(args))

    if args.mask and args.no_mask:
        p.error("--mask and --no-mask are mutually exclusive")

    bval = args.bval or infer_grad_path(args.dwi, ".bval")
    bvec = args.bvec or infer_grad_path(args.dwi, ".bvec")
    log.debug("resolved bval=%s bvec=%s", bval, bvec)
    inputs = [args.dwi, bval, bvec] + ([args.mask] if args.mask else [])
    for f in inputs:
        if not os.path.isfile(f):
            p.error(f"file not found: {f}")

    os.makedirs(args.out, exist_ok=True)
    prefix = args.prefix
    if prefix is None:
        name = os.path.basename(args.dwi)
        for suffix in (".nii.gz", ".nii"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        prefix = name
    log.debug("output dir=%s, prefix=%s", args.out, prefix)

    # --- load -------------------------------------------------------------
    log.info("[1/5] Loading DWI: %s", args.dwi)
    img = nib.load(args.dwi)
    affine = img.affine
    data = np.asarray(img.dataobj, dtype=np.float32)  # (X, Y, Z, N)
    log.info("      data shape %s, voxel %s", data.shape, tuple(img.header.get_zooms()[:3]))
    log.debug("      stored dtype=%s, in-memory %.2f GB (float32)", img.get_data_dtype(), data.nbytes / 1e9)
    log.debug("      affine:\n%s", np.array2string(affine, precision=4))

    bvals, bvecs = read_bvals_bvecs(bval, bvec)

    # Shells across the FULL acquisition (before DTI b-value filtering) -- used
    # to decide whether NODDI (a multi-shell model) is applicable.
    full_shells = np.unique(np.round(bvals[bvals > args.b0_threshold] / 100.0) * 100).astype(int)
    n_shells = len(full_shells)
    log.debug("full acquisition shells ~%s s/mm^2 (%d shells)", full_shells.tolist(), n_shells)

    # Drop high-b volumes (e.g. keep only b0 + low shell for a DTI-appropriate fit).
    keep = bvals <= args.max_bval
    n_drop = int((~keep).sum())
    if n_drop:
        dropped_shells = np.unique(np.round(bvals[~keep] / 100.0) * 100).astype(int)
        log.info(
            "      dropping %d/%d volumes with b>%g s/mm^2 (shells ~%s)",
            n_drop, len(bvals), args.max_bval, dropped_shells.tolist(),
        )
        data = data[..., keep]
        bvals = bvals[keep]
        bvecs = bvecs[keep]
    else:
        log.debug("      no volumes exceed b>%g; keeping all %d", args.max_bval, len(bvals))

    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=args.b0_threshold)
    n_b0 = int(gtab.b0s_mask.sum())
    shells = np.unique(np.round(bvals[~gtab.b0s_mask] / 100.0) * 100).astype(int)
    log.info("      %d volumes, %d b0, shells ~%s s/mm^2", len(bvals), n_b0, shells.tolist())
    log.debug("      b0 volume indices: %s", np.where(gtab.b0s_mask)[0].tolist())
    log.debug("      bval range [%.1f, %.1f]; bvec norm range [%.4f, %.4f]",
              bvals.min(), bvals.max(),
              np.linalg.norm(bvecs, axis=1).min(), np.linalg.norm(bvecs, axis=1).max())

    model = dti.TensorModel(gtab, fit_method=args.fit_method)

    # --- brain mask -------------------------------------------------------
    prefit = None  # unmasked fit, reused later when --mask-input ad is chosen
    mask_file = None  # on-disk mask path, passed to AMICO for NODDI
    if args.no_mask:
        mask = None
        log.info("[2/5] No mask (fitting all voxels)")
    elif args.mask:
        mask_file = args.mask
        log.info("[2/5] Loading brain mask: %s", args.mask)
        mimg = nib.load(args.mask)
        if mimg.shape[:3] != data.shape[:3]:
            p.error(f"mask shape {mimg.shape[:3]} does not match DWI {data.shape[:3]}")
        if not np.allclose(mimg.affine, affine, atol=1e-3):
            log.warning("      mask affine differs from DWI affine; using DWI geometry")
        mask = np.asarray(mimg.dataobj) > 0
        log.info("      %d voxels in mask", int(mask.sum()))
    else:
        bc = ", N4 bias-corrected" if args.bias_correct else ""
        log.info("[2/5] Computing brain mask (method=%s, input=%s%s)", args.mask_method, args.mask_input, bc)

        # Build the image that is fed to the masking method.
        if args.mask_input == "ad":
            log.info("      fitting unmasked tensor to derive the AD input image")
            t0 = time.perf_counter()
            with thread_limits(args.threads):
                prefit = model.fit(data, mask=None)
            mask_input_img = np.nan_to_num(prefit.ad, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            log.debug("      unmasked fit done in %.1fs; AD %s", time.perf_counter() - t0, _img_stats(mask_input_img))
        else:
            mask_input_img = mean_b0(data, gtab.b0s_mask)
            log.debug("      mean-b0 input %s", _img_stats(mask_input_img))

        if args.bias_correct:
            log.info("      applying N4 bias-field correction (ITK)")
            voxel_sizes = [float(np.linalg.norm(affine[:3, i])) for i in range(3)]
            mask_input_img = n4_bias_correct(mask_input_img, voxel_sizes)

        in_path = os.path.join(args.out, f"{prefix}_maskinput.nii.gz")
        nib.save(nib.Nifti1Image(mask_input_img, affine), in_path)
        log.info("      wrote mask input image %s", in_path)

        mask = compute_brain_mask(
            mask_input_img, affine, args.mask_method, args.synthstrip_exec, args.cuda,
            threads=args.threads,
        )
        mask_path = os.path.join(args.out, f"{prefix}_brainmask.nii.gz")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), mask_path)
        mask_file = mask_path
        log.info("      %d voxels in mask; wrote %s", int(mask.sum()), mask_path)

    if mask is not None:
        frac = 100.0 * float(mask.mean())
        idx = np.where(mask)
        bbox = tuple((int(a.min()), int(a.max())) for a in idx) if idx[0].size else ()
        log.debug("mask: %.1f%% of volume, bounding box (x,y,z)=%s", frac, bbox)

    # --- tensor fit -------------------------------------------------------
    log.info("[3/5] Fitting diffusion tensor (fit_method=%s)", args.fit_method)
    if prefit is not None:
        log.info("      reusing the unmasked fit computed for the AD mask input")
        fit = prefit
    else:
        t0 = time.perf_counter()
        with thread_limits(args.threads):
            fit = model.fit(data, mask=mask)
        log.debug("      fit done in %.1fs", time.perf_counter() - t0)

    # --- scalar maps ------------------------------------------------------
    log.info("[4/5] Computing scalar maps (FA, MD, RD, AD)")
    maps = {
        "FA": fit.fa,
        "MD": fit.md,
        "RD": fit.rd,
        "AD": fit.ad,
    }
    for key, vol in maps.items():
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if mask is not None:
            vol = vol * mask
        out_nii = os.path.join(args.out, f"{prefix}_{key}.nii.gz")
        nib.save(nib.Nifti1Image(vol, affine), out_nii)
        log.info("      wrote %s", out_nii)
        log.debug("      %s stats: %s", key, _img_stats(vol[vol != 0] if mask is not None else vol))

    # --- tensor to NRRD ---------------------------------------------------
    log.info("[5/5] Writing tensor to NRRD")
    quad = fit.quadratic_form.astype(np.float32)  # (X, Y, Z, 3, 3)
    if mask is not None:
        quad = quad * mask[..., None, None]
    out_nrrd = os.path.join(args.out, f"{prefix}_tensor.nrrd")
    write_tensor_nrrd(out_nrrd, quad, affine)
    log.info("      wrote %s", out_nrrd)

    # --- NODDI (AMICO) ----------------------------------------------------
    if args.noddi is False:
        do_noddi = False
    elif args.noddi is True:
        do_noddi = True
        if n_shells < 2:
            log.warning("NODDI requested but only %d shell(s) present; results may be unreliable", n_shells)
    else:  # auto
        do_noddi = n_shells >= 2

    if do_noddi:
        log.info(
            "[NODDI] Fitting NODDI with AMICO (all %d shells ~%s, ndirs=%d, bStep=%g)",
            n_shells, full_shells.tolist(), args.noddi_ndirs, args.noddi_bstep,
        )
        if mask_file is None:
            log.warning("      no brain mask available; NODDI will fit the whole volume (slow)")
        try:
            t0 = time.perf_counter()
            compute_noddi(
                args.dwi, bval, bvec, mask_file, affine, args.out, prefix,
                threads=args.threads, ndirs=args.noddi_ndirs, bstep=args.noddi_bstep,
                b0_thr=args.b0_threshold,
            )
            log.info("      NODDI done (%.1fs)", time.perf_counter() - t0)
        except Exception as exc:
            log.error("      NODDI failed (%s): %s", type(exc).__name__, exc)
    else:
        log.info("[NODDI] Skipped (%d shell(s) present; needs >=2 or --noddi to force)", n_shells)

    # --- free-water DTI (DIPY) --------------------------------------------
    if args.fwdti is False:
        do_fwdti = False
    elif args.fwdti is True:
        do_fwdti = True
        if n_shells < 2:
            log.warning("fwDTI requested but only %d shell(s) present; the bi-tensor model is "
                        "ill-posed on single-shell data -- results may be unreliable", n_shells)
    else:  # auto
        do_fwdti = n_shells >= 2

    if do_fwdti:
        log.info("[fwDTI] Fitting free-water-corrected DTI (all %d shells ~%s, fit_method=%s)",
                 n_shells, full_shells.tolist(), args.fwdti_fit_method)
        if mask is None:
            log.warning("      no brain mask; fwDTI will fit the whole volume (slow)")
        try:
            t0 = time.perf_counter()
            compute_fwdti(args.dwi, bval, bvec, mask, affine, args.out, prefix,
                          fit_method=args.fwdti_fit_method, b0_threshold=args.b0_threshold,
                          threads=args.threads)
            log.info("      fwDTI done (%.1fs)", time.perf_counter() - t0)
        except Exception as exc:
            log.error("      fwDTI failed (%s): %s", type(exc).__name__, exc)
    else:
        log.info("[fwDTI] Skipped (%d shell(s) present; needs >=2 or --fwdti to force)", n_shells)

    log.info("Done. (%.1fs total)", time.perf_counter() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
