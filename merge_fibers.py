#!/usr/bin/env python3
"""Merge a folder of VTK fiber polydata files into a single polydata.

Each point in the merged output carries an integer `FiberLabel` array
identifying which source file it came from. A companion CSV records the
label-to-filename mapping.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import get_numpy_array_type, numpy_to_vtk


def read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise SystemExit(f"Failed to read polydata or no points in: {path}")
    return polydata


def add_label_array(
    polydata: vtk.vtkPolyData, label: int, array_name: str
) -> None:
    n = polydata.GetNumberOfPoints()
    labels = np.full(n, label, dtype=np.int32)
    arr = numpy_to_vtk(labels, deep=True, array_type=vtk.VTK_INT)
    arr.SetName(array_name)
    polydata.GetPointData().AddArray(arr)


def collect_array_specs(
    polydatas: list[vtk.vtkPolyData],
) -> dict[str, tuple[int, int]]:
    """Scan all polydatas and return {array_name: (vtk_type, n_components)}.
    Raises if two inputs declare the same name with conflicting type/components."""
    specs: dict[str, tuple[int, int]] = {}
    for pd in polydatas:
        pdata = pd.GetPointData()
        for i in range(pdata.GetNumberOfArrays()):
            arr = pdata.GetArray(i)
            if arr is None:
                continue
            name = arr.GetName()
            if not name:
                continue
            spec = (arr.GetDataType(), arr.GetNumberOfComponents())
            existing = specs.get(name)
            if existing is not None and existing != spec:
                raise SystemExit(
                    f"Array {name!r} has conflicting specs across inputs: "
                    f"existing (vtk_type={existing[0]}, comps={existing[1]}) vs "
                    f"new (vtk_type={spec[0]}, comps={spec[1]})"
                )
            specs.setdefault(name, spec)
    return specs


def pad_missing_arrays(
    polydata: vtk.vtkPolyData,
    specs: dict[str, tuple[int, int]],
    fill_value: float,
) -> int:
    """Add any arrays from `specs` that are absent on `polydata`, filled with
    `fill_value`. Returns the number of arrays added."""
    pdata = polydata.GetPointData()
    present = {
        pdata.GetArrayName(i) for i in range(pdata.GetNumberOfArrays())
    }
    n_points = polydata.GetNumberOfPoints()
    added = 0
    for name, (vtk_type, n_components) in specs.items():
        if name in present:
            continue
        np_dtype = get_numpy_array_type(vtk_type)
        shape = (n_points, n_components) if n_components > 1 else (n_points,)
        filler = np.full(shape, fill_value, dtype=np_dtype)
        vtk_arr = numpy_to_vtk(filler, deep=True, array_type=vtk_type)
        vtk_arr.SetName(name)
        pdata.AddArray(vtk_arr)
        added += 1
    return added


def write_polydata(
    polydata: vtk.vtkPolyData, path: Path, binary: bool, legacy: bool
) -> None:
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    if binary:
        writer.SetFileTypeToBinary()
    else:
        writer.SetFileTypeToASCII()
    if legacy:
        writer.SetFileVersion(vtk.vtkPolyDataWriter.VTK_LEGACY_READER_VERSION_4_2)
    writer.Write()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all VTK polydata files in a folder into a single polydata "
            "with a FiberLabel point-data array. Writes a CSV mapping each "
            "label to its source filename."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing VTK polydata files to merge.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output merged VTK file. Defaults to <input-dir>_merged.vtk next "
            "to the input folder."
        ),
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=Path,
        default=None,
        help=(
            "Output CSV file mapping labels to filenames. Defaults to the "
            "merged VTK path with a .csv extension."
        ),
    )
    parser.add_argument(
        "--label-name",
        default="FiberLabel",
        help="Name of the per-point label array (default: FiberLabel).",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=-1.0,
        help=(
            "Sentinel value used to fill point-data arrays that are present on "
            "some inputs but missing on others (default: -1). Cast to each "
            "array's native VTK dtype, so integer arrays receive int(-1)."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.vtk",
        help="Glob pattern for input files within the folder (default: *.vtk).",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write merged VTK in binary format (default: ASCII).",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help=(
            "Write merged VTK using legacy file format version 4.2 "
            "instead of the default 5.1."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {args.input_dir}")

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(
            f"No files matching {args.pattern!r} in {args.input_dir}"
        )

    output_vtk = args.output or args.input_dir.with_name(
        f"{args.input_dir.name}_merged.vtk"
    )
    output_csv = args.csv or output_vtk.with_suffix(".csv")

    polydatas: list[vtk.vtkPolyData] = []
    mapping: list[tuple[int, str]] = []
    for label, path in enumerate(files):
        polydata = read_polydata(path)
        add_label_array(polydata, label, args.label_name)
        polydatas.append(polydata)
        mapping.append((label, path.name))

    specs = collect_array_specs(polydatas)
    total_padded = 0
    for polydata in polydatas:
        total_padded += pad_missing_arrays(polydata, specs, args.fill_value)

    appender = vtk.vtkAppendPolyData()
    for polydata in polydatas:
        appender.AddInputData(polydata)
    appender.Update()
    merged = appender.GetOutput()

    write_polydata(merged, output_vtk, binary=args.binary, legacy=args.legacy)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "filename"])
        writer.writerows(mapping)

    print(
        f"Merged {len(files)} files "
        f"({merged.GetNumberOfPoints()} points, "
        f"{merged.GetNumberOfCells()} cells, "
        f"{len(specs)} unique point-data arrays, "
        f"{total_padded} filled with {args.fill_value}) "
        f"-> {output_vtk}"
    )
    print(f"Label mapping -> {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
