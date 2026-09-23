"""Build a compact DSSAT teacher-trajectory cache for the differentiable model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _number(value: str) -> float:
    value = value.strip()
    if value in {"", "-99", "-99.0", "-99.00"}:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _parse_summary(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="replace").splitlines()
    header = next(line[1:].split() for line in lines if line.startswith("@"))
    rows = [line.split() for line in lines if line.strip()[:1].isdigit()]
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"Summary columns do not align: {path}")
    frame = pd.DataFrame(rows, columns=header)
    for column in frame.columns:
        frame[column] = frame[column].map(_number)
    frame["TRNO"] = frame["TRNO"].astype(int)
    return frame.sort_values("TRNO").reset_index(drop=True)


def _parse_daily(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="replace").splitlines()
    records: list[dict[str, float]] = []
    run = None
    header: list[str] | None = None
    for line in lines:
        if line.startswith("*RUN"):
            match = re.search(r"\s(\d+)\s*$", line.strip())
            run = int(match.group(1)) if match else (run or 0) + 1
            header = None
            continue
        if run is not None and line.startswith("@"):
            header = line[1:].split()
            continue
        if run is None or header is None or not line.strip() or not line.lstrip()[0].isdigit():
            continue
        values = line.split()
        if len(values) < len(header):
            continue
        record = {"TRNO": run}
        record.update({name: _number(value) for name, value in zip(header, values)})
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No daily rows found: {path}")
    frame["TRNO"] = frame["TRNO"].astype(int)
    frame["YEAR"] = frame["YEAR"].astype(int)
    frame["DOY"] = frame["DOY"].astype(int)
    return frame


def build_reference(input_dir: Path, output_path: Path) -> None:
    summary = _parse_summary(input_dir / "Summary.OUT")
    plant = _parse_daily(input_dir / "PlantGro.OUT")
    water = _parse_daily(input_dir / "SoilWat.OUT")
    evap = _parse_daily(input_dir / "ET.OUT")

    daily = water.merge(
        evap[["TRNO", "YEAR", "DOY", "EOAA", "EOPA", "EOSA", "ETAA", "EPAA", "ESAA"]],
        on=["TRNO", "YEAR", "DOY"],
        how="outer",
    ).merge(
        plant[["TRNO", "YEAR", "DOY", "GSTD", "LAID", "CWAD", "HIAD", "WSPD", "WSGD"]],
        on=["TRNO", "YEAR", "DOY"],
        how="outer",
    )
    daily = daily.sort_values(["TRNO", "YEAR", "DOY"])
    treatment_ids = summary["TRNO"].to_numpy(dtype=np.int64)
    dates = daily[["YEAR", "DOY"]].drop_duplicates().sort_values(["YEAR", "DOY"])
    date_index = pd.MultiIndex.from_frame(dates)

    def matrix(column: str) -> np.ndarray:
        pivot = daily.pivot(index="TRNO", columns=["YEAR", "DOY"], values=column)
        pivot = pivot.reindex(index=treatment_ids, columns=date_index)
        return pivot.to_numpy(dtype=np.float32)

    def daily_increment(values: np.ndarray) -> np.ndarray:
        """Convert DSSAT cumulative water outputs to non-negative daily fluxes."""
        result = np.full_like(values, np.nan, dtype=np.float32)
        for row in range(values.shape[0]):
            valid = np.isfinite(values[row])
            if not valid.any():
                continue
            indices = np.flatnonzero(valid)
            series = values[row, indices]
            result[row, indices[0]] = series[0]
            if len(indices) > 1:
                result[row, indices[1:]] = np.maximum(np.diff(series), 0.0)
        return result

    precipitation_cumulative = matrix("PREC")
    irrigation_cumulative = matrix("IRRC")

    variables = {
        "soil_water_mm": matrix("SWTD"),
        "soil_water_extractable_mm": matrix("SWXD"),
        "precipitation_cumulative_mm": precipitation_cumulative,
        "irrigation_cumulative_mm": irrigation_cumulative,
        "precipitation_mm": daily_increment(precipitation_cumulative),
        "irrigation_mm": daily_increment(irrigation_cumulative),
        "et_mm": matrix("ETAA"),
        "reference_et_mm": matrix("EOAA"),
        "potential_transpiration_mm": matrix("EOPA"),
        "potential_soil_evaporation_mm": matrix("EOSA"),
        "transpiration_mm": matrix("EPAA"),
        "soil_evaporation_mm": matrix("ESAA"),
        "stage": matrix("GSTD"),
        "lai": matrix("LAID"),
        "biomass_kg_ha": matrix("CWAD"),
        "hi_daily": matrix("HIAD"),
        "water_stress_photosynthesis": matrix("WSPD"),
        "water_stress_growth": matrix("WSGD"),
    }
    metadata = {
        "source": str(input_dir),
        "records": int(len(treatment_ids)),
        "dates": int(len(date_index)),
        "cultivar": "IB0007 ZM113 XJ/HX / CO0005",
        "weather_note": "Weather dependency and temporary filename aliases are recorded in run_manifest.txt",
        "soil_note": "AA.SOL supplied from the WL-DSSAT reference dependency",
        "sentinel": "DSSAT -99 values are stored as NaN",
        "variables": {
            "soil_water_mm": "SWTD, mm",
            "irrigation_cumulative_mm": "IRRC, cumulative mm",
            "irrigation_mm": "difference of IRRC, mm/day",
            "precipitation_cumulative_mm": "PREC, cumulative mm",
            "precipitation_mm": "difference of PREC, mm/day",
            "et_mm": "ETAA, mm/day",
            "reference_et_mm": "EOAA, mm/day",
            "potential_transpiration_mm": "EOPA, mm/day",
            "potential_soil_evaporation_mm": "EOSA, mm/day",
            "transpiration_mm": "EPAA, mm/day",
            "soil_evaporation_mm": "ESAA, mm/day",
            "lai": "LAID, m2/m2",
            "biomass_kg_ha": "CWAD, kg/ha",
            "water_stress_photosynthesis": "WSPD, 0=no stress and 1=maximum stress",
            "water_stress_growth": "WSGD, 0=no stress and 1=maximum stress",
            "yield_kg_ha": "HWAM, kg/ha",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        trno=treatment_ids,
        year=dates["YEAR"].to_numpy(dtype=np.int32),
        doy=dates["DOY"].to_numpy(dtype=np.int32),
        yield_kg_ha=summary["HWAM"].to_numpy(dtype=np.float32),
        irrigation_total_mm=summary["IRCM"].to_numpy(dtype=np.float32),
        et_total_mm=summary["ETCM"].to_numpy(dtype=np.float32),
        **variables,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps({"output": str(output_path), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_reference(args.input_dir, args.output)
