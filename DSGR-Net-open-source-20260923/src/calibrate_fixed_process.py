"""Calibrate the fixed, differentiable cotton process against a DSSAT teacher cache.

This is deliberately separate from the neural model.  It answers one question first:
can the hard water--stress--biomass equations reproduce the scale and seasonal
trajectory of the ZM113 DSSAT run before a DNN is allowed to learn corrections?
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np


def read_weather(path: Path, year: int, doys: np.ndarray) -> dict:
    rows: dict[int, tuple[float, float, float, float, float, float]] = {}
    header_seen = False
    site_row_pending = False
    latitude = elevation = wind_height = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("@ INSI"):
            site_row_pending = True
            continue
        if site_row_pending and line.strip() and not line.lstrip().startswith("!"):
            site = line.split()
            latitude = float(site[1])
            elevation = float(site[3])
            wind_height = float(site[7])
            site_row_pending = False
            continue
        if line.startswith("@DATE"):
            header = line[1:].split()
            expected = ["DATE", "SRAD", "TMAX", "TMIN", "RAIN", "DEWP", "WIND"]
            if header[: len(expected)] != expected:
                raise ValueError(f"Unexpected weather columns: {header}")
            header_seen = True
            continue
        if not header_seen or not line.strip() or not line.lstrip()[0].isdigit():
            continue
        values = line.split()
        code = int(values[0])
        row_year = 2000 + code // 1000
        doy = code % 1000
        if row_year != year:
            continue
        rows[doy] = (
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
            float(values[5]),
            float(values[6]) / 86.4,
        )
    if not rows:
        raise ValueError(f"No {year} weather rows found in {path}")
    missing = [int(d) for d in doys if int(d) not in rows]
    if missing:
        raise ValueError(f"Weather missing DOY values: {missing[:10]}")
    matrix = np.asarray([rows[int(d)] for d in doys], dtype=np.float32)
    if latitude is None or elevation is None or wind_height is None:
        raise ValueError(f"Weather station metadata missing in {path}")
    return dict(
        srad=matrix[:, 0], tmax=matrix[:, 1], tmin=matrix[:, 2], rain=matrix[:, 3],
        dewp=matrix[:, 4], wind=matrix[:, 5], latitude=latitude, elevation=elevation,
        wind_height=wind_height,
    )


def reference_et0(
    weather: dict, doys: np.ndarray, model: str
) -> np.ndarray:
    """Daily reference ET in mm/day; all operations have direct PyTorch analogues."""
    srad = np.asarray(weather["srad"], dtype=np.float64)
    if model in {"radiation", "dual_source_radiation", "dssat_components_radiation"}:
        return 0.408 * 0.65 * np.maximum(srad, 0.0)
    if model != "fao56_pm":
        raise ValueError(f"Unknown ET model: {model}")

    # FAO-56 Penman--Monteith with dew-point vapour pressure and measured wind.
    # Daily soil heat flux is zero. WIND in the DSSAT file was converted from
    # km/day to m/s by read_weather and is adjusted from WNDHT to 2 m.
    tmax = np.asarray(weather["tmax"], dtype=np.float64)
    tmin = np.asarray(weather["tmin"], dtype=np.float64)
    tmean = (tmax + tmin) / 2.0
    dewp = np.asarray(weather["dewp"], dtype=np.float64)
    wind = np.asarray(weather["wind"], dtype=np.float64)
    latitude = np.deg2rad(float(weather["latitude"]))
    elevation = float(weather["elevation"])
    wind_height = float(weather["wind_height"])

    saturation = lambda temp: 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    es = (saturation(tmax) + saturation(tmin)) / 2.0
    ea = saturation(dewp)
    delta = 4098.0 * saturation(tmean) / (tmean + 237.3) ** 2
    pressure = 101.3 * ((293.0 - 0.0065 * elevation) / 293.0) ** 5.26
    gamma = 0.000665 * pressure
    u2 = wind * 4.87 / np.log(67.8 * wind_height - 5.42)

    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doys / 365.0)
    solar_declination = 0.409 * np.sin(2.0 * np.pi * doys / 365.0 - 1.39)
    sunset_angle = np.arccos(np.clip(-np.tan(latitude) * np.tan(solar_declination), -1.0, 1.0))
    ra = (
        24.0 * 60.0 / np.pi * 0.0820 * dr
        * (sunset_angle * np.sin(latitude) * np.sin(solar_declination)
           + np.cos(latitude) * np.cos(solar_declination) * np.sin(sunset_angle))
    )
    clear_sky = np.maximum((0.75 + 2e-5 * elevation) * ra, 1e-6)
    net_shortwave = 0.77 * np.maximum(srad, 0.0)
    cloud_factor = np.clip(1.35 * srad / clear_sky - 0.35, 0.05, 1.0)
    net_longwave = (
        4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0
        * np.maximum(0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)), 0.05)
        * cloud_factor
    )
    net_radiation = net_shortwave - net_longwave
    numerator = (
        0.408 * delta * net_radiation
        + gamma * 900.0 / (tmean + 273.0) * u2 * np.maximum(es - ea, 0.0)
    )
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    return np.maximum(numerator / np.maximum(denominator, 1e-6), 0.0)


def simulate(params: dict, data: dict[str, np.ndarray], weather: dict) -> dict[str, np.ndarray]:
    # Missing DSSAT management rows mean no event.  They must not poison the
    # recurrent state; missing teacher states remain NaN and are skipped only
    # by the evaluation metric.
    irrigation = np.nan_to_num(data["irrigation_mm"], nan=0.0)
    rain = weather["rain"][None, :]
    srad = weather["srad"][None, :]
    tmean = ((weather["tmax"] + weather["tmin"]) / 2.0)[None, :]
    n, t = irrigation.shape
    water = data["soil_water_extractable_mm"][:, 0].astype(np.float64).copy()
    thermal = np.zeros(n, dtype=np.float64)
    biomass = np.zeros(n, dtype=np.float64)
    surface_wetness = np.ones(n, dtype=np.float64)
    water_out = np.full((n, t), np.nan, dtype=np.float64)
    et_out = np.full((n, t), np.nan, dtype=np.float64)
    lai_out = np.full((n, t), np.nan, dtype=np.float64)
    biomass_out = np.full((n, t), np.nan, dtype=np.float64)
    soil_evaporation_out = np.full((n, t), np.nan, dtype=np.float64)
    transpiration_out = np.full((n, t), np.nan, dtype=np.float64)
    surface_wetness_out = np.full((n, t), np.nan, dtype=np.float64)
    doys = data["doy"]
    et0_daily = reference_et0(weather, doys, str(params.get("et_model", "radiation")))
    sowing_doy = int(data["sowing_doy"])
    first_row_is_initial_state = not np.any(np.isfinite(data["et_mm"][:, 0]))

    for day in range(t):
        if day == 0 and first_row_is_initial_state:
            water_out[:, day] = water
            lai_out[:, day] = 0.0
            biomass_out[:, day] = biomass
            surface_wetness_out[:, day] = surface_wetness
            continue
        planted = float(doys[day] >= sowing_doy)
        thermal += np.maximum(float(tmean[0, day]) - 10.0, 0.0) * planted
        stage = np.clip(thermal / params["ttmat"], 0.0, 1.0)
        gate = planted / (1.0 + np.exp((thermal - params["ttmat"]) / params["gate_width"]))
        canopy_shape = np.maximum(np.sin(np.pi * stage), 0.0)
        if params.get("lai_model") == "double_sigmoid":
            lai = (
                params["lai_asymptote"]
                / (1.0 + np.exp(-(thermal - params["lai_growth_midpoint_gdd"]) / params["lai_growth_width_gdd"]))
                / (1.0 + np.exp((thermal - params["lai_decline_midpoint_gdd"]) / params["lai_decline_width_gdd"]))
                * planted
            )
        else:
            lai = params["max_lai"] * canopy_shape ** 1.2 * planted
        water_before_fluxes = (
            water
            + params["rain_efficiency"] * rain[0, day]
            + params["irrigation_efficiency"] * irrigation[:, day]
        )
        relative_water = np.clip(water_before_fluxes / params["field_capacity"], 0.0, 1.5)
        stress = 1.0 / (1.0 + np.exp(-params["stress_slope"] * (relative_water - params["stress_threshold"])))
        rising_temperature = np.clip((float(tmean[0, day]) - 10.0) / 18.0, 0.0, 1.0)
        falling_temperature = np.clip((40.0 - float(tmean[0, day])) / 12.0, 0.0, 1.0)
        temp_factor = min(rising_temperature, falling_temperature)
        et0 = float(et0_daily[day])
        if params.get("legacy_et_gate", False):
            kc = 0.4 + 0.8 * canopy_shape
            et_floor = params.get("et_stress_floor", 0.3)
            demand = et0 * kc * params["et_scale"] * (et_floor + (1.0 - et_floor) * stress) * gate
            soil_evaporation = np.zeros_like(demand)
            transpiration = demand
        elif params.get("et_model") in {"dual_source_radiation", "dssat_components_radiation"}:
            # A bounded wetness memory represents short-lived evaporation after
            # rain or irrigation.  It is an auxiliary response state; total ET
            # is still removed once from the root-zone water balance below.
            effective_input = (
                params["rain_efficiency"] * rain[0, day]
                + params["irrigation_efficiency"] * irrigation[:, day]
            )
            event_wetness = 1.0 - np.exp(
                -np.maximum(effective_input, 0.0) / params["surface_wetting_scale_mm"]
            )
            surface_wetness = 1.0 - (
                1.0 - params["surface_memory"] * surface_wetness
            ) * (1.0 - event_wetness)
            if params.get("et_model") == "dssat_components_radiation":
                potential_soil_evaporation = (
                    et0 * params["potential_soil_scale"]
                    * np.exp(-params["soil_lai_extinction"] * lai)
                )
                potential_transpiration = (
                    et0 * params["potential_transpiration_scale"]
                    * (1.0 - np.exp(-params["transpiration_lai_extinction"] * lai))
                )
                transpiration_stress = 1.0 / (1.0 + np.exp(
                    -params["transpiration_stress_slope"]
                    * (relative_water - params["transpiration_stress_threshold"])
                ))
                soil_evaporation = potential_soil_evaporation * surface_wetness
                transpiration = potential_transpiration * transpiration_stress * gate
            else:
                canopy_cover = 1.0 - np.exp(-params["k"] * lai)
                soil_evaporation = (
                    et0 * params["soil_evaporation_coefficient"]
                    * (1.0 - canopy_cover) * surface_wetness
                )
                transpiration = (
                    et0 * params["basal_crop_coefficient"]
                    * canopy_cover * stress * gate
                )
            demand = soil_evaporation + transpiration
        else:
            # ETAA contains soil evaporation outside the active crop period.
            kc = 0.3 + 0.9 * canopy_shape * gate
            et_floor = params.get("et_stress_floor", 0.3)
            demand = et0 * kc * params["et_scale"] * (et_floor + (1.0 - et_floor) * stress)
            soil_evaporation = np.zeros_like(demand)
            transpiration = demand
        actual_et = np.minimum(np.maximum(demand, 0.0), np.maximum(water_before_fluxes, 0.0))
        flux_fraction = actual_et / np.maximum(demand, 1e-8)
        soil_evaporation = soil_evaporation * flux_fraction
        transpiration = transpiration * flux_fraction
        if params.get("et_model") in {"dual_source_radiation", "dssat_components_radiation"}:
            surface_wetness *= np.exp(
                -soil_evaporation / params["surface_wetting_scale_mm"]
            )
        water_after_et = water_before_fluxes - actual_et
        # SWXD can temporarily exceed field-capacity extractable water after an
        # irrigation pulse.  Drain only a calibrated fraction of that excess
        # each day instead of deleting all excess water immediately.
        excess = np.logaddexp(
            0.0, params["drainage_smoothing"] * (water_after_et - params["field_capacity"])
        ) / params["drainage_smoothing"]
        drainage = params["drainage_fraction"] * excess
        water = water_after_et - drainage
        apar = 0.48 * max(float(weather["srad"][day]), 0.0) * (1.0 - np.exp(-params["k"] * lai))
        growth = params["rue"] * apar * stress * temp_factor * 10.0 * gate
        biomass += np.maximum(growth, 0.0)
        water_out[:, day] = water
        et_out[:, day] = actual_et
        lai_out[:, day] = lai
        biomass_out[:, day] = biomass
        soil_evaporation_out[:, day] = soil_evaporation
        transpiration_out[:, day] = transpiration
        surface_wetness_out[:, day] = surface_wetness
    return {
        "soil_water_extractable_mm": water_out,
        "et_mm": et_out,
        "lai": lai_out,
        "biomass_kg_ha": biomass_out,
        "yield_kg_ha": biomass * params["yield_coefficient"],
        "soil_evaporation_mm": soil_evaporation_out,
        "transpiration_mm": transpiration_out,
        "surface_wetness": surface_wetness_out,
    }


def score(params: dict[str, float], data: dict[str, np.ndarray], weather: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    pred = simulate(params, data, weather)
    components = {
        "soil_water_rmse_mm": float(np.sqrt(np.nanmean((pred["soil_water_extractable_mm"] - data["soil_water_extractable_mm"]) ** 2))),
        "et_rmse_mm_day": float(np.sqrt(np.nanmean((pred["et_mm"] - data["et_mm"]) ** 2))),
        "lai_rmse": float(np.sqrt(np.nanmean((pred["lai"] - data["lai"]) ** 2))),
        "biomass_rmse_kg_ha": float(np.sqrt(np.nanmean((pred["biomass_kg_ha"] - data["biomass_kg_ha"]) ** 2))),
        "yield_rmse_kg_ha": float(np.sqrt(np.nanmean((pred["yield_kg_ha"] - data["yield_kg_ha"][:, None]) ** 2))),
    }
    total = components["soil_water_rmse_mm"] / 50.0 + components["et_rmse_mm_day"] / 3.0 + components["lai_rmse"] + components["biomass_rmse_kg_ha"] / 3000.0 + components["yield_rmse_kg_ha"] / 1000.0
    return float(total), components


def water_score(
    params: dict[str, float], data: dict[str, np.ndarray], weather: dict[str, np.ndarray]
) -> tuple[float, dict[str, float]]:
    """Fit only water state and ET; crop/yield targets must not tune water parameters."""
    pred = simulate(params, data, weather)
    components = {
        "soil_water_rmse_mm": float(np.sqrt(np.nanmean(
            (pred["soil_water_extractable_mm"] - data["soil_water_extractable_mm"]) ** 2
        ))),
        "et_rmse_mm_day": float(np.sqrt(np.nanmean((pred["et_mm"] - data["et_mm"]) ** 2))),
    }
    if "soil_evaporation_mm" in data and "transpiration_mm" in data:
        components["soil_evaporation_rmse_mm_day"] = float(np.sqrt(np.nanmean(
            (pred["soil_evaporation_mm"] - data["soil_evaporation_mm"]) ** 2
        )))
        components["transpiration_rmse_mm_day"] = float(np.sqrt(np.nanmean(
            (pred["transpiration_mm"] - data["transpiration_mm"]) ** 2
        )))
    if params.get("et_model") in {"dual_source_radiation", "dssat_components_radiation"} and "soil_evaporation_mm" in data:
        # Direct DSSAT component supervision removes the ambiguity in fitting
        # two fluxes only through their ETAA sum.
        return (
            components["soil_water_rmse_mm"] / 50.0
            + components["soil_evaporation_rmse_mm_day"] / 1.5
            + components["transpiration_rmse_mm_day"] / 1.5
        ), components
    # Normalize by explicit tolerances so one variable cannot dominate merely
    # because it is measured on a larger numerical scale.
    return components["soil_water_rmse_mm"] / 50.0 + components["et_rmse_mm_day"] / 1.5, components


def load_reference(cache: Path, weather_path: Path) -> tuple[dict, dict, dict]:
    raw = np.load(cache, allow_pickle=True)
    metadata = json.loads(str(raw["metadata"]))
    data = {
        name: raw[name].astype(np.float64)
        for name in [
            "soil_water_extractable_mm", "et_mm", "lai", "biomass_kg_ha",
            "irrigation_mm", "yield_kg_ha",
        ]
    }
    for name in (
        "reference_et_mm", "potential_transpiration_mm", "potential_soil_evaporation_mm",
        "transpiration_mm", "soil_evaporation_mm", "water_stress_photosynthesis",
        "water_stress_growth",
    ):
        if name in raw.files:
            data[name] = raw[name].astype(np.float64)
    data["doy"] = raw["doy"].astype(np.int32)
    year = int(raw["year"][0])
    if np.any(raw["year"] != year):
        raise ValueError(f"Teacher cache mixes years: {cache}")
    data["sowing_doy"] = date(year, 4, 10).timetuple().tm_yday
    return data, read_weather(weather_path, year, raw["doy"]), metadata


def fit_growth_stress(data: dict[str, np.ndarray], field_capacity: float) -> tuple[dict, dict]:
    """Fit F_W=1-WSGD from DSSAT without using biomass, yield, or field data."""
    if "water_stress_growth" not in data:
        raise ValueError("teacher cache has no water_stress_growth/WSGD; rebuild it first")
    relative_water = data["soil_water_extractable_mm"] / field_capacity
    target = 1.0 - data["water_stress_growth"]
    mask = (
        np.isfinite(relative_water) & np.isfinite(target)
        & np.isfinite(data["lai"]) & (data["lai"] > 0.02)
    )
    x = relative_water[mask]
    y = target[mask]
    best = (float("inf"), 0.2, 9.0)
    threshold_grid = np.linspace(0.0, 0.6, 31)
    slope_grid = np.linspace(1.0, 30.0, 30)
    for refinement in range(4):
        for threshold in threshold_grid:
            for slope in slope_grid:
                prediction = 1.0 / (1.0 + np.exp(-slope * (x - threshold)))
                mse = float(np.mean((prediction - y) ** 2))
                if mse < best[0]:
                    best = (mse, float(threshold), float(slope))
        threshold_span = 0.03 / (2 ** refinement)
        slope_span = 2.0 / (2 ** refinement)
        threshold_grid = np.linspace(max(0.0, best[1] - threshold_span), best[1] + threshold_span, 13)
        slope_grid = np.linspace(max(0.1, best[2] - slope_span), best[2] + slope_span, 13)
    metrics = {
        "records": int(mask.sum()),
        "target_mean": float(y.mean()),
        "prediction_mean": float((1.0 / (1.0 + np.exp(-best[2] * (x - best[1])))).mean()),
        "rmse": float(np.sqrt(best[0])),
    }
    return {"stress_threshold": best[1], "stress_slope": best[2]}, metrics


def evaluate_growth_stress(
    params: dict[str, float], data: dict[str, np.ndarray], field_capacity: float,
) -> dict | None:
    if "water_stress_growth" not in data:
        return None
    relative_water = data["soil_water_extractable_mm"] / field_capacity
    target = 1.0 - data["water_stress_growth"]
    mask = (
        np.isfinite(relative_water) & np.isfinite(target)
        & np.isfinite(data["lai"]) & (data["lai"] > 0.02)
    )
    prediction = 1.0 / (1.0 + np.exp(
        -params["stress_slope"] * (relative_water[mask] - params["stress_threshold"])
    ))
    return {
        "records": int(mask.sum()),
        "target_mean": float(target[mask].mean()),
        "prediction_mean": float(prediction.mean()),
        "rmse": float(np.sqrt(np.mean((prediction - target[mask]) ** 2))),
        "correlation": float(np.corrcoef(prediction, target[mask])[0, 1]),
    }


def evaluate_reference(
    params: dict[str, float], data: dict[str, np.ndarray], weather: dict[str, np.ndarray]
) -> dict[str, float]:
    pred = simulate(params, data, weather)
    metrics = {
        "soil_water_rmse_mm": float(np.sqrt(np.nanmean(
            (pred["soil_water_extractable_mm"] - data["soil_water_extractable_mm"]) ** 2
        ))),
        "et_rmse_mm_day": float(np.sqrt(np.nanmean((pred["et_mm"] - data["et_mm"]) ** 2))),
        "water_bias_mm": float(np.nanmean(
            pred["soil_water_extractable_mm"] - data["soil_water_extractable_mm"]
        )),
        "et_bias_mm_day": float(np.nanmean(pred["et_mm"] - data["et_mm"])),
    }
    if "soil_evaporation_mm" in data and "transpiration_mm" in data:
        metrics.update({
            "soil_evaporation_rmse_mm_day": float(np.sqrt(np.nanmean(
                (pred["soil_evaporation_mm"] - data["soil_evaporation_mm"]) ** 2
            ))),
            "transpiration_rmse_mm_day": float(np.sqrt(np.nanmean(
                (pred["transpiration_mm"] - data["transpiration_mm"]) ** 2
            ))),
            "soil_evaporation_bias_mm_day": float(np.nanmean(
                pred["soil_evaporation_mm"] - data["soil_evaporation_mm"]
            )),
            "transpiration_bias_mm_day": float(np.nanmean(
                pred["transpiration_mm"] - data["transpiration_mm"]
            )),
            "teacher_et_component_closure_max_mm_day": float(np.nanmax(np.abs(
                data["et_mm"] - data["soil_evaporation_mm"] - data["transpiration_mm"]
            ))),
        })
    return metrics


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation without adding a scipy dependency."""
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty_like(order, dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result
    a, b = ranks(left), ranks(right)
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--validation-weather", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--sowing-date", default="2023-04-10")
    parser.add_argument(
        "--et-model", choices=(
            "radiation", "fao56_pm", "dual_source_radiation", "dssat_components_radiation",
        ),
        default="radiation",
    )
    parser.add_argument("--lai-model", choices=("sine", "double_sigmoid"), default="sine")
    parser.add_argument("--calibrate-et-stress-floor", action="store_true")
    parser.add_argument(
        "--fit-growth-stress", action="store_true",
        help="fit F_W directly to 1-WSGD in the calibration-year DSSAT cache",
    )
    args = parser.parse_args()
    raw = np.load(args.cache, allow_pickle=True)
    reference_metadata = json.loads(str(raw["metadata"]))
    manifest_candidates = (
        Path(reference_metadata["source"]) / "run_manifest.json",
        args.cache.parent / "run_manifest.json",
    )
    manifest_path = next((path for path in manifest_candidates if path.exists()), manifest_candidates[0])
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    paper_date_aligned = (
        args.sowing_date == "2023-04-10"
        and manifest.get("changed_only_in_temporary_COX", {}).get("PDATE")
        == "23102 -> 23100 (2023-04-12 -> 2023-04-10)"
        and manifest.get("treatments") == 99
    )
    data, weather, _ = load_reference(args.cache, args.weather)
    sowing_date = date.fromisoformat(args.sowing_date)
    source_year = int(raw["year"][0])
    if sowing_date.year != source_year or np.any(raw["year"] != source_year):
        raise ValueError("Sowing date and DSSAT teacher trajectory must belong to the same year")
    data["sowing_doy"] = sowing_date.timetuple().tm_yday
    # With AA soil initialized at DUL, the first SWXD is field-capacity
    # extractable water (287 mm).  It is deliberately not the maximum SWXD,
    # because post-irrigation SWXD may temporarily exceed field capacity.
    field_capacity = float(np.nanmedian(data["soil_water_extractable_mm"][:, 0]))
    bounds = {
        "rain_efficiency": (0.60, 1.00), "irrigation_efficiency": (0.50, 1.00),
        "et_scale": (0.40, 1.50), "drainage_fraction": (0.05, 1.00),
    }
    if args.calibrate_et_stress_floor:
        bounds["et_stress_floor"] = (0.0, 0.4)
    if args.et_model == "dual_source_radiation":
        bounds.pop("et_scale")
        bounds.update({
            "surface_memory": (0.30, 0.95),
            "soil_evaporation_coefficient": (0.05, 1.00),
            "basal_crop_coefficient": (0.70, 1.60),
        })
    if args.et_model == "dssat_components_radiation":
        bounds.pop("et_scale")
    fixed_params = {
        "rue": 2.63405157330431, "k": 0.630041864066784,
        "max_lai": 3.787344538270601, "ttmat": 2600.0,
        "yield_coefficient": 0.42305774973367094,
        "field_capacity": field_capacity, "gate_width": 20.0,
        "drainage_smoothing": 10.0,
        # SWXD and ETAA do not identify a growth-stress response uniquely.
        # Keep the previously audited F_W shape fixed during water calibration.
        "stress_threshold": 0.6579627613886836,
        "stress_slope": 2.932634170384998,
        "et_model": args.et_model,
        "et_stress_floor": 0.3,
        "surface_memory": 0.70,
        "surface_wetting_scale_mm": 10.0,
        "soil_evaporation_coefficient": 0.50,
        "basal_crop_coefficient": 1.10,
        "lai_model": args.lai_model,
        # Fixed on 2022 teacher process components before water-balance search.
        "lai_asymptote": 5.885967456970629,
        "lai_growth_midpoint_gdd": 1039.7977587408282,
        "lai_growth_width_gdd": 167.49811192234083,
        "lai_decline_midpoint_gdd": 2284.342738225305,
        "lai_decline_width_gdd": 499.7057243055772,
        "potential_transpiration_scale": 0.8363597558664126,
        "transpiration_lai_extinction": 0.8941386340468911,
        "potential_soil_scale": 0.9741626699053171,
        "soil_lai_extinction": 0.4096330275229358,
        "transpiration_stress_threshold": 0.135,
        "transpiration_stress_slope": 11.25,
    }
    if args.et_model == "dssat_components_radiation":
        # Fit on 2022 ESAA/EOSA wetness decay, then freeze for 2023.
        fixed_params.update({"surface_memory": 0.92, "surface_wetting_scale_mm": 24.5})
    growth_stress_calibration = None
    if args.fit_growth_stress:
        fitted_stress, growth_stress_calibration = fit_growth_stress(data, field_capacity)
        fixed_params.update(fitted_stress)
    rng = np.random.default_rng(args.seed)
    names = list(bounds)
    best_total = float("inf")
    best_params: dict[str, float] = {}
    best_components: dict[str, float] = {}
    for _ in range(args.samples):
        params = {**fixed_params, **{name: float(rng.uniform(*bounds[name])) for name in names}}
        total, components = water_score(params, data, weather)
        if total < best_total:
            best_total, best_params, best_components = total, params, components
    # Small coordinate refinement around the best random point.
    for _ in range(4):
        improved = False
        for name, (lo, hi) in bounds.items():
            center = best_params[name]
            span = (hi - lo) * (0.12 / (2 ** _))
            for value in np.linspace(max(lo, center - span), min(hi, center + span), 7):
                candidate = dict(best_params)
                candidate[name] = float(value)
                total, components = water_score(candidate, data, weather)
                if total < best_total:
                    best_total, best_params, best_components = total, candidate, components
                    improved = True
        if not improved:
            break
    pred = simulate(best_params, data, weather)
    validation = None
    if bool(args.validation_cache) != bool(args.validation_weather):
        raise ValueError("--validation-cache and --validation-weather must be provided together")
    if args.validation_cache:
        validation_data, validation_weather, _ = load_reference(
            args.validation_cache, args.validation_weather
        )
        validation = {
            "cache": str(args.validation_cache),
            "weather": str(args.validation_weather),
            "metrics": evaluate_reference(best_params, validation_data, validation_weather),
        }
        if args.fit_growth_stress:
            validation["growth_stress_metrics"] = evaluate_growth_stress(
                best_params, validation_data, field_capacity
            )
    baseline_params = {
        **fixed_params,
        "et_model": "radiation",
        "et_stress_floor": 0.3,
        "et_scale": 1.0,
        "rain_efficiency": 0.8,
        "irrigation_efficiency": float(0.5 + 0.5 / (1.0 + np.exp(-2.20))),
        "et_scale": 1.0,
        "drainage_fraction": 1.0,
        "legacy_et_gate": True,
    }
    baseline_calibration_metrics = evaluate_reference(baseline_params, data, weather)
    baseline_validation_metrics = (
        evaluate_reference(baseline_params, validation_data, validation_weather)
        if args.validation_cache else None
    )
    calibration_metrics = evaluate_reference(best_params, data, weather)
    gate_metrics = ("soil_water_rmse_mm", "et_rmse_mm_day")
    acceptance_passed = bool(
        baseline_validation_metrics
        and all(calibration_metrics[name] < baseline_calibration_metrics[name] for name in gate_metrics)
        and all(validation["metrics"][name] < baseline_validation_metrics[name] for name in gate_metrics)
    )
    absolute_validation_passed = bool(
        validation
        and validation["metrics"]["soil_water_rmse_mm"] <= 40.0
        and validation["metrics"]["et_rmse_mm_day"] <= 1.55
    )
    result = {
        "method": "water_only_differentiable_process_calibration",
        "seed": args.seed,
        "random_samples": args.samples,
        "source_cache": str(args.cache),
        "source_weather": str(args.weather),
        "et_model": args.et_model,
        "lai_model": args.lai_model,
        "sowing_date_assumed": args.sowing_date,
        "sowing_date_source": "Wang et al. (2024) field protocol; teacher PDATE is verified by run_manifest.json when available",
        "teacher_run_manifest": str(manifest_path) if paper_date_aligned else None,
        "status": "paper_date_aligned_diagnostic" if paper_date_aligned else "diagnostic_not_paper_aligned",
        "records": int(data["irrigation_mm"].shape[0]),
        "dates": int(data["irrigation_mm"].shape[1]),
        "bounds": bounds,
        "best_params": best_params,
        "weighted_loss": best_total,
        "calibration_metrics": calibration_metrics,
        "growth_stress_calibration": growth_stress_calibration,
        "validation": validation,
        "legacy_baseline": {
            "params": baseline_params,
            "calibration_metrics": baseline_calibration_metrics,
            "validation_metrics": baseline_validation_metrics,
        },
        "acceptance": {
            "passed": acceptance_passed,
            "rule": "SWXD and ETAA RMSE must each improve over the legacy equation in both 2022 calibration and 2023 validation",
            "action": "eligible_for_50_group_training" if acceptance_passed else "stop_before_50_group_training",
        },
        "absolute_validation_gate": {
            "passed": absolute_validation_passed,
            "thresholds": {"soil_water_rmse_mm": 40.0, "et_rmse_mm_day": 1.55},
            "role": "preset development gate, not an ICASSP requirement",
        },
        "yield_rank_spearman": rank_correlation(pred["yield_kg_ha"], data["yield_kg_ha"]),
        "target_ranges": {
            "soil_water_extractable_mm": [
                float(np.nanmin(data["soil_water_extractable_mm"])),
                float(np.nanmax(data["soil_water_extractable_mm"])),
            ],
            "et_mm": [float(np.nanmin(data["et_mm"])), float(np.nanmax(data["et_mm"]))],
            "lai": [float(np.nanmin(data["lai"])), float(np.nanmax(data["lai"]))],
            "biomass_kg_ha": [float(np.nanmin(data["biomass_kg_ha"])), float(np.nanmax(data["biomass_kg_ha"]))],
            "yield_kg_ha": [float(np.nanmin(data["yield_kg_ha"])), float(np.nanmax(data["yield_kg_ha"]))],
        },
        "prediction_ranges": {name: [float(np.nanmin(value)), float(np.nanmax(value))] for name, value in pred.items()},
        "notes": [
            "Only SWXD and ETAA tune water parameters; LAI, biomass, yield, and 2024 field data are excluded from selection.",
            "Field capacity is the first-day median SWXD under AA soil initialized at DUL; post-irrigation SWXD may exceed it.",
            "Excess above field capacity drains gradually through a calibrated fraction instead of being removed completely on the same day.",
            "Initial root-zone water is initialized from each DSSAT treatment's first extractable-water value.",
            "F_W is fitted directly to 1-WSGD and excludes biomass, yield, and 2024 field data." if args.fit_growth_stress else "F_W threshold and slope are fixed during water calibration because SWXD/ETAA do not identify growth stress uniquely.",
            "Teacher PDATE is April 10 in a temporary COX; emergence and other management dates were not altered." if paper_date_aligned else "The teacher run does not have a verified April 10 PDATE in its run manifest.",
            "This calibration is a fixed-process diagnostic; it is not yet wired into the DNN checkpoint.",
            "The same parameter file will be used before adding bounded DNN corrections.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
