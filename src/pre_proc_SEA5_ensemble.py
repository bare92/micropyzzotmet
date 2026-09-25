import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def load_configuration(config_file):
    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    required = ["source_root", "output_root", "emission_year", "emission_month"]
    missing = [key for key in required if key not in config]

    if missing:
        raise KeyError("Missing configuration option(s): " + ", ".join(missing))

    config["source_root"] = Path(config["source_root"]).expanduser()
    config["output_root"] = Path(config["output_root"]).expanduser()
    config["emission_year"] = int(config["emission_year"])
    config["emission_month"] = int(config["emission_month"])

    if not 1 <= config["emission_month"] <= 12:
        raise ValueError("emission_month must be between 1 and 12")

    config.setdefault("forecast_months_ahead", 3)
    config["forecast_months_ahead"] = int(config["forecast_months_ahead"])

    if config["forecast_months_ahead"] < 0:
        raise ValueError("forecast_months_ahead must be >= 0")

    config.setdefault("source_subfolder_template", "SEAS5_{year:04d}_{month:02d}")
    config.setdefault("source_filename_pattern", "SEAS5_BC_EM{emission}_FM*.nc")
    config.setdefault("output_folder_template", "EM{emission}_FM{forecast}")
    config.setdefault("output_filename_template", "SEAS5_9km_EM{emission}_FM{forecast}_{member}.nc")
    config.setdefault("compression", True)
    config.setdefault("compression_level", 4)
    config.setdefault("overwrite", False)

    return config


def parse_forecast_month(path):
    stem = path.stem

    if "_FM" not in stem:
        raise ValueError(f"Cannot find '_FM' in filename: {path.name}")

    token = stem.split("_FM", 1)[1]
    forecast_string = token[:6]

    if len(forecast_string) != 6 or not forecast_string.isdigit():
        raise ValueError(f"Cannot determine forecast month from: {path.name}")

    year = int(forecast_string[:4])
    month = int(forecast_string[4:6])

    if not 1 <= month <= 12:
        raise ValueError(f"Invalid forecast month in {path.name}: {forecast_string}")

    return year, month


def find_monthly_files(config):
    year = config["emission_year"]
    month = config["emission_month"]
    emission = f"{year:04d}{month:02d}"

    subfolder = config["source_subfolder_template"].format(year=year, month=month)
    source_dir = config["source_root"] / subfolder
    pattern = config["source_filename_pattern"].format(emission=emission)
    candidate_files = sorted(source_dir.glob(pattern))

    if not candidate_files:
        raise FileNotFoundError(f"No monthly SEAS5 files found in {source_dir} using pattern {pattern}")

    allowed = []

    for offset in range(config["forecast_months_ahead"] + 1):
        total_months = year * 12 + (month - 1) + offset
        forecast_year = total_months // 12
        forecast_month = total_months % 12 + 1
        allowed.append((forecast_year, forecast_month))

    allowed = set(allowed)

    logging.info("Allowed forecast months: %s", ", ".join(f"{y:04d}{m:02d}" for y, m in sorted(allowed)))

    selected = []

    for path in candidate_files:
        forecast_year, forecast_month = parse_forecast_month(path)

        if (forecast_year, forecast_month) in allowed:
            selected.append(path)
            logging.info("Selected FM %04d-%02d: %s", forecast_year, forecast_month, path.name)
        else:
            logging.info("Skipping FM %04d-%02d: %s", forecast_year, forecast_month, path.name)

    if not selected:
        raise RuntimeError("No monthly files fall inside the requested forecast-month window")

    return selected


def sanitize_attrs(attrs):
    clean = {}

    for key, value in attrs.items():
        if isinstance(value, (bool, np.bool_)):
            clean[key] = int(value)
        elif value is None:
            clean[key] = ""
        else:
            clean[key] = value

    return clean


def netcdf_safe_dataset(ds):
    out = ds.copy(deep=False)
    out.attrs = sanitize_attrs(out.attrs)

    for name in out.variables:
        out[name].attrs = sanitize_attrs(out[name].attrs)

    return out


def validate_input_dataset(ds, source_file):
    required_coords = ["latitude", "longitude"]
    missing = [coord for coord in required_coords if coord not in ds.coords]

    if missing:
        raise ValueError(f"{source_file.name}: missing coordinate(s): " + ", ".join(missing))

    if "time" not in ds.coords and "valid_time" not in ds.coords:
        raise ValueError(f"{source_file.name}: neither 'time' nor 'valid_time' coordinate is available")

    if "number" not in ds.dims:
        raise ValueError(f"{source_file.name}: ensemble dimension 'number' not found")


def validate_member_dataset(ds, output_file):
    if "number" in ds.dims:
        raise ValueError(f"{output_file.name}: dimension 'number' still present after split")

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError(f"{output_file.name}: missing latitude/longitude")

    if "time" not in ds.coords and "valid_time" not in ds.coords:
        raise ValueError(f"{output_file.name}: missing time coordinate")

    for variable in ("t2m", "d2m"):
        if variable in ds:
            units = str(ds[variable].attrs.get("units", "")).strip().lower()

            if units not in {"k", "kelvin"}:
                logging.warning("%s in %s has units '%s', expected Kelvin", variable, output_file.name, units)


def save_netcdf(ds, output_file, compression=True, compression_level=4):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    safe = netcdf_safe_dataset(ds)

    if not compression:
        safe.to_netcdf(output_file)
        return

    encoding = {variable: {"zlib": True, "complevel": int(compression_level)} for variable in safe.data_vars}

    try:
        safe.to_netcdf(output_file, encoding=encoding)
    except ValueError as exc:
        if "unexpected encoding" not in str(exc).lower():
            raise

        logging.warning("Compression unsupported. Saving uncompressed: %s", output_file.name)
        safe.to_netcdf(output_file)


def split_monthly_file(source_file, config):
    emission = f"{config['emission_year']:04d}{config['emission_month']:02d}"
    forecast_year, forecast_month = parse_forecast_month(source_file)
    forecast = f"{forecast_year:04d}{forecast_month:02d}"

    output_folder_name = config["output_folder_template"].format(emission=emission, forecast=forecast)
    output_dir = config["output_root"] / output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 72)
    logging.info("SOURCE   : %s", source_file)
    logging.info("EMISSION : %s", emission)
    logging.info("FORECAST : %s", forecast)
    logging.info("OUTPUT   : %s", output_dir)

    with xr.open_dataset(source_file, decode_times=True) as ds:
        validate_input_dataset(ds, source_file)

        members = np.asarray(ds["number"].values)
        logging.info("Members  : %s", ", ".join(str(int(member)) for member in members))

        for member in members:
            member_value = int(member)

            output_filename = config["output_filename_template"].format(
                emission=emission,
                forecast=forecast,
                member=member_value
            )

            output_file = output_dir / output_filename

            if output_file.exists() and not config["overwrite"]:
                logging.info("Already exists, skipping: %s", output_file)
                continue

            logging.info("Processing member %d", member_value)

            member_ds = ds.sel(number=member).load()
            member_ds.attrs = dict(ds.attrs)

            member_ds.attrs.update({
                "ensemble_member": member_value,
                "emission_year": int(config["emission_year"]),
                "emission_month": int(config["emission_month"]),
                "forecast_year": int(forecast_year),
                "forecast_month": int(forecast_month),
                "source_file": source_file.name,
                "ensemble_split": "single member"
            })

            validate_member_dataset(member_ds, output_file)

            logging.info("Saving: %s", output_file)

            save_netcdf(
                member_ds,
                output_file,
                compression=config["compression"],
                compression_level=config["compression_level"]
            )

            with xr.open_dataset(output_file, decode_times=True) as check:
                validate_member_dataset(check, output_file)
                variables = ", ".join(check.data_vars)
                logging.info("Verified member %d | vars=%s", member_value, variables)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Split monthly SEAS5 bias-corrected NetCDF files into one file per ensemble member."
    )
    parser.add_argument("config", help="Path to JSON configuration file")
    args = parser.parse_args()

    config = load_configuration(args.config)
    files = find_monthly_files(config)

    year = config["emission_year"]
    month = config["emission_month"]
    allowed = []

    for offset in range(config["forecast_months_ahead"] + 1):
        total_months = year * 12 + (month - 1) + offset
        forecast_year = total_months // 12
        forecast_month = total_months % 12 + 1
        allowed.append((forecast_year, forecast_month))

    logging.info("=" * 72)
    logging.info("SEAS5 ENSEMBLE MEMBER SPLIT")
    logging.info("Emission: %04d-%02d", year, month)
    logging.info("Forecast range: EM -> EM + %d months", config["forecast_months_ahead"])
    logging.info("Forecast months: %s", ", ".join(f"{y:04d}{m:02d}" for y, m in allowed))
    logging.info("Monthly files selected: %d", len(files))
    logging.info("=" * 72)

    for source_file in files:
        split_monthly_file(source_file, config)

    logging.info("=" * 72)
    logging.info("COMPLETED")
    logging.info("=" * 72)


if __name__ == "__main__":
    main()