from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_collisions_2019_2023.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TARGET_MAP = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["Fatal", "Serious", "Slight"]

NUM_COLS = ["speed_limit", "number_of_vehicles", "latitude", "longitude", "kinetic_energy_proxy",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "speed_x_rural"]
CAT_COLS = ["road_type_label", "first_road_class_label", "urban_or_rural_area_label",
            "light_conditions_label", "weather_conditions_label", "road_surface_conditions_label",
            "junction_detail_label", "junction_control_label", "time_of_day_band"]
BIN_COLS = ["is_weekend", "is_rush_hour", "is_night_risk", "is_weekend_night",
            "single_vehicle_crash", "high_speed_unlit"]

RAW_INPUT_COLS = ["speed_limit", "number_of_vehicles", "latitude", "longitude",
                  "collision_hour", "collision_month", "is_weekend"] + CAT_COLS


def extract_engineered_features(data):
    df_in = data.copy()
    X = pd.DataFrame(index=df_in.index)

    X["speed_limit"] = df_in["speed_limit"].astype(float)
    X["number_of_vehicles"] = df_in["number_of_vehicles"].astype(int)
    X["latitude"] = df_in["latitude"].astype(float)
    X["longitude"] = df_in["longitude"].astype(float)
    X["kinetic_energy_proxy"] = (X["speed_limit"] / 10.0) ** 2

    hour = df_in["collision_hour"].astype(int)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    month = df_in["collision_month"].astype(int)
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    is_weekend = df_in["is_weekend"].astype(int)
    X["is_weekend"] = is_weekend
    is_weekday = (is_weekend == 0).astype(int)

    X["is_rush_hour"] = (is_weekday & hour.isin([7, 8, 9, 16, 17, 18])).astype(int)
    X["is_night_risk"] = hour.isin([1, 2, 3, 4]).astype(int)
    X["is_weekend_night"] = (is_weekend & hour.isin([22, 23, 0, 1, 2, 3, 4])).astype(int)
    X["single_vehicle_crash"] = (X["number_of_vehicles"] == 1).astype(int)

    is_rural = (df_in["urban_or_rural_area_label"].astype(str).str.lower() == "rural").astype(int)
    X["speed_x_rural"] = X["speed_limit"] * is_rural

    is_unlit = (df_in["light_conditions_label"].astype(str).str.lower().str.contains("no lighting|unlit", na=False)).astype(int)
    X["high_speed_unlit"] = ((X["speed_limit"] >= 50) & (is_unlit == 1)).astype(int)

    cat_columns = [
        "road_type_label", "first_road_class_label", "urban_or_rural_area_label",
        "light_conditions_label", "weather_conditions_label", "road_surface_conditions_label",
        "junction_detail_label", "junction_control_label", "time_of_day_band"
    ]
    for c in cat_columns:
        if c in df_in.columns:
            X[c] = df_in[c].fillna("Unknown").astype(str)
        else:
            X[c] = "Unknown"

    return X


def time_of_day_band(hour: int) -> str:
    if 7 <= hour <= 9:
        return "Morning Peak (07:00-09:59)"
    if 10 <= hour <= 15:
        return "Daytime (10:00-15:59)"
    if 16 <= hour <= 18:
        return "Evening Peak (16:00-18:59)"
    return "Night / Off-Peak (19:00-06:59)"


def load_pipeline(path):
    import joblib

    main = sys.modules["__main__"]
    if not hasattr(main, "extract_engineered_features"):
        setattr(main, "extract_engineered_features", extract_engineered_features)
    return joblib.load(path)


def load_data_and_split(test_size: float = 0.20):
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(DATA_PATH)
    y = df["collision_severity"].map(TARGET_MAP).astype(int)
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return df, y, train_idx, test_idx
