import os
import geopandas as gpd
import numpy as np
import pandas as pd
import joblib
from icecream import ic
from tensorflow import keras

def build_features(df, N=14, offset=0):
    features = pd.DataFrame(index=df.index)
    days = [i for i in range(1+offset, N+1+offset) if i <= 14]

    # Extract rainfall window
    rain_vals = df[[f'P_{i}' for i in days]]

    # Rain
    features['rain_sum'] = rain_vals.sum(axis=1)
    features['rain_max'] = rain_vals.max(axis=1)
    features['rain_events'] = (rain_vals > 5).sum(axis=1)
    features['rain_var'] = rain_vals.var(axis=1)  # distribution
    features['rain_intensity'] = features['rain_sum'] / (features['rain_events'] + 1e-6)

    # Days since last rain >1mm
    def calc_days_since_rain(row):
        vals = row.values[::-1]  # reverse (day 14 back to day 1)
        for i, v in enumerate(vals, 1):
            if v > 1:
                return i
        return N  # no rain in window → max lag
    features['days_since_rain'] = rain_vals.apply(calc_days_since_rain, axis=1)

    # Longest dry spell (<1mm)
    def calc_dry_spell(row):
        dry_streak = 0
        max_streak = 0
        for v in row.values:
            if v < 1:
                dry_streak += 1
                max_streak = max(max_streak, dry_streak)
            else:
                dry_streak = 0
        return max_streak
    features['max_dry_spell'] = rain_vals.apply(calc_dry_spell, axis=1)

    # Temp
    features['temp_mean'] = df[[f'Temp_{i}' for i in days]].mean(axis=1)
    features['tmax_mean'] = df[[f'Tmax_{i}' for i in days]].mean(axis=1)
    features['tmin_mean'] = df[[f'Tmin_{i}' for i in days]].mean(axis=1)
    features['temp_range'] = features['tmax_mean'] - features['tmin_mean']

    # Humidity
    features['relhum_mean'] = df[[f'RelHum_{i}' for i in days]].mean(axis=1)
    features['relhum_min'] = df[[f'RelHum_{i}' for i in days]].min(axis=1)

    # Terrain
    features['dem'] = df['dem']
    features['slope'] = df['slope']

    return features



def prepare_data_for_prediction(row, climate_variables, num_days):
    """Prepare flat data for sklearn models"""
    training_feature_order = []

    for variable in climate_variables:
        for day_number in range(1, num_days + 1):
            feature_name = f"{variable}_{day_number}"
            training_feature_order.append(feature_name)

    training_feature_order.extend(["dem", "slope"])

    variables_for_prediction = {}
    for feature_name in training_feature_order:
        try:
            if feature_name in ["dem", "slope"]:
                variables_for_prediction[feature_name] = row[feature_name]
            else:
                if feature_name in row:
                    variables_for_prediction[feature_name] = row[feature_name]
                else:
                    variables_for_prediction[feature_name] = np.nan
        except KeyError as e:
            raise KeyError(
                f"Feature '{feature_name}' not found in the input data row."
            ) from e

    return variables_for_prediction


def prepare_sequence_for_lstm(row, climate_variables, num_days):
    """Prepare sequential input for LSTM: shape (1, timesteps, features)"""
    seq = []
    for t in range(1, num_days + 1):
        # ✅ use getattr for dynamic attribute access
        timestep_features = [getattr(row, f"{var}_{t}") for var in climate_variables]
        timestep_features.extend([getattr(row, "dem"), getattr(row, "slope")])
        seq.append(timestep_features)
    return np.array(seq).reshape(num_days, len(climate_variables) + 2)

def generarate_predictions(
    input_geojson,
    output_geojson,
    model_path="data/models/lr_model_v5.pkl",
    num_days: int = 14,
    use_engineered: bool = False,  # 👈 new flag
):
    """Generate predictions using climate data already embedded in the GeoJSON from API"""
    gdf = gpd.read_file(input_geojson)
    gdf = gdf.to_crs("EPSG:4326")

    # Load model
    if model_path.endswith(".pkl"):
        model = joblib.load(model_path)
        model_type = "sklearn"
    elif model_path.endswith(".keras"):
        model = keras.models.load_model(model_path)
        model_type = "lstm"
    else:
        raise ValueError(
            "Unsupported model format. Use .pkl for sklearn or .keras for LSTM"
        )

    climate_variables = ["P", "Pres", "RelHum", "SpecHum", "Temp", "Tmax", "Tmin"]

    ic(f"🔮 Making predictions for {len(gdf)} grid cells using {model_type} model...")

    if use_engineered:
        # Always use engineered feature pipeline regardless of model type
        df = build_features(gdf, N=num_days)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        preds = model.predict_proba(df)[:, 1]
        gdf["species_prediction"] = preds

        # Save engineered features for inspection
        for col in df.columns:
            if col not in gdf.columns:
                gdf[col] = df[col]

    else:
        if model_type == "sklearn":
            # Generic sklearn: use raw flat climate variables
            rows = []
            for row in gdf.itertuples(index=False):
                variables_for_prediction = prepare_data_for_prediction(
                    row, climate_variables, num_days
                )
                rows.append(variables_for_prediction)

            df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce").fillna(0)
            preds = model.predict_proba(df)[:, 1]
            gdf["species_prediction"] = preds

        elif model_type == "lstm":
            # Sequential input for keras model
            sequences = []
            for row in gdf.itertuples(index=False):
                seq = prepare_sequence_for_lstm(row, climate_variables, num_days)
                seq = np.nan_to_num(seq.astype("float32"), nan=0.0)
                sequences.append(seq)

            X = np.stack(sequences)  # shape: (n_samples, timesteps, features)
            preds = model.predict(X, verbose=0).flatten()
            gdf["species_prediction"] = preds

    # Save results
    output_dir = os.path.dirname(output_geojson)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")

    gdf.to_file(output_geojson, driver="GeoJSON")
    ic("✅ Predictions have been made and saved to", output_geojson)



def filter_predictions(path_output_geojson, path_filtered_geojson, threshold=0.01):
    # Remove polygons with a species_prediction value lower than 0.0001
    df = gpd.read_file(path_output_geojson)
    initial_count = len(df)
    df = df[df["species_prediction"] >= threshold]
    removed_count = initial_count - len(df)
    df = df[["geometry", "species_prediction"]]

    # Print the number of polygons removed
    ic(f"Number of polygons removed: {removed_count}")
    ic(f"Number of polygons left: {len(df)}")

    # Write the GeoDataFrame with predictions to a new GeoJSON file
    df.to_file(path_filtered_geojson, driver="GeoJSON")
    ic("Predictions have been made and saved to", path_filtered_geojson)
