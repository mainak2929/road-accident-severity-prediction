import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.compose import ColumnTransformer

from src.features import CAT_COLS, CLASS_NAMES, MODELS_DIR, load_pipeline, time_of_day_band

CLASS_COLORS = {"Fatal": "#2a78d6", "Serious": "#eb6834", "Slight": "#1baf7a"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LOCATIONS = {
    "London": (51.507, -0.128), "Birmingham": (52.486, -1.890), "Manchester": (53.480, -2.242),
    "Leeds": (53.800, -1.549), "Glasgow": (55.864, -4.252), "Cardiff": (51.482, -3.179),
    "Rural Cumbria": (54.460, -3.090), "Scottish Highlands": (57.480, -4.900), "Custom": None,
}
FIELD_LABELS = {
    "road_type_label": "Road type", "first_road_class_label": "Road class",
    "urban_or_rural_area_label": "Urban or rural", "light_conditions_label": "Light conditions",
    "weather_conditions_label": "Weather", "road_surface_conditions_label": "Road surface",
    "junction_detail_label": "Junction detail", "junction_control_label": "Junction control",
}

st.set_page_config(page_title="Collision Severity Demo", layout="wide")


@st.cache_resource
def load_model():
    summary_path = MODELS_DIR / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    model_file = ROOT / summary.get("selected_model_file", "models/best_model_pipeline.joblib")
    model = load_pipeline(model_file)
    levels = {}
    for _, step in model.steps:
        if isinstance(step, ColumnTransformer):
            enc = step.named_transformers_["cat"]
            levels = {c: list(cats) for c, cats in zip(CAT_COLS, enc.categories_)}
            break
    return model, summary, model_file.name, levels


@st.cache_data
def load_holdout():
    return pd.read_parquet(MODELS_DIR / "test_holdout_with_predictions.parquet")


model, summary, model_name, levels = load_model()
holdout = load_holdout()


@st.cache_data
def holdout_fatal_scores(_model_name):
    return model.predict_proba(holdout)[:, 0]


DEFAULTS = {
    "speed_limit": 30, "number_of_vehicles": 2, "location": "London", "latitude": 51.507, "longitude": -0.128,
    "collision_hour": 14, "month": "May", "day": "Wednesday",
    "road_type_label": "Single carriageway", "first_road_class_label": "A", "urban_or_rural_area_label": "Urban",
    "light_conditions_label": "Daylight", "weather_conditions_label": "Fine no high winds",
    "road_surface_conditions_label": "Dry", "junction_detail_label": "Not at junction",
    "junction_control_label": "Give way or uncontrolled",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("example_truth", None)


def load_random_example(severity=None):
    pool = holdout if severity is None else holdout[holdout["target_severity_label"] == severity]
    row = pool.sample(1).iloc[0]
    st.session_state.update({
        "speed_limit": int(row.speed_limit), "number_of_vehicles": int(row.number_of_vehicles),
        "location": "Custom", "latitude": float(row.latitude), "longitude": float(row.longitude),
        "collision_hour": int(row.collision_hour), "month": MONTHS[int(row.collision_month) - 1],
        "day": row.day_of_week_name if row.day_of_week_name in DAYS else ("Saturday" if row.is_weekend else "Wednesday"),
        **{c: row[c] for c in FIELD_LABELS if row[c] in levels.get(c, [row[c]])},
        "example_truth": (row.collision_index, row.target_severity_label),
    })


def on_location_change():
    coords = LOCATIONS.get(st.session_state["location"])
    if coords:
        st.session_state["latitude"], st.session_state["longitude"] = coords


st.title("Road-collision severity classifier")
st.caption(
    f"Model: `{model_name}` · selected config: **{summary.get('selected_config', 'RF - class weights')}** · "
    "trained on GB police-reported collisions 2019–2023 (STATS19). Educational demo, not a decision tool."
)

with st.sidebar:
    st.header("Try a real holdout collision")
    st.write("These rows were never used in training.")
    c1, c2 = st.columns(2)
    c1.button("Random", on_click=load_random_example, use_container_width=True)
    c2.button("Random fatal", on_click=load_random_example, args=("Fatal",), use_container_width=True)
    c1.button("Random serious", on_click=load_random_example, args=("Serious",), use_container_width=True)
    c2.button("Random slight", on_click=load_random_example, args=("Slight",), use_container_width=True)
    if st.session_state["example_truth"]:
        cid, truth = st.session_state["example_truth"]
        st.info(f"Loaded collision `{cid}`. Recorded severity: **{truth}**")
    m = summary.get("holdout_metrics")
    if m:
        st.header("Holdout performance")
        st.write(f"Macro F1 **{m['Macro F1']:.3f}** · balanced accuracy **{m['Balanced Accuracy']:.3f}** · accuracy {m['Accuracy']:.3f}")
        st.dataframe(pd.DataFrame({c: {"Recall": m[f"{c} Recall"], "Precision": m[f"{c} Precision"]} for c in CLASS_NAMES}).T.style.format("{:.2f}"))

st.subheader("Collision circumstances")
a, b, c = st.columns(3)
with a:
    st.markdown("**Road**")
    st.select_slider("Speed limit (mph)", options=[20, 30, 40, 50, 60, 70], key="speed_limit")
    for f in ["road_type_label", "first_road_class_label", "urban_or_rural_area_label", "junction_detail_label", "junction_control_label"]:
        st.selectbox(FIELD_LABELS[f], levels[f], key=f)
with b:
    st.markdown("**Conditions and vehicles**")
    st.number_input("Number of vehicles involved", 1, 17, key="number_of_vehicles")
    for f in ["light_conditions_label", "weather_conditions_label", "road_surface_conditions_label"]:
        st.selectbox(FIELD_LABELS[f], levels[f], key=f)
with c:
    st.markdown("**Time and place**")
    st.slider("Hour of day", 0, 23, key="collision_hour")
    st.selectbox("Month", MONTHS, key="month")
    st.selectbox("Day of week", DAYS, key="day")
    st.selectbox("Location", list(LOCATIONS), key="location", on_change=on_location_change)
    st.number_input("Latitude", 49.8, 60.9, format="%.4f", key="latitude")
    st.number_input("Longitude", -8.6, 1.8, format="%.4f", key="longitude")

if st.session_state["junction_detail_label"] == "Missing / Unknown":
    st.caption("Note: in the cleaned data, 'Missing / Unknown' junction detail mostly means *at a junction* "
               "(junction-type codes were not mapped to labels in cleaning).")

hour = st.session_state["collision_hour"]
row = pd.DataFrame([{
    "speed_limit": float(st.session_state["speed_limit"]),
    "number_of_vehicles": int(st.session_state["number_of_vehicles"]),
    "latitude": float(st.session_state["latitude"]), "longitude": float(st.session_state["longitude"]),
    "collision_hour": hour, "collision_month": MONTHS.index(st.session_state["month"]) + 1,
    "is_weekend": int(st.session_state["day"] in ("Saturday", "Sunday")),
    "time_of_day_band": time_of_day_band(hour),
    **{f: st.session_state[f] for f in FIELD_LABELS},
}])

proba = model.predict_proba(row)[0]
pred = CLASS_NAMES[int(np.argmax(proba))]
fatal_scores = holdout_fatal_scores(model_name)
pct = (fatal_scores < proba[0]).mean()

st.divider()
left, right = st.columns([1, 2])
with left:
    st.subheader("Prediction")
    st.metric("Predicted severity class", pred)
    st.metric("Fatal-risk score percentile", f"{pct:.0%}",
              help="Share of real holdout collisions with a lower Fatal score than this one.")
with right:
    st.subheader("Model scores per class")
    for name, p in zip(CLASS_NAMES, proba):
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0'>"
            f"<span style='width:64px'>{name}</span>"
            f"<div style='flex:1;background:#e4e3df;border-radius:4px;height:18px'>"
            f"<div style='width:{p*100:.1f}%;background:{CLASS_COLORS[name]};height:18px;border-radius:4px'></div></div>"
            f"<span style='width:56px;text-align:right'>{p:.1%}</span></div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "These are **class-weighted model scores, not real-world probabilities**. Training up-weights the rare "
        "Fatal and Serious classes, so their scores are inflated. Only about 1.5% of recorded collisions are fatal. "
        "Compare collisions with each other (the percentile) rather than reading a score as a literal chance."
    )

with st.expander("Responsible use: read before demoing"):
    st.markdown(
        """
- **Not for operational use.** Do not use this to triage emergencies, set insurance prices, or judge any individual or location.
  Macro F1 is about 0.36, and most "Fatal" predictions are false alarms (fatal precision is a few percent).
- **Association, not causation.** The model learned correlations in police-reported data. Changing an input here
  (e.g. lowering the speed limit) shows how the *model* responds, not what would happen on a real road.
- **Reporting bias.** STATS19 covers only police-reported injury collisions. "Unknown" codes and location partly reflect
  recording practice, and severity coding changed with the CRASH system rollout.
- **Inputs are recorded at the scene.** `number_of_vehicles` and the conditions are known only once a collision has
  happened. This is a post-collision severity classifier, not a pre-trip risk predictor.
"""
    )
st.caption("Engineered input row passed to the pipeline:")
st.dataframe(row, hide_index=True)
