# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
import os
import base64
import streamlit.components.v1 as components

from utils import (
    p_wave_features_calc,
    detect_p_and_window_from_trace,
    read_uploaded_trace,
    fetch_trace_from_iris,
)

st.set_page_config(page_title="PGA from P-wave features (XGB)", layout="wide")

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgb_eew_final.joblib")
PREPROC_PATH = os.path.join(ARTIFACT_DIR, "preproc_objects.joblib")

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROC_PATH):
        raise FileNotFoundError(
            f"Missing artifacts. Please ensure:\n - {MODEL_PATH}\n - {PREPROC_PATH}\n"
            "These are produced by your notebook (joblib.dump)."
        )
    model = joblib.load(MODEL_PATH)
    preproc = joblib.load(PREPROC_PATH)
    # expected keys in preproc: 'scaler','imputer','selector','p_wave_features'
    scaler = preproc['scaler']
    imputer = preproc['imputer']
    selector = preproc['selector']
    p_wave_features = preproc['p_wave_features']
    return model, scaler, imputer, selector, p_wave_features

def preprocess_and_predict(feats: dict, model, scaler, imputer, selector, p_wave_features):
    feats_df = pd.DataFrame([feats])
    # same transformations used in notebook: log1p on features
    X_log = np.log1p(feats_df[p_wave_features].astype(float))
    X_scaled = scaler.transform(X_log)
    X_imputed = imputer.transform(X_scaled)
    X_sel = selector.transform(X_imputed)
    pred_log = model.predict(X_sel)[0]
    pred_pga_cm_s2 = float(np.expm1(pred_log))
    pred_pga_g = pred_pga_cm_s2 / 980.0
    return pred_pga_cm_s2, pred_pga_g, X_log.iloc[0].to_dict()

# --- UI ---
st.title("PGA prediction from P-wave features — XGBoost pipeline")
st.markdown(
    "This app uses the **exact preprocessing pipeline** (scaler, imputer, selector) "
    "and a trained XGBoost regressor to predict PGA (cm/s²) from 17 P-wave features."
)

left, right = st.columns([1, 1.6])

with left:
    st.header("Input waveform")
    upload = st.file_uploader("Upload a waveform (miniSEED / SAC) or choose 'Fetch from IRIS'", type=['mseed','sac','msd','mseed.gz'], accept_multiple_files=False)
    st.markdown("Or fetch from IRIS by specifying network/station/time below.")
    st.write("## Fetch options")
    net = st.text_input("Network", value="IU")
    station = st.text_input("Station", value="ANMO")
    date_str = st.text_input("Start UTC (YYYY-mm-ddTHH:MM) or leave blank for random sample", "")
    duration_sec = st.number_input("Duration (seconds)", min_value=10, max_value=7200, value=3600, step=60)

    fetch_button = st.button("Fetch & predict (IRIS)")

with right:
    st.header("Model / Artifacts")
    st.write("Artifacts are loaded from `./artifacts/` (joblib). This ensures predictions are identical to the notebook.")
    try:
        model, scaler, imputer, selector, p_wave_features = load_artifacts()
        st.success("✅ Artifacts loaded.")
        st.write(f"Features used ({len(p_wave_features)}):")
        st.write(", ".join(p_wave_features))
    except Exception as e:
        st.error("Artifacts not available: " + str(e))
        st.stop()

# Function to run when we have a trace object
def handle_trace_and_predict(trace_obj, trace_meta_desc="uploaded/iris"):
    # find p window + features using the same logic as notebook
    res = detect_p_and_window_from_trace(trace_obj)
    if res is None:
        st.warning("No P pick found in the waveform (or window too short). Try another trace or parameters.")
        return
    tr, p_window, p_index, feats = res['trace'], res['p_window'], res['p_index'], res['feats']
    pred_cm_s2, pred_g, X_log_dict = preprocess_and_predict(feats, model, scaler, imputer, selector, p_wave_features)

    # show summary and badge
    st.subheader(f"Prediction for {trace_meta_desc}")
    st.markdown(f"**Predicted PGA:** `{pred_cm_s2:.5g} cm/s²` — `{pred_g:.5g} g`")

    # Plot waveform, zoom P window, spectrogram
    fig, axs = plt.subplots(3,1, figsize=(9,8), constrained_layout=True)
    t = np.arange(tr.stats.npts) * tr.stats.delta
    axs[0].plot(t, tr.data); axs[0].axvline(p_index * tr.stats.delta, color='r', linestyle='--'); axs[0].set_title("Full seismogram")
    win_t = np.arange(len(p_window)) * tr.stats.delta + p_index * tr.stats.delta
    axs[1].plot(win_t, p_window); axs[1].set_title("P-window (zoom)")
    axs[2].specgram(tr.data, Fs=1.0/tr.stats.delta)
    st.pyplot(fig)

    # Feature table & bar chart
    feat_df = pd.DataFrame.from_dict(feats, orient='index', columns=['value']).reset_index().rename(columns={'index':'feature'})
    st.write("Extracted P-wave features (used for prediction)")
    st.dataframe(feat_df.style.format({'value':'{:.6g}'}), height=300)

    # Download CSV of sample
    out = {**trace_obj.stats.__dict__, **feats, 'pred_pga_cm_s2': pred_cm_s2, 'pred_pga_g': pred_g}
    csv = pd.DataFrame([out]).to_csv(index=False).encode()
    st.download_button("Download CSV", csv, file_name="single_sample_prediction.csv", mime="text/csv")

    # Interactively show small folium map in HTML if station lat/lon available in trace.stats
    lat = getattr(trace_obj.stats, "coordinates", None)
    if lat is not None:
        # Some traces may include coordinates; if present, embed folium map
        pass

# Handle uploaded file
if upload is not None:
    try:
        tr = read_uploaded_trace(upload)
        handle_trace_and_predict(tr, trace_meta_desc=f"uploaded {upload.name}")
    except Exception as e:
        st.error("Failed to read uploaded file: " + str(e))

# Handle fetch from IRIS
if fetch_button:
    # parse date if provided
    try:
        if date_str.strip() == "":
            # random sample from IRIS — use the notebook helper
            st.info("Fetching a random sample from IRIS (same logic as notebook)...")
            tr = fetch_trace_from_iris(network=net, station=station, random_sample=True, duration_sec=duration_sec)
        else:
            start = pd.to_datetime(date_str)
            tr = fetch_trace_from_iris(network=net, station=station, starttime=start, duration_sec=duration_sec)
        handle_trace_and_predict(tr, trace_meta_desc=f"{net}.{station} {date_str or 'random'}")
    except Exception as e:
        st.error("Fetch/predict failed: " + str(e))

st.markdown("---")
st.caption("App created to mirror notebook pipeline: feature extraction → log1p → scaler → imputer → selector → XGBoost (trained in notebook).")
