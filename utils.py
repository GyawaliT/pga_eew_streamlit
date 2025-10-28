# utils.py
import numpy as np
from obspy import read
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import random, datetime

client = Client("IRIS")

# IMPORTANT: keep p_wave_features list identical to notebook's p_wave_features
p_wave_features = [
    "durP","PDd","PVd","PAd","PDt","PVt","PAt",
    "p_mean","p_std","p_skew","p_kurt","p_absmax","p_absmean",
    "p_rms","p_energy","p_entropy","p_peak_index"
]
# If your notebook uses a different ordering or names, ensure the same list is saved in preproc_objects.joblib.

def p_wave_features_calc(window: np.ndarray, dt: float) -> dict:
    # copy exactly from notebook to preserve features (same math ops)
    if len(window) == 0:
        return {k: np.nan for k in p_wave_features}
    durP = len(window) * dt
    PDd = np.max(window) - np.min(window)
    grad = np.gradient(window) / dt
    PVd = np.max(np.abs(grad)) if len(grad) > 0 else 0
    PAd = np.mean(np.abs(window))
    PDt = np.max(window)
    PVt = np.max(grad) if len(grad) > 0 else 0
    PAt = np.sqrt(np.mean(window ** 2))
    p_mean = np.mean(window)
    p_std = np.std(window)
    # minor numeric stability checks
    p_skew = float(np.nan) if len(window) < 3 else float(((np.mean((window - p_mean)**3))/ (p_std**3 + 1e-12)))
    p_kurt = float(np.nan)
    try:
        from scipy.stats import kurtosis
        p_kurt = kurtosis(window, fisher=True, bias=False)
    except Exception:
        p_kurt = np.nan
    p_absmax = np.max(np.abs(window))
    p_absmean = np.mean(np.abs(window))
    p_rms = np.sqrt(np.mean(window**2))
    p_energy = np.sum(window**2)
    # entropy simple approximation (shannon with histogram)
    hist, edges = np.histogram(window, bins=20, density=True)
    probs = hist.clip(min=1e-12)
    p_entropy = -np.sum(probs * np.log(probs))
    p_peak_index = float(np.argmax(np.abs(window)))
    feats = {
        "durP": durP, "PDd": PDd, "PVd": PVd, "PAd": PAd, "PDt": PDt, "PVt": PVt, "PAt": PAt,
        "p_mean": p_mean, "p_std": p_std, "p_skew": p_skew, "p_kurt": p_kurt,
        "p_absmax": p_absmax, "p_absmean": p_absmean, "p_rms": p_rms,
        "p_energy": p_energy, "p_entropy": p_entropy, "p_peak_index": p_peak_index
    }
    # ensure ordering (same keys)
    ordered = {k: feats[k] for k in p_wave_features}
    return ordered

def detect_p_and_window_from_trace(tr, sta=1, lta=10, on=2.5, off=1.0, win_before=0.0, win_after=2.0):
    """
    tr: obspy Trace
    returns dict with keys: trace, p_window (np array), p_index (sample index), feats dict
    This uses classic_sta_lta and trigger thresholds — same logic as notebook.
    """
    tr = tr.copy()
    tr.detrend("demean")
    tr.filter("bandpass", freqmin=0.5, freqmax=20.0)
    dt = tr.stats.delta
    try:
        cft = classic_sta_lta(tr.data, int(sta/dt), int(lta/dt))
    except Exception:
        # fallback if dt small/large
        cft = classic_sta_lta(tr.data, int(1/dt), int(10/dt))
    trig = trigger_onset(cft, on, off)
    if len(trig) == 0:
        return None
    p_index = int(trig[0][0])
    start = max(0, p_index - int(win_before / dt))
    end = min(tr.stats.npts, p_index + int(win_after / dt))
    p_window = tr.data[start:end]
    feats = p_wave_features_calc(p_window, dt)
    meta = {"station": tr.stats.station, "network": tr.stats.network, "sampling_rate": tr.stats.sampling_rate, "p_index": p_index, "win_samples": len(p_window)}
    return {"trace": tr, "p_window": p_window, "p_index": p_index, "feats": feats, "meta": meta}

def read_uploaded_trace(uploaded_file):
    # accepts a file-like uploaded via Streamlit
    # write to BytesIO and read via obspy
    bio = uploaded_file.read()
    b = BytesIO(bio)
    st = read(b)  # obspy can infer format
    return st[0]

def fetch_trace_from_iris(network='IU', station='ANMO', starttime=None, duration_sec=3600, random_sample=False):
    """
    If random_sample=True, picks a random year/day/time (similar to notebook).
    Returns first Trace.
    """
    max_attempts = 8
    stations = [station] if station else ["ANMO","COR","MAJO","KBL"]
    for attempt in range(max_attempts):
        try:
            if random_sample or starttime is None:
                yr = random.choice([2022,2023,2024])
                st_time = UTCDateTime(datetime.datetime(yr, random.randint(1,12), random.randint(1,25),
                                                        random.randint(0,21), 0, 0))
            else:
                st_time = UTCDateTime(starttime.astype("datetime64[s]").astype(datetime.datetime))
            endtime = st_time + duration_sec
            # try each provided station candidate
            for stn in stations:
                try:
                    st = client.get_waveforms(network, stn, "*", "BHZ", st_time, endtime)
                    if st and len(st) > 0:
                        return st[0]
                except Exception:
                    continue
        except Exception:
            continue
    raise RuntimeError("Failed to fetch seismogram from IRIS")

