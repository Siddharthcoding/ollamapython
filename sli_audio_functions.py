"""
sli_audio_functions.py
─────────────────────────────────────────────────────────────────────────────
All audio-related helpers for the LANNA SLI pipeline:
  • WAV loading & pre-processing
  • Feature extraction  (MFCCs, pitch, formant-proxies, jitter/shimmer, ZCR,
                         energy, spectral features)
  • SLI classifier inference
  • Whisper transcription (optional – falls back gracefully if not installed)
  • RAG indexing of audio findings
─────────────────────────────────────────────────────────────────────────────
Dependencies (all available in your env):
    numpy, scipy, scikit-learn, joblib, matplotlib, streamlit
Optional:
    openai-whisper  →  pip install openai-whisper
"""

import os
import io
import json
import struct
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter, find_peaks
try:
    from scipy.signal import hamming          # scipy < 1.8
except ImportError:
    from scipy.signal.windows import hamming  # scipy >= 1.8
from scipy.fft import fft, fftfreq
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
SLI_MODEL_PATH   = "models/sli_classifier.pkl"
SLI_SCALER_PATH  = "models/sli_scaler.pkl"
SLI_META_PATH    = "models/sli_meta.json"


# ═════════════════════════════════════════════
#  1.  WAV LOADING
# ═════════════════════════════════════════════

def load_wav(path: str):
    """
    Load a WAV file.  Returns (signal_float32, sample_rate).
    Handles stereo → mono by averaging channels.
    Resamples to 16 kHz for uniform feature extraction.
    """
    sr, data = wavfile.read(path)

    # convert to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # resample to 16 kHz using simple decimation/interpolation
    target_sr = 16000
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_len)
        data = np.interp(indices, np.arange(len(data)), data)
        sr = target_sr

    return data.astype(np.float32), sr


def trim_silence(signal: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Remove leading/trailing near-silence."""
    energy = signal ** 2
    mask = energy > threshold * energy.max()
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return signal
    return signal[indices[0]: indices[-1] + 1]


# ═════════════════════════════════════════════
#  2.  FEATURE EXTRACTION
# ═════════════════════════════════════════════

def _preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def _framing(signal, sr, frame_ms=25, step_ms=10):
    frame_len  = int(sr * frame_ms  / 1000)
    frame_step = int(sr * step_ms   / 1000)
    num_frames = 1 + (len(signal) - frame_len) // frame_step
    frames = np.stack([
        signal[i * frame_step: i * frame_step + frame_len]
        for i in range(num_frames)
    ])
    window = np.hamming(frame_len)
    return frames * window, frame_len, frame_step


def extract_mfcc(signal, sr, n_mfcc=13, n_fft=512, n_mels=26):
    """Return (n_mfcc,) mean + (n_mfcc,) std  →  2*n_mfcc values."""
    signal = _preemphasis(signal)
    frames, frame_len, _ = _framing(signal, sr)

    # power spectrum
    mag = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

    # mel filterbank
    fmin, fmax = 0, sr / 2
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bins[m-1], bins[m], bins[m+1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    filter_banks = np.dot(mag, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # DCT
    n_frames, _ = filter_banks.shape
    mfcc = np.zeros((n_frames, n_mfcc))
    for n in range(n_mfcc):
        mfcc[:, n] = np.sum(
            filter_banks * np.cos(np.pi * n / n_mels * (np.arange(n_mels) + 0.5)),
            axis=1
        )

    return np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])   # 26 values


def extract_pitch_features(signal, sr, frame_ms=40, step_ms=10,
                            f0_min=60, f0_max=500):
    """
    Autocorrelation-based F0 estimation.
    Returns: [f0_mean, f0_std, f0_min, f0_max, voiced_ratio]
    """
    frame_len  = int(sr * frame_ms  / 1000)
    frame_step = int(sr * step_ms   / 1000)
    num_frames = max(1, 1 + (len(signal) - frame_len) // frame_step)

    f0_values = []
    for i in range(num_frames):
        frame = signal[i * frame_step: i * frame_step + frame_len]
        if len(frame) < frame_len:
            break
        # autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]

        lag_min = int(sr / f0_max)
        lag_max = int(sr / f0_min)
        lag_max = min(lag_max, len(corr) - 1)

        if lag_max <= lag_min:
            continue

        peak_idx = np.argmax(corr[lag_min:lag_max]) + lag_min
        if corr[peak_idx] / (corr[0] + 1e-9) > 0.3:   # voiced threshold
            f0_values.append(sr / peak_idx)

    if not f0_values:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    f0 = np.array(f0_values)
    voiced_ratio = len(f0_values) / num_frames
    return [f0.mean(), f0.std(), f0.min(), f0.max(), voiced_ratio]


def extract_jitter_shimmer(signal, sr):
    """
    Simplified jitter (period variability) and shimmer (amplitude variability).
    Returns [jitter_pct, shimmer_pct]
    """
    frame_len  = int(sr * 0.04)
    frame_step = int(sr * 0.01)
    num_frames = max(1, 1 + (len(signal) - frame_len) // frame_step)

    periods, amps = [], []
    for i in range(num_frames):
        frame = signal[i * frame_step: i * frame_step + frame_len]
        if len(frame) < frame_len:
            break
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]
        lag_min, lag_max = int(sr / 500), int(sr / 60)
        lag_max = min(lag_max, len(corr) - 1)
        if lag_max <= lag_min:
            continue
        peak_idx = np.argmax(corr[lag_min:lag_max]) + lag_min
        if corr[peak_idx] / (corr[0] + 1e-9) > 0.3:
            periods.append(peak_idx / sr)
            amps.append(np.max(np.abs(frame)))

    if len(periods) < 2:
        return [0.0, 0.0]

    periods = np.array(periods)
    amps    = np.array(amps)
    jitter  = np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9) * 100
    shimmer = np.mean(np.abs(np.diff(amps)))    / (np.mean(amps)    + 1e-9) * 100
    return [min(jitter, 100.0), min(shimmer, 100.0)]


def extract_spectral_features(signal, sr, n_fft=512):
    """
    Returns [spectral_centroid, spectral_bandwidth, spectral_rolloff,
             spectral_flatness, zcr_mean, zcr_std, rms_mean, rms_std]
    """
    frames, frame_len, _ = _framing(signal, sr)
    mag = np.abs(np.fft.rfft(frames, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
    power = mag ** 2

    # centroid
    centroid = np.sum(freqs * power, axis=1) / (np.sum(power, axis=1) + 1e-9)

    # bandwidth
    bandwidth = np.sqrt(
        np.sum(((freqs - centroid[:, None]) ** 2) * power, axis=1) /
        (np.sum(power, axis=1) + 1e-9)
    )

    # rolloff (85%)
    cumpower = np.cumsum(power, axis=1)
    total    = cumpower[:, -1:] + 1e-9
    rolloff_idx = np.argmax(cumpower / total >= 0.85, axis=1)
    rolloff  = freqs[rolloff_idx]

    # flatness
    geom_mean = np.exp(np.mean(np.log(power + 1e-9), axis=1))
    arith_mean = np.mean(power, axis=1) + 1e-9
    flatness  = geom_mean / arith_mean

    # ZCR
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    return [
        centroid.mean(), bandwidth.mean(), rolloff.mean(), flatness.mean(),
        zcr.mean(), zcr.std(), rms.mean(), rms.std()
    ]


def extract_formant_proxies(signal, sr, n_formants=3):
    """
    LPC-based formant estimation (proxy — not as precise as PRAAT/FORANA
    but captures the same envelope peaks).
    Returns flattened [F1_mean, F1_std, F2_mean, F2_std, F3_mean, F3_std].
    """
    frames, frame_len, _ = _framing(signal, sr)
    lpc_order = 2 + sr // 1000   # rule of thumb

    all_formants = []
    for frame in frames:
        # LPC via autocorrelation (Levinson-Durbin)
        try:
            r = np.correlate(frame, frame, mode='full')
            r = r[len(r)//2: len(r)//2 + lpc_order + 1]
            R = np.array([[r[abs(i-j)] for j in range(lpc_order)]
                          for i in range(lpc_order)])
            if np.linalg.matrix_rank(R) < lpc_order:
                continue
            a = np.linalg.solve(R, -r[1:lpc_order+1])
            a = np.concatenate([[1], a])

            # roots of LPC polynomial → formants
            roots = np.roots(a)
            roots = roots[np.imag(roots) > 0]   # upper half-plane
            angles = np.arctan2(np.imag(roots), np.real(roots))
            freqs_hz = angles * sr / (2 * np.pi)
            freqs_hz = np.sort(freqs_hz[freqs_hz > 90])
            if len(freqs_hz) >= n_formants:
                all_formants.append(freqs_hz[:n_formants])
        except Exception:
            continue

    if not all_formants:
        return [0.0] * (n_formants * 2)

    arr = np.array(all_formants)   # (frames, n_formants)
    result = []
    for k in range(n_formants):
        result += [arr[:, k].mean(), arr[:, k].std()]
    return result


def extract_all_features(wav_path: str) -> np.ndarray:
    """
    Master feature extraction. Returns a 1-D float32 numpy array.
    Feature vector layout:
        mfcc        26
        pitch        5
        jitter_sh    2
        spectral     8
        formants     6
        ─────────────
        TOTAL       47
    """
    signal, sr = load_wav(wav_path)
    signal = trim_silence(signal)

    if len(signal) < sr * 0.1:   # shorter than 100 ms — skip
        return np.zeros(47, dtype=np.float32)

    mfcc     = extract_mfcc(signal, sr)
    pitch    = extract_pitch_features(signal, sr)
    jitter   = extract_jitter_shimmer(signal, sr)
    spectral = extract_spectral_features(signal, sr)
    formants = extract_formant_proxies(signal, sr)

    features = np.concatenate([mfcc, pitch, jitter, spectral, formants])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)


FEATURE_NAMES = (
    [f"MFCC_{i}_mean" for i in range(13)] +
    [f"MFCC_{i}_std"  for i in range(13)] +
    ["F0_mean", "F0_std", "F0_min", "F0_max", "Voiced_ratio"] +
    ["Jitter_%", "Shimmer_%"] +
    ["Spectral_centroid", "Spectral_bandwidth", "Spectral_rolloff",
     "Spectral_flatness", "ZCR_mean", "ZCR_std", "RMS_mean", "RMS_std"] +
    ["F1_mean", "F1_std", "F2_mean", "F2_std", "F3_mean", "F3_std"]
)


# ═════════════════════════════════════════════
#  3.  CLASSIFIER INFERENCE
# ═════════════════════════════════════════════

def load_sli_model():
    if not os.path.exists(SLI_MODEL_PATH):
        raise FileNotFoundError(
            f"SLI model not found at {SLI_MODEL_PATH}. "
            "Please run train_sli_model.py first."
        )
    clf    = joblib.load(SLI_MODEL_PATH)
    scaler = joblib.load(SLI_SCALER_PATH)
    with open(SLI_META_PATH) as f:
        meta = json.load(f)
    return clf, scaler, meta


def predict_sli(wav_path: str):
    """
    Run full inference on one WAV file.
    Returns dict:
        label       : "Healthy" | "SLI"
        severity    : None | "Mild" | "Moderate" | "Severe"
        confidence  : float  [0, 1]
        probabilities: dict  {class_name: prob}
        features    : np.ndarray
    """
    clf, scaler, meta = load_sli_model()
    features = extract_all_features(wav_path)
    X = scaler.transform(features.reshape(1, -1))

    classes    = meta["classes"]          # e.g. ["healthy","sli_mild",...]
    probs      = clf.predict_proba(X)[0]
    pred_idx   = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict  = {c: float(p) for c, p in zip(classes, probs)}

    if pred_class == "healthy":
        label, severity = "Healthy", None
    elif "mild" in pred_class:
        label, severity = "SLI", "Mild"
    elif "moderate" in pred_class:
        label, severity = "SLI", "Moderate"
    elif "severe" in pred_class:
        label, severity = "SLI", "Severe"
    else:
        label, severity = "SLI", "Unknown"

    return {
        "label":         label,
        "severity":      severity,
        "confidence":    confidence,
        "probabilities": prob_dict,
        "features":      features,
    }


# ═════════════════════════════════════════════
#  4.  TRANSCRIPTION  (Whisper – optional)
# ═════════════════════════════════════════════

def transcribe_audio(wav_path: str, language: str = "cs") -> str:
    """
    Transcribe WAV using OpenAI Whisper.
    Falls back to a helpful message if Whisper is not installed.
    language='cs' for Czech (LANNA database).  Change to 'en' if needed.
    """
    try:
        import whisper as _whisper
        model = _whisper.load_model("base")
        result = model.transcribe(wav_path, language=language)
        return result["text"].strip()
    except ImportError:
        return (
            "[Whisper not installed. "
            "Run:  pip install openai-whisper  to enable transcription.]"
        )
    except Exception as e:
        return f"[Transcription failed: {e}]"


# ═════════════════════════════════════════════
#  5.  VISUALISATIONS  (return matplotlib figs)
# ═════════════════════════════════════════════

def plot_waveform(signal: np.ndarray, sr: int, title: str = "Waveform"):
    fig, ax = plt.subplots(figsize=(8, 2))
    t = np.linspace(0, len(signal) / sr, len(signal))
    ax.plot(t, signal, color="#2563eb", linewidth=0.5, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_facecolor("#f8fafc")
    fig.tight_layout()
    return fig


def plot_spectrogram(signal: np.ndarray, sr: int, title: str = "Spectrogram"):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.specgram(signal, Fs=sr, cmap="viridis", NFFT=512, noverlap=256)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 13):
    signal_pe = _preemphasis(signal)
    frames, _, _ = _framing(signal_pe, sr)
    n_fft, n_mels = 512, 26
    mag = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

    fmin, fmax = 0, sr / 2
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = 700 * (10 ** (mel_pts / 2595) - 1)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bins[m-1], bins[m], bins[m+1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    fb = np.dot(mag, fbank.T)
    fb = np.where(fb == 0, np.finfo(float).eps, fb)
    fb = 20 * np.log10(fb)

    mfcc = np.zeros((len(frames), n_mfcc))
    for n in range(n_mfcc):
        mfcc[:, n] = np.sum(
            fb * np.cos(np.pi * n / n_mels * (np.arange(n_mels) + 0.5)),
            axis=1
        )

    fig, ax = plt.subplots(figsize=(8, 3))
    img = ax.imshow(mfcc.T, aspect="auto", origin="lower",
                    cmap="magma", interpolation="nearest")
    ax.set_xlabel("Frame")
    ax.set_ylabel("MFCC Coefficient")
    ax.set_title("MFCCs")
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def plot_feature_radar(features: np.ndarray):
    """
    Radar chart of the 8 most interpretable per-file features.
    Values are normalised to [0,1] for display.
    """
    labels = ["F0 Mean", "F0 Std", "Voiced\nRatio", "Jitter",
              "Shimmer", "Spectral\nCentroid", "ZCR", "RMS"]

    mfcc_len  = 26
    pitch_off = mfcc_len
    jitter_off = pitch_off + 5
    spec_off   = jitter_off + 2

    vals = np.array([
        features[pitch_off],          # F0 mean
        features[pitch_off + 1],      # F0 std
        features[pitch_off + 4],      # voiced ratio
        features[jitter_off],         # jitter
        features[jitter_off + 1],     # shimmer
        features[spec_off],           # spectral centroid
        features[spec_off + 4],       # ZCR mean
        features[spec_off + 6],       # RMS mean
    ])

    # normalise to [0,1] using typical ranges
    ranges = np.array([500, 200, 1, 10, 30, 4000, 0.5, 0.3])
    vals = np.clip(vals / (ranges + 1e-9), 0, 1)

    N   = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals_plot = list(vals) + [vals[0]]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, vals_plot, color="#2563eb", linewidth=2)
    ax.fill(angles, vals_plot, color="#2563eb", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=6)
    ax.set_title("Feature Radar", pad=14)
    fig.tight_layout()
    return fig


def plot_probability_bar(probabilities: dict):
    """Horizontal bar chart of class probabilities."""
    classes = list(probabilities.keys())
    probs   = [probabilities[c] for c in classes]
    colors  = ["#16a34a" if c == "healthy" else
               "#f59e0b" if "mild" in c else
               "#ef4444" if "moderate" in c else
               "#7c3aed" for c in classes]

    fig, ax = plt.subplots(figsize=(6, max(2, len(classes) * 0.7)))
    bars = ax.barh(classes, probs, color=colors, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("SLI Classification Probabilities")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════
#  6.  BUILD FINDINGS TEXT  (for RAG indexing)
# ═════════════════════════════════════════════

def build_audio_findings_text(wav_path: str, prediction: dict,
                               transcription: str) -> str:
    fname    = os.path.basename(wav_path)
    label    = prediction["label"]
    severity = prediction["severity"] or "N/A"
    conf     = prediction["confidence"] * 100
    probs    = prediction["probabilities"]
    features = prediction["features"]

    prob_lines = "\n".join(
        f"  {cls}: {p*100:.1f}%" for cls, p in probs.items()
    )

    feat_summary = (
        f"  F0 mean        : {features[26]:.1f} Hz\n"
        f"  F0 std         : {features[27]:.1f} Hz\n"
        f"  Voiced ratio   : {features[30]:.2f}\n"
        f"  Jitter         : {features[31]:.2f}%\n"
        f"  Shimmer        : {features[32]:.2f}%\n"
        f"  Spectral centrd: {features[33]:.1f} Hz\n"
        f"  ZCR mean       : {features[37]:.4f}\n"
        f"  RMS mean       : {features[39]:.4f}\n"
        f"  F1 mean        : {features[41]:.1f} Hz\n"
        f"  F2 mean        : {features[43]:.1f} Hz\n"
        f"  F3 mean        : {features[45]:.1f} Hz\n"
    )

    return f"""SLI AUDIO ANALYSIS REPORT
==========================
File: {fname}

CLASSIFICATION RESULT:
  Prediction : {label}
  Severity   : {severity}
  Confidence : {conf:.1f}%

CLASS PROBABILITIES:
{prob_lines}

ACOUSTIC FEATURES (summary):
{feat_summary}

SPEECH TRANSCRIPTION:
{transcription}

NOTE: This analysis is generated by an automated machine-learning system
trained on the LANNA Czech children's speech database (CTU Prague / Motol
University Hospital).  Results should be reviewed by a qualified speech
therapist.  This tool does NOT replace clinical assessment.
"""