"""
sli_audio_functions.py  -- LANNA SLI Pipeline v16
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BINARY FEATURES  (305-dim per utterance):
  MFCC  26 coeffs × 4 stats (mean/std/Δmean/Δstd) = 104
  PNCC  26 coeffs × 4 stats                        = 104
  CQCC  20 bins   × 4 stats                        =  80
  Spectral (centroid/spread/skew/kurt/flux/         =   9
            rolloff/HNR_spec/jitter/shimmer)
  Prosodic (voiced_ratio/energy_mean/std/           =   8
            F0_mean/std/range/ZCR/HNR)
  ─────────────────────────────────────────────────
  TOTAL                                             = 305

SEVERITY FEATURES  (21-dim per speaker, inference):
  Identical to training: per-task p(SLI) from binary model → 21-dim profile.
  This ensures training/inference feature alignment.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, json, tempfile, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
import joblib

warnings.filterwarnings("ignore")

# ─── Model paths ────────────────────────────────────────────────────────────
SLI_MODEL_PATH        = "models/sli_binary_clf.pkl"
SLI_SCALER_PATH       = "models/sli_binary_scaler.pkl"
SLI_META_PATH         = "models/sli_meta.json"
SLI_SEV_MODEL_PATH    = "models/sli_severity_clf.pkl"
SLI_SEV_SCALER_PATH   = "models/sli_severity_scaler.pkl"
SLI_BINARY_CLF_PATH   = SLI_MODEL_PATH
SLI_BINARY_SCL_PATH   = SLI_SCALER_PATH
SLI_SEVERITY_CLF_PATH = SLI_SEV_MODEL_PATH
SLI_SEVERITY_SCL_PATH = SLI_SEV_SCALER_PATH

# ─── Task constants ──────────────────────────────────────────────────────────
TASK_ORDER = ["SAMHOL", "SOUHL", "1SL", "2SL", "3SL", "4SL", "VSL"]
TASK_LABELS = {
    "SAMHOL": "Vowels",        "SOUHL": "Consonants",
    "1SL":    "Syllables",     "2SL":   "Words (2-syl)",
    "3SL":    "Words (3-syl)", "4SL":   "Words (4-syl)",
    "VSL":    "Sentences",
}
TASK_COMPLEXITY      = {t: i for i, t in enumerate(TASK_ORDER)}
EASY_TASKS           = ["SAMHOL", "SOUHL"]
MEDIUM_TASKS         = ["1SL", "2SL"]
HARD_TASKS           = ["3SL", "4SL", "VSL"]
ALL_CLASSES          = ["healthy", "mild", "moderate", "severe"]
BINARY_SLI_THRESHOLD = 0.5

# ─── Feature dimensions ──────────────────────────────────────────────────────
N_MFCC       = 26
N_PNCC       = 26
N_CQT_BINS   = 20
N_SPECTRAL   = 9
N_PROSODIC   = 8

N_MFCC_FEATS = N_MFCC     * 4   # 104
N_PNCC_FEATS = N_PNCC     * 4   # 104
N_CQCC_FEATS = N_CQT_BINS * 4   # 80
N_SPEC_FEATS = N_SPECTRAL        # 9
N_PROS_FEATS = N_PROSODIC        # 8

N_UTT_FEATS  = N_MFCC_FEATS + N_PNCC_FEATS + N_CQCC_FEATS + N_SPEC_FEATS + N_PROS_FEATS  # 305
N_FEATURES   = N_UTT_FEATS

# Severity: 21-dim p(SLI) profile (training and inference are identical)
N_SEV_GROUP_FEATS = 10   # legacy compat
N_SEV_FEATS       = 21

FEATURE_NAMES = (
    [f"mfcc{i+1}_mean"  for i in range(N_MFCC)] +
    [f"mfcc{i+1}_std"   for i in range(N_MFCC)] +
    [f"mfcc{i+1}_dmean" for i in range(N_MFCC)] +
    [f"mfcc{i+1}_dstd"  for i in range(N_MFCC)] +
    [f"pncc{i+1}_mean"  for i in range(N_PNCC)] +
    [f"pncc{i+1}_std"   for i in range(N_PNCC)] +
    [f"pncc{i+1}_dmean" for i in range(N_PNCC)] +
    [f"pncc{i+1}_dstd"  for i in range(N_PNCC)] +
    [f"cqcc{i+1}_mean"  for i in range(N_CQT_BINS)] +
    [f"cqcc{i+1}_std"   for i in range(N_CQT_BINS)] +
    [f"cqcc{i+1}_dmean" for i in range(N_CQT_BINS)] +
    [f"cqcc{i+1}_dstd"  for i in range(N_CQT_BINS)] +
    ["spec_centroid","spec_spread","spec_skewness","spec_kurtosis",
     "spec_flux","spec_rolloff","hnr_spec","jitter","shimmer"] +
    ["voiced_ratio","energy_mean","energy_std",
     "f0_mean","f0_std","f0_range","zcr_mean","hnr"]
)

TASK_FEAT_DIM   = 1
_PER_TASK_NAMES = ["task_psli"]

# ════════════════════════════════════════════════════════════════════════════
#  WAV I/O
# ════════════════════════════════════════════════════════════════════════════

def load_wav(path, target_sr=16000):
    sr, data = wavfile.read(path)
    if   data.dtype == np.int16:  data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:  data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:  data = (data.astype(np.float32) - 128.0) / 128.0
    else:                         data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        n    = int(len(data) * target_sr / sr)
        data = np.interp(np.linspace(0, len(data)-1, n), np.arange(len(data)), data)
        sr   = target_sr
    pk = np.abs(data).max()
    if pk > 1e-6: data /= pk
    return data.astype(np.float32), sr


def trim_silence(sig, sr, threshold_db=-38, frame_ms=20):
    fl    = int(sr * frame_ms / 1000)
    steps = list(range(0, len(sig)-fl, fl//2))
    if not steps: return sig
    rms_db = [20*np.log10(max(np.sqrt(np.mean(sig[i:i+fl]**2)), 1e-9)) for i in steps]
    thr    = max(rms_db) + threshold_db
    vi     = [i for i, d in enumerate(rms_db) if d > thr]
    if not vi: return sig
    return sig[max(0, vi[0]*fl//2): min(len(sig), (vi[-1]+1)*fl//2+fl)]

# ════════════════════════════════════════════════════════════════════════════
#  DSP HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _frames(sig, sr, frame_ms=25, step_ms=10):
    fl = int(sr*frame_ms/1000); fs = int(sr*step_ms/1000)
    if len(sig) < fl: sig = np.pad(sig, (0, fl-len(sig)))
    n   = 1 + (len(sig)-fl)//fs
    out = np.zeros((n, fl), np.float32)
    for i in range(n): out[i] = sig[i*fs: i*fs+fl]
    return out, fl, fs


def _mel_filterbank(sr, n_fft, n_mels=52, fmin=80.0, fmax=None):
    fmax = fmax or min(sr/2.0, 8000.0)
    mmin = 2595*np.log10(1+fmin/700); mmax = 2595*np.log10(1+fmax/700)
    mp   = np.linspace(mmin, mmax, n_mels+2)
    hp   = 700*(10**(mp/2595)-1)
    bns  = np.clip(np.floor((n_fft+1)*hp/sr).astype(int), 0, n_fft//2)
    fb   = np.zeros((n_mels, n_fft//2+1))
    for m in range(1, n_mels+1):
        lo, mid, hi = bns[m-1], bns[m], bns[m+1]
        for k in range(lo, mid):
            if mid > lo: fb[m-1,k] = (k-lo)/(mid-lo)
        for k in range(mid, hi):
            if hi > mid: fb[m-1,k] = (hi-k)/(hi-mid)
    return fb


def _delta(coef):
    T, D  = coef.shape; delta = np.zeros_like(coef)
    if T < 3: return delta
    for t in range(1, T-1): delta[t] = (coef[t+1]-coef[t-1])/2.0
    delta[0] = delta[1]; delta[-1] = delta[-2]
    return delta

# ════════════════════════════════════════════════════════════════════════════
#  MFCC  (26 coefficients)
# ════════════════════════════════════════════════════════════════════════════

def _compute_mfcc(sig, sr, n_mfcc=None, n_fft=512, n_mels=None):
    if n_mfcc is None: n_mfcc = N_MFCC
    if n_mels is None: n_mels = max(52, n_mfcc*2)
    fr, fl, _ = _frames(sig, sr)
    if len(fr) == 0: return np.zeros((0, n_mfcc), np.float32)
    win = np.hamming(fl).astype(np.float32)
    mag = np.abs(np.fft.rfft(fr*win, n=n_fft))**2
    fb  = _mel_filterbank(sr, n_fft, n_mels)
    fbe = np.dot(mag, fb.T)
    fbe = np.where(fbe < 1e-9, 1e-9, fbe); fbe = 20*np.log10(fbe)
    mfcc = np.zeros((len(fr), n_mfcc), np.float32)
    for n in range(n_mfcc):
        mfcc[:,n] = (fbe * np.cos(np.pi*(n+1)/n_mels*(np.arange(n_mels)+0.5))).sum(1)
    return mfcc

# ════════════════════════════════════════════════════════════════════════════
#  PNCC  (26 coefficients, gammatone-inspired)
# ════════════════════════════════════════════════════════════════════════════

def _gammatone_filterbank(sr, n_fft, n_filters=52, fmin=80.0, fmax=None):
    fmax    = fmax or min(sr/2.0, 8000.0)
    erb_min = 9.265*np.log(1+fmin/(24.7*9.265))
    erb_max = 9.265*np.log(1+fmax/(24.7*9.265))
    erb_pts = np.linspace(erb_min, erb_max, n_filters+2)
    cf      = 24.7*9.265*(np.exp(erb_pts/9.265)-1)
    freqs   = np.linspace(0, sr/2.0, n_fft//2+1)
    fb      = np.zeros((n_filters, n_fft//2+1), np.float32)
    for m in range(n_filters):
        fc = cf[m+1]; bw = 24.7*(4.37*fc/1000+1)
        fb[m] = np.exp(-0.5*((freqs-fc)/(bw*0.8))**2)
    fb = fb / (fb.sum(axis=1, keepdims=True) + 1e-9)
    return fb


def _compute_pncc(sig, sr, n_pncc=None, n_fft=512, n_filters=None, power=1.0/15.0):
    if n_pncc    is None: n_pncc    = N_PNCC
    if n_filters is None: n_filters = max(52, n_pncc*2)
    fr, fl, _ = _frames(sig, sr)
    if len(fr) == 0: return np.zeros((0, n_pncc), np.float32)
    win      = np.hamming(fl).astype(np.float32)
    mag      = np.abs(np.fft.rfft(fr*win, n=n_fft))**2
    fb       = _gammatone_filterbank(sr, n_fft, n_filters)
    fbe      = np.dot(mag, fb.T)
    mp       = np.median(fbe, axis=0)
    fbe_norm = np.maximum(fbe - mp[None,:], 0.0)
    fbe_norm = np.where(fbe_norm < 1e-9, 1e-9, fbe_norm)
    fbe_comp = fbe_norm**power
    pncc     = np.zeros((len(fr), n_pncc), np.float32)
    for n in range(n_pncc):
        pncc[:,n] = (fbe_comp * np.cos(np.pi*(n+1)/n_filters*(np.arange(n_filters)+0.5))).sum(1)
    return pncc

# ════════════════════════════════════════════════════════════════════════════
#  CQCC  (20 bins)
# ════════════════════════════════════════════════════════════════════════════

def _compute_cqcc(sig, sr, n_bins=None, fmin=80.0, bins_per_octave=12, n_fft=512):
    if n_bins is None: n_bins = N_CQT_BINS
    fr, fl, _ = _frames(sig, sr)
    if len(fr) == 0: return np.zeros((0, n_bins), np.float32)
    win   = np.hamming(fl).astype(np.float32)
    mag   = np.abs(np.fft.rfft(fr*win, n=n_fft))**2
    freqs = np.linspace(0, sr/2.0, n_fft//2+1)
    n_oct = int(np.floor(np.log2((sr/2.0)/fmin)))
    n_cqf = min(n_oct*bins_per_octave, 96)
    cf    = fmin*2.0**(np.arange(n_cqf)/bins_per_octave)
    cf    = cf[cf < sr/2.0*0.95]
    if len(cf) == 0: return np.zeros((len(fr), n_bins), np.float32)
    Q  = 1.0/(2**(1.0/bins_per_octave)-1)
    fb = np.zeros((len(cf), n_fft//2+1), np.float32)
    for m, fc in enumerate(cf):
        bw = fc/Q; fb[m] = np.exp(-0.5*((freqs-fc)/(bw*0.5))**2)
    fb      = fb/(fb.sum(axis=1, keepdims=True)+1e-9)
    cq_spec = np.dot(mag, fb.T)
    cq_spec = np.where(cq_spec < 1e-9, 1e-9, cq_spec)
    log_cq  = np.log(cq_spec)
    n_keep  = min(n_bins, len(cf))
    out     = np.zeros((len(fr), n_bins), np.float32)
    for n in range(n_keep):
        out[:,n] = (log_cq * np.cos(np.pi*n/len(cf)*(np.arange(len(cf))+0.5))).sum(1)
    return out

# ════════════════════════════════════════════════════════════════════════════
#  SPECTRAL FEATURES  (9 dims)
# ════════════════════════════════════════════════════════════════════════════

def _compute_spectral(sig, sr, fr, fl, n_fft=512):
    win    = np.hamming(fl).astype(np.float32)
    mag    = np.abs(np.fft.rfft(fr*win, n=n_fft))
    power  = mag**2
    freqs  = np.linspace(0, sr/2.0, n_fft//2+1)
    psum   = power.sum(axis=1, keepdims=True) + 1e-12
    pnorm  = power / psum

    centroid = (pnorm * freqs[None,:]).sum(axis=1)
    diff2    = (freqs[None,:] - centroid[:,None])**2
    spread   = np.sqrt((pnorm*diff2).sum(axis=1))
    diff3    = (freqs[None,:] - centroid[:,None])**3
    skewness = (pnorm*diff3).sum(axis=1) / (spread**3 + 1e-9)
    diff4    = (freqs[None,:] - centroid[:,None])**4
    kurtosis = (pnorm*diff4).sum(axis=1) / (spread**4 + 1e-9)

    flux = np.zeros(len(fr), np.float32)
    if len(fr) > 1:
        flux[1:] = np.sqrt(((mag[1:]-mag[:-1])**2).sum(axis=1))
        flux[0]  = flux[1]

    cum     = np.cumsum(power, axis=1)
    thr     = 0.85 * cum[:,-1]
    rolloff = np.zeros(len(fr), np.float32)
    for i in range(len(fr)):
        idx = np.searchsorted(cum[i], thr[i])
        rolloff[i] = freqs[min(idx, len(freqs)-1)]

    lmin = max(1, int(sr/600)); lmax = min(int(sr/60), fl-1)
    win2 = np.hamming(fl).astype(np.float32)
    hnr_vals = []; f0_per = []
    for frm in fr:
        wf = frm*win2; c = np.correlate(wf, wf, "full")[fl-1:]
        if c[0]<1e-9 or lmax<=lmin or lmax>=len(c):
            hnr_vals.append(0.0); f0_per.append(0.0); continue
        pk = np.argmax(c[lmin:lmax])+lmin; r = c[pk]/c[0]
        hnr_vals.append(float(10*np.log10(max(r,1e-9)/max(1.0-r,1e-9))))
        f0_per.append(float(sr/pk) if r > 0.28 else 0.0)
    hnr_spec = float(np.mean(hnr_vals))

    voiced_f0  = [v for v in f0_per if v > 0]
    rms_frames = np.sqrt(np.mean(fr**2, axis=1))
    jitter  = (float(np.mean(np.abs(np.diff([sr/f for f in voiced_f0]))) /
               (np.mean([sr/f for f in voiced_f0])+1e-9))
               if len(voiced_f0) > 1 else 0.0)
    shimmer = (float(np.mean(np.abs(np.diff(rms_frames))) /
               (np.mean(rms_frames)+1e-9))
               if len(rms_frames) > 1 else 0.0)

    return np.array([
        float(centroid.mean()), float(spread.mean()),
        float(skewness.mean()), float(kurtosis.mean()),
        float(flux.mean()),     float(rolloff.mean()),
        hnr_spec, jitter, shimmer,
    ], np.float32)

# ════════════════════════════════════════════════════════════════════════════
#  PROSODIC FEATURES  (8 dims)
# ════════════════════════════════════════════════════════════════════════════

def _compute_prosodic(sig, sr, fr, fl):
    win  = np.hamming(fl).astype(np.float32)
    lmin = max(1, int(sr/600)); lmax = min(int(sr/60), fl-1)
    voiced = 0; f0_vals = []
    for frm in fr:
        wf = frm*win; c = np.correlate(wf, wf, "full")[fl-1:]
        if c[0]<1e-9 or lmax<=lmin or lmax>=len(c): continue
        pk = np.argmax(c[lmin:lmax])+lmin
        if c[pk]/c[0] > 0.28: voiced += 1; f0_vals.append(float(sr/pk))
    voiced_ratio = float(voiced/max(len(fr), 1))
    energy       = np.sqrt(np.mean(fr**2, axis=1))
    energy_mean  = float(energy.mean()); energy_std = float(energy.std())
    f0_mean  = float(np.mean(f0_vals))  if f0_vals else 0.0
    f0_std   = float(np.std(f0_vals))   if f0_vals else 0.0
    f0_range = float(np.ptp(f0_vals))   if f0_vals else 0.0
    zcr      = np.mean(np.abs(np.diff(np.sign(fr), axis=1)), axis=1)/2.0
    zcr_mean = float(zcr.mean())
    hnr = 0.0
    if f0_vals:
        f0e = np.median(f0_vals); period = int(sr/f0e) if f0e > 0 else 0
        if period > 0 and period*2 < len(sig):
            h = np.correlate(sig[:period*2], sig[:period], "valid")
            nv = max(np.var(sig)-(np.var(h[:period]) if len(h)>=period else 0), 1e-9)
            hnr = float(10*np.log10(np.var(sig)/nv+1e-9))
    return np.array([voiced_ratio,energy_mean,energy_std,
                     f0_mean,f0_std,f0_range,zcr_mean,hnr], np.float32)

# ════════════════════════════════════════════════════════════════════════════
#  COMBINED UTTERANCE FEATURES  (305-dim)
# ════════════════════════════════════════════════════════════════════════════

def extract_utterance_features(wav_path):
    """305-dim: MFCC(104)+PNCC(104)+CQCC(80)+Spectral(9)+Prosodic(8)"""
    ZERO        = np.zeros(N_UTT_FEATS, np.float32)
    actual_path = wav_path; tmp_path = None
    if hasattr(wav_path, "read"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(wav_path.read()); tmp.close()
        actual_path = tmp.name; tmp_path = tmp.name
        if hasattr(wav_path, "seek"): wav_path.seek(0)
    try:
        sig, sr = load_wav(actual_path)
        sig     = trim_silence(sig, sr)
        if len(sig) < sr*0.04: return ZERO
        fr, fl, _ = _frames(sig, sr)
        if len(fr) == 0: return ZERO

        mfcc      = _compute_mfcc(sig, sr, n_mfcc=N_MFCC)
        d_mfcc    = _delta(mfcc)
        mfcc_feat = np.concatenate([mfcc.mean(0),mfcc.std(0),d_mfcc.mean(0),d_mfcc.std(0)])

        pncc      = _compute_pncc(sig, sr, n_pncc=N_PNCC)
        d_pncc    = _delta(pncc)
        pncc_feat = np.concatenate([pncc.mean(0),pncc.std(0),d_pncc.mean(0),d_pncc.std(0)])

        cqcc      = _compute_cqcc(sig, sr, n_bins=N_CQT_BINS)
        d_cqcc    = _delta(cqcc)
        cqcc_feat = np.concatenate([cqcc.mean(0),cqcc.std(0),d_cqcc.mean(0),d_cqcc.std(0)])

        spec_feat = _compute_spectral(sig, sr, fr, fl)
        pros_feat = _compute_prosodic(sig, sr, fr, fl)

        feat = np.concatenate([mfcc_feat,pncc_feat,cqcc_feat,
                               spec_feat,pros_feat]).astype(np.float32)
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return ZERO
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass

# ════════════════════════════════════════════════════════════════════════════
#  SEVERITY FEATURE VECTOR  (21-dim, SAME LOGIC AS TRAINING)
# ════════════════════════════════════════════════════════════════════════════

def _task_psli_to_21d(task_psli):
    """
    Convert per-task p(SLI) dict → 21-dim feature vector.
    MUST match the _psli_to_21d() function in train_sli_model.py exactly.
    """
    def gv(tasks):
        return [task_psli[t] for t in tasks if task_psli.get(t) is not None]

    tv    = [task_psli.get(t) or 0.0 for t in TASK_ORDER]  # 7
    ev, mv, hv = gv(EASY_TASKS), gv(MEDIUM_TASKS), gv(HARD_TASKS)
    all_v = [v for v in tv if v > 0]

    sm = lambda v: float(np.mean(v)) if v else 0.0
    ss = lambda v: float(np.std(v))  if len(v) > 1 else 0.0
    sr = lambda v: float(np.ptp(v))  if len(v) > 1 else 0.0
    ca = lambda v: float(sum(x >= 0.5 for x in v))

    feat = (
        tv +
        [sm(ev), sm(mv), sm(hv), sm(all_v)] +
        [ss(ev), ss(mv), ss(hv)] +
        [ca(ev), ca(mv), ca(hv)] +
        [sr(ev), sr(mv), sr(hv), ss(all_v)]
    )
    return np.array(feat, np.float32)

# ════════════════════════════════════════════════════════════════════════════
#  INFERENCE UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _model_ready():
    return os.path.exists(SLI_MODEL_PATH) and os.path.exists(SLI_SCALER_PATH)

def _severity_model_ready():
    return os.path.exists(SLI_SEV_MODEL_PATH) and os.path.exists(SLI_SEV_SCALER_PATH)


def classify_utterances(wav_paths):
    """Return list of p(SLI) for each WAV."""
    if not _model_ready(): return []
    try:
        clf     = joblib.load(SLI_MODEL_PATH)
        scl     = joblib.load(SLI_SCALER_PATH)
        meta    = json.load(open(SLI_META_PATH))
        classes = meta.get("binary_classes", ["healthy","sli"])
        sli_idx = classes.index("sli") if "sli" in classes else 1
        results = []
        for wp in wav_paths:
            feat = extract_utterance_features(wp)
            if feat.sum() == 0: continue
            Xs   = scl.transform(feat.reshape(1,-1))
            prob = clf.predict_proba(Xs)[0]
            results.append(float(prob[sli_idx]))
        return results
    except Exception: return []


def _collect_task_wavs(speaker_dir, task_key):
    try:
        kc = task_key.upper().replace("_","")
        for entry in sorted(os.listdir(speaker_dir)):
            if os.path.isdir(os.path.join(speaker_dir,entry)):
                cl = entry.upper().replace("_","").replace(" ","")
                if kc in cl:
                    tp = os.path.join(speaker_dir,entry)
                    return sorted(os.path.join(tp,f) for f in os.listdir(tp)
                                  if f.lower().endswith(".wav"))
    except Exception: pass
    return []


def compute_task_psli(task_wav_map):
    """Mean p(SLI) per task."""
    task_psli = {}
    for task in TASK_ORDER:
        wavs = task_wav_map.get(task, [])
        if not wavs: task_psli[task] = None; continue
        probs = classify_utterances(wavs)
        task_psli[task] = float(np.mean(probs)) if probs else None
    return task_psli


def determine_severity_rule(task_psli, threshold=BINARY_SLI_THRESHOLD):
    """Rule-based severity (used as fallback if trained severity model absent)."""
    def gm(tasks):
        v = [task_psli.get(t) for t in tasks if task_psli.get(t) is not None]
        return float(np.mean(v)) if v else None
    ep = gm(EASY_TASKS); mp = gm(MEDIUM_TASKS); hp = gm(HARD_TASKS)
    ap = [task_psli.get(t) for t in TASK_ORDER if task_psli.get(t) is not None]
    ov = float(np.mean(ap)) if ap else 0.0
    if ep is not None and ep >= threshold: return "severe",   ov
    if mp is not None and mp >= threshold: return "moderate", ov
    if hp is not None and hp >= threshold: return "mild",     ov
    return "healthy", ov

determine_severity = determine_severity_rule   # compat alias


def get_task_quality_scores_from_wavs(task_wav_map):
    tp = compute_task_psli(task_wav_map)
    return {t: ((1.0-p) if p is not None else None) for t,p in tp.items()}

def get_task_quality_scores(sv): return {t: None for t in TASK_ORDER}
def build_speaker_vector(sd):    return np.zeros(N_UTT_FEATS, np.float32)
def build_speaker_vector_from_uploaded(tm): return np.zeros(N_UTT_FEATS, np.float32)

# Legacy compat: old severity feature builder (no longer used for model calls)
def build_severity_feature_vector(task_wav_map):
    tp = compute_task_psli(task_wav_map)
    return _task_psli_to_21d(tp)

# ════════════════════════════════════════════════════════════════════════════
#  MAIN INFERENCE
# ════════════════════════════════════════════════════════════════════════════

def predict_from_task_map(task_wav_map):
    """
    Full inference:
      1. Binary model → healthy / SLI
      2. If SLI → severity model (21-dim p(SLI) profile) → mild/moderate/severe
    """
    if not _model_ready(): return _no_model_result()

    path_map = {}; tmps = []
    for task, files in task_wav_map.items():
        paths = []
        for f in files:
            if hasattr(f, "read"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp.write(f.read()); tmp.close()
                paths.append(tmp.name); tmps.append(tmp.name)
                if hasattr(f,"seek"): f.seek(0)
            else: paths.append(str(f))
        path_map[task] = paths

    task_psli                    = compute_task_psli(path_map)
    severity_rule, overall_psli  = determine_severity_rule(task_psli)
    task_scores                  = {t:((1.0-p) if p is not None else None)
                                    for t,p in task_psli.items()}
    label                        = "Healthy" if severity_rule == "healthy" else "SLI"
    binary_probs                 = {"healthy": 1.0-overall_psli, "sli": overall_psli}

    if label == "SLI":
        if _severity_model_ready():
            try:
                sev_clf  = joblib.load(SLI_SEV_MODEL_PATH)
                sev_scl  = joblib.load(SLI_SEV_SCALER_PATH)
                meta     = json.load(open(SLI_META_PATH))
                sev_cls  = meta.get("severity_classes", ["mild","moderate","severe"])

                # Build the SAME 21-dim p(SLI) profile used during training
                feat_sv  = _task_psli_to_21d(task_psli).reshape(1, -1)
                Xs_sv    = sev_scl.transform(feat_sv)
                sprob    = sev_clf.predict_proba(Xs_sv)[0]
                sev_probs = {c: float(p) for c,p in zip(sev_cls, sprob)}
                sev_label = sev_cls[int(np.argmax(sprob))].capitalize()
            except Exception:
                sev_label = severity_rule.capitalize()
                sev_probs = {c:(1.0 if c==severity_rule else 0.0)
                             for c in ["mild","moderate","severe"]}
        else:
            sev_label = severity_rule.capitalize()
            sev_probs = {c:(1.0 if c==severity_rule else 0.0)
                         for c in ["mild","moderate","severe"]}
    else:
        sev_label = None; sev_probs = None

    for tmp in tmps:
        try: os.unlink(tmp)
        except: pass

    profile = _build_profile(label, sev_label, overall_psli, task_scores, task_psli)
    return {
        "label":          label,
        "severity":       sev_label,
        "disorder_score": overall_psli,
        "confidence":     overall_psli if label == "SLI" else 1.0-overall_psli,
        "binary_probs":   binary_probs,
        "severity_probs": sev_probs,
        "task_scores":    task_scores,
        "task_psli":      task_psli,
        "features":       np.zeros(N_UTT_FEATS, np.float32),
        "profile":        profile,
    }


def _no_model_result():
    ts = {t: None for t in TASK_ORDER}
    return {"label":"No model","severity":None,"disorder_score":0.0,"confidence":0.0,
            "binary_probs":{"healthy":0.0,"sli":0.0},"severity_probs":None,
            "task_scores":ts,"features":np.zeros(N_UTT_FEATS),
            "profile":"No model trained. Run: python train_sli_model.py"}

def predict_single_wav(wav_path, assumed_task="VSL"):
    return predict_from_task_map({assumed_task: [wav_path]})

def predict_from_speaker_dir(speaker_dir):
    pm = {t: _collect_task_wavs(speaker_dir, t) for t in TASK_ORDER}
    return predict_from_task_map(pm)

# ════════════════════════════════════════════════════════════════════════════
#  CLINICAL PROFILE
# ════════════════════════════════════════════════════════════════════════════

def _build_profile(label, severity, disorder_score, task_scores, task_psli):
    lines = []
    if label == "No model":
        lines.append("No trained model. Run: python train_sli_model.py")
        return "\n".join(lines)
    if label == "Healthy":
        lines.append(
            f"Speech is within typical development across all task levels. "
            f"Utterance-level SLI probability is low (mean: {disorder_score*100:.0f}%). "
            "Articulation is consistent from consonants through sentences.")
    else:
        descs = {
            "Mild":
                f"Mild SLI detected (mean SLI probability: {disorder_score*100:.0f}%). "
                "Vowels, consonants, syllables and simple words are produced adequately. "
                "Difficulty appears at the level of 3–4 syllable words and sentences.",
            "Moderate":
                f"Moderate SLI detected (mean SLI probability: {disorder_score*100:.0f}%). "
                "Vowel and consonant production is adequate. "
                "Difficulty begins from syllable/2-syllable word level onwards.",
            "Severe":
                f"Severe SLI detected (mean SLI probability: {disorder_score*100:.0f}%). "
                "Even basic vowel and consonant production shows SLI patterns. "
                "All task levels are affected.",
        }
        lines.append(descs.get(severity or "",
                     f"SLI detected (probability: {disorder_score*100:.0f}%)."))

    lines.append("\nPer-task SLI probability (higher = more SLI-like):")
    for task in TASK_ORDER:
        p  = task_psli.get(task); tl = TASK_LABELS.get(task, task)
        if p is None: lines.append(f"  — {tl:<24}[not tested]"); continue
        bar  = "█"*int(p*10) + "░"*(10-int(p*10))
        flag = "✓" if p < 0.40 else ("△" if p < 0.60 else "✗")
        lines.append(f"  {flag} {tl:<24}[{bar}] p(SLI)={p*100:.0f}%")

    ap = [task_psli[t] for t in TASK_ORDER if task_psli.get(t) is not None]
    if len(ap) >= 3:
        ep = np.mean([task_psli[t] for t in EASY_TASKS   if task_psli.get(t) is not None] or [0])
        mp = np.mean([task_psli[t] for t in MEDIUM_TASKS if task_psli.get(t) is not None] or [0])
        hp = np.mean([task_psli[t] for t in HARD_TASKS   if task_psli.get(t) is not None] or [0])
        lines.append(f"\nComplexity breakdown:")
        lines.append(f"  Easy   (vowels/consonants):     p(SLI)={ep*100:.0f}%")
        lines.append(f"  Medium (syllables/2-syl words): p(SLI)={mp*100:.0f}%")
        lines.append(f"  Hard   (3-4 syl/sentences):     p(SLI)={hp*100:.0f}%")

    if label == "SLI":
        sev = (severity or "").lower()
        lines.append("\nRecommended therapy focus:")
        if sev == "severe":
            lines.append("  1. Basic phoneme production (m,b,p,t,d,k,a,o,u)")
            lines.append("  2. Auditory discrimination training")
            lines.append("  3. Phonological awareness at segmental level")
        elif sev == "moderate":
            lines.append("  1. CV syllable repetition (pa,ta,ka,ba,da,ga)")
            lines.append("  2. Minimal pair exercises (bat/pat, cap/cup)")
            lines.append("  3. 2-syllable word production and prosody")
        elif sev == "mild":
            lines.append("  1. 3-4 syllable word articulation and stress patterns")
            lines.append("  2. Sentence prosody and rhythm")
            lines.append("  3. Connected speech and narrative practice")

    lines.append("\n[DISCLAIMER: Automated screening — review by speech-language therapist required]")
    return "\n".join(lines)

# ════════════════════════════════════════════════════════════════════════════
#  TRANSCRIPTION & REPORT
# ════════════════════════════════════════════════════════════════════════════

def transcribe_audio(wav_path, language="cs"):
    try:
        import whisper as _w
        model  = _w.load_model("base")
        kwargs = {} if language == "auto" else {"language": language}
        return model.transcribe(str(wav_path), **kwargs)["text"].strip()
    except ImportError: return "[Whisper not installed]"
    except Exception as e: return f"[Transcription error: {e}]"


def build_audio_findings_text(source_name, result, transcription=""):
    label   = result["label"]; severity = result.get("severity") or "N/A"
    score   = result["disorder_score"]*100
    bp      = result.get("binary_probs",{}); sp = result.get("severity_probs") or {}
    ts      = result.get("task_scores",{}); profile = result.get("profile","")
    bp_str  = "\n".join(f"  {k.capitalize()}: {v*100:.1f}%" for k,v in bp.items())
    sp_str  = "\n".join(f"  {k.capitalize()}: {v*100:.1f}%" for k,v in sp.items()) if sp else "  N/A"
    ts_lines = [
        f"  {TASK_LABELS.get(t,t):<24}: "
        f"{'not tested' if ts.get(t) is None else f'p(healthy)={ts[t]*100:.0f}%'}"
        for t in TASK_ORDER
    ]
    return (f"SLI SPEECH ANALYSIS REPORT\n===========================\n"
            f"Source: {source_name}\nDiagnosis: {label}\nSeverity: {severity}\n"
            f"SLI Probability: {score:.1f}%\n\nClassification:\n{bp_str}\n"
            f"Severity:\n{sp_str}\n\nTask results:\n{chr(10).join(ts_lines)}\n\n"
            f"Clinical Profile:\n{profile}\nTranscription:\n"
            f"{transcription or '[Not requested]'}\n\n"
            f"DISCLAIMER: Automated ML screening. Must be reviewed by therapist.\n")

# ════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def plot_waveform(sig, sr, title="Waveform"):
    fig,ax=plt.subplots(figsize=(9,2))
    ax.plot(np.linspace(0,len(sig)/sr,len(sig)),sig,color="#2563eb",lw=0.5,alpha=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.set_title(title)
    ax.set_facecolor("#f8fafc"); fig.tight_layout(); return fig

def plot_spectrogram(sig, sr, title="Spectrogram"):
    fig,ax=plt.subplots(figsize=(9,3))
    ax.specgram(sig,Fs=sr,cmap="viridis",NFFT=512,noverlap=384)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq (Hz)"); ax.set_title(title)
    fig.tight_layout(); return fig

def plot_mfcc(sig, sr):
    mfcc=_compute_mfcc(sig,sr,n_mfcc=13)
    fig,ax=plt.subplots(figsize=(9,3))
    im=ax.imshow(mfcc.T,aspect="auto",origin="lower",cmap="magma")
    ax.set_xlabel("Frame"); ax.set_ylabel("MFCC"); ax.set_title("MFCCs")
    fig.colorbar(im,ax=ax); fig.tight_layout(); return fig

def plot_task_quality(task_scores, label="", severity=""):
    scores=[task_scores.get(t) for t in TASK_ORDER]
    labels_=[TASK_LABELS.get(t,t) for t in TASK_ORDER]
    values=[s if s is not None else 0.0 for s in scores]; absent=[s is None for s in scores]
    colors=[]
    for s,ab in zip(scores,absent):
        if ab:          colors.append("#e2e8f0")
        elif s>=0.65:   colors.append("#16a34a")
        elif s>=0.50:   colors.append("#84cc16")
        elif s>=0.35:   colors.append("#f59e0b")
        else:           colors.append("#ef4444")
    fig,ax=plt.subplots(figsize=(8,4.5))
    bars=ax.barh(labels_,values,color=colors,edgecolor="white",height=0.6)
    ax.set_xlim(0,1.15)
    ax.axvline(0.50,color="#f97316",ls="--",lw=2.0,alpha=0.9,label="Decision boundary")
    ax.axvline(0.65,color="#16a34a",ls="--",lw=1.2,alpha=0.7,label="Clearly healthy")
    for bar,s,ab in zip(bars,scores,absent):
        txt="—" if ab else f"{(1-s)*100:.0f}%SLI"
        ax.text(bar.get_width()+0.01,bar.get_y()+bar.get_height()/2,txt,va="center",fontsize=8)
    title="Per-Task SLI Classification  (bar = p(healthy))"
    if label: title+=f"  →  {label}"+(f" ({severity})" if severity else "")
    ax.set_title(title,fontweight="bold"); ax.set_xlabel("p(healthy)")
    ax.legend(fontsize=8,loc="lower right"); ax.invert_yaxis()
    fig.tight_layout(); return fig

def plot_complexity_profile(task_scores):
    present=[(t,task_scores[t]) for t in TASK_ORDER if task_scores.get(t) is not None]
    if len(present)<2: return None
    xs=list(range(len(present))); ys=[s for _,s in present]
    tlbls=[TASK_LABELS.get(t,t) for t,_ in present]
    colors=["#16a34a" if s>=0.65 else "#f59e0b" if s>=0.50 else "#ef4444" for s in ys]
    fig,ax=plt.subplots(figsize=(9,3.5))
    ax.fill_between(xs,ys,alpha=0.15,color="#3b82f6")
    ax.plot(xs,ys,"o-",color="#3b82f6",lw=2,ms=8,zorder=5)
    for x,y,c in zip(xs,ys,colors): ax.plot(x,y,"o",color=c,ms=10,zorder=6)
    ax.axhline(0.50,color="#f97316",ls="--",lw=2.0,alpha=0.9,label="Decision threshold")
    ax.set_xticks(xs); ax.set_xticklabels(tlbls,rotation=15,ha="right")
    ax.set_ylim(0,1.05); ax.set_ylabel("p(healthy)")
    ax.set_title("Task-Level Healthy Probability Across Complexity",fontweight="bold")
    ax.legend(fontsize=9); fig.tight_layout(); return fig

def plot_disorder_gauge(disorder_score, label, severity):
    fig,ax=plt.subplots(figsize=(8,2.2))
    zones=[(0,0.25,"#16a34a","Healthy"),(0.25,0.50,"#eab308","Mild"),
           (0.50,0.75,"#f97316","Moderate"),(0.75,1.00,"#dc2626","Severe")]
    for lo,hi,color,zlabel in zones:
        ax.barh(0,hi-lo,left=lo,height=0.35,color=color,alpha=0.30)
        ax.text((lo+hi)/2,-0.28,zlabel,ha="center",fontsize=8,color=color,fontweight="bold")
    ax.axvline(disorder_score,color="#1e293b",lw=3,zorder=5)
    ax.plot(disorder_score,0,marker="v",color="#1e293b",ms=14,zorder=6)
    sev_str=f" ({severity})" if severity else ""
    ax.set_title(f"Mean SLI Probability: {disorder_score*100:.0f}%  →  {label}{sev_str}",
                 fontweight="bold",fontsize=12)
    ax.set_xlim(0,1); ax.set_ylim(-0.45,0.45)
    ax.set_xlabel("← Healthy (0%)       Mild       Moderate       Severe (100%) →")
    ax.get_yaxis().set_visible(False); fig.tight_layout(); return fig

def plot_binary_donut(binary_probs):
    vals=[binary_probs.get("healthy",0),binary_probs.get("sli",0)]
    fig,ax=plt.subplots(figsize=(4,4))
    _,_,at=ax.pie(vals,labels=["Healthy","SLI"],colors=["#16a34a","#ef4444"],
                  autopct="%1.1f%%",startangle=90,wedgeprops=dict(width=0.55))
    for a in at: a.set_fontsize(11)
    ax.set_title("Overall SLI Probability"); fig.tight_layout(); return fig

def plot_severity_probs(severity_probs):
    if not severity_probs: return None
    order=["mild","moderate","severe"]; labels=["Mild","Moderate","Severe"]
    probs=[severity_probs.get(k,0) for k in order]
    fig,ax=plt.subplots(figsize=(5,2.5))
    bars=ax.barh(labels,probs,color=["#eab308","#f97316","#dc2626"],edgecolor="white")
    ax.set_xlim(0,1.2)
    for bar,p in zip(bars,probs):
        ax.text(bar.get_width()+0.01,bar.get_y()+bar.get_height()/2,
                f"{p*100:.0f}%",va="center",fontsize=9)
    ax.set_xlabel("Probability"); ax.set_title("Severity Classification")
    fig.tight_layout(); return fig