"""
train_sli_model.py
─────────────────────────────────────────────────────────────────────────────
Train an SLI vs Healthy classifier from the LANNA speech database.

Expected folder layout (the zip you downloaded):
    Data/
    ├── Healthy/
    │   ├── H26/
    │   │   ├── 03_1SL/   ← session folders
    │   │   │   ├── *.wav
    │   │   │   └── *.lbl
    │   │   └── 04_2SL/
    │   └── H28/ ...
    └── Patient/
        ├── P01/
        │   ├── 01_1SL/
        │   │   ├── *.wav
        │   │   └── *.lbl
        │   ├── 02_2SL/
        │   └── 03_3SL/   ← digit after underscore = severity (1/2/3)
        └── P02/ ...

Severity is read from the session-folder name pattern  ##_<N>SL
  → N=1  mild, N=2  moderate, N=3  severe
  (Healthy folders may also have ##_<N>SL names; they are all labelled 0.)

Labels produced:
    "healthy"       → class 0
    "sli_mild"      → class 1
    "sli_moderate"  → class 2
    "sli_severe"    → class 3

Output files (saved to models/):
    sli_classifier.pkl   - trained sklearn RandomForest
    sli_scaler.pkl       - StandardScaler fitted on training data
    sli_meta.json        - class names, feature count, dataset stats

Usage:
    python train_sli_model.py --data_root ./Data [--model_dir ./models]
                              [--n_estimators 300] [--test_size 0.2]
                              [--severity]

    --severity   : train a 4-class model (healthy/mild/moderate/severe)
                   default is 2-class (healthy / sli)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import json
import argparse
import warnings
import numpy as np
import joblib
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, balanced_accuracy_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─── import our feature extractor ─────────────────────────────────────────
try:
    from sli_audio_functions import extract_all_features, FEATURE_NAMES
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from sli_audio_functions import extract_all_features, FEATURE_NAMES


# ═════════════════════════════════════════════
#  FOLDER SCANNING
# ═════════════════════════════════════════════

def severity_from_session(session_name: str) -> int:
    """
    Parse severity digit from session folder names like:
        03_1SL  →  1 (mild)
        04_2SL  →  2 (moderate)
        05_3SL  →  3 (severe)
    Returns 0 if pattern not found (treated as unspecified / healthy).
    """
    m = re.search(r'_(\d)SL', session_name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def scan_database(data_root: str, use_severity: bool = True):
    """
    Walk the LANNA Data/ directory and collect (wav_path, label_str) pairs.

    Parameters
    ----------
    data_root    : path to  Data/  folder
    use_severity : if True → 4 classes; if False → 2 classes

    Returns
    -------
    List of (wav_path: str, label: str)
    """
    data_root = Path(data_root)
    records = []

    for group_folder in data_root.iterdir():
        if not group_folder.is_dir():
            continue

        name_lower = group_folder.name.lower()
        if "healthy" in name_lower or name_lower.startswith("h"):
            group = "healthy"
        elif "patient" in name_lower or name_lower.startswith("p"):
            group = "patient"
        else:
            # Try to infer: if folder contains sub-folders starting with H/P
            sub_names = [s.name for s in group_folder.iterdir() if s.is_dir()]
            if any(s.startswith("H") for s in sub_names):
                group = "healthy"
            elif any(s.startswith("P") for s in sub_names):
                group = "patient"
            else:
                print(f"  [SKIP] Cannot determine group for: {group_folder.name}")
                continue

        # walk speaker folders (H26, P01, ...)
        for speaker_folder in sorted(group_folder.iterdir()):
            if not speaker_folder.is_dir():
                continue

            # walk session folders (03_1SL, 04_2SL, ...)
            for session_folder in sorted(speaker_folder.iterdir()):
                if not session_folder.is_dir():
                    continue

                sev = severity_from_session(session_folder.name)

                # assign label
                if group == "healthy":
                    label = "healthy"
                else:
                    if use_severity:
                        if sev == 1:
                            label = "sli_mild"
                        elif sev == 2:
                            label = "sli_moderate"
                        elif sev == 3:
                            label = "sli_severe"
                        else:
                            label = "sli_mild"   # unknown severity → mild
                    else:
                        label = "sli"

                # collect all WAV files in this session
                for wav_file in sorted(session_folder.glob("*.wav")):
                    records.append((str(wav_file), label))

                # also search one level deeper (some databases nest further)
                for sub in session_folder.iterdir():
                    if sub.is_dir():
                        for wav_file in sorted(sub.glob("*.wav")):
                            records.append((str(wav_file), label))

    return records


# ═════════════════════════════════════════════
#  FEATURE EXTRACTION LOOP
# ═════════════════════════════════════════════

def build_feature_matrix(records, verbose=True):
    """
    Extract features for every (wav_path, label) record.
    Returns X (n_samples, n_features), y (n_samples,), labels list.
    """
    X_list, y_list = [], []
    skipped = 0

    for i, (wav_path, label) in enumerate(records):
        if verbose and (i % 20 == 0 or i == len(records) - 1):
            print(f"  Extracting features {i+1}/{len(records)}  ({label})  "
                  f"{os.path.basename(wav_path)}")
        try:
            feat = extract_all_features(wav_path)
            if feat.sum() == 0:   # empty / too-short file
                skipped += 1
                continue
            X_list.append(feat)
            y_list.append(label)
        except Exception as e:
            print(f"  [WARN] Failed on {wav_path}: {e}")
            skipped += 1

    if verbose:
        print(f"\n  Total extracted : {len(X_list)}")
        print(f"  Skipped / failed: {skipped}")

    return np.array(X_list, dtype=np.float32), np.array(y_list)


# ═════════════════════════════════════════════
#  TRAINING
# ═════════════════════════════════════════════

def train(X, y, n_estimators=300, test_size=0.2, random_state=42):
    """
    Train RandomForest + GradientBoosting, pick best by CV F1-macro.
    Returns (best_clf, scaler, report_str, classes).
    """
    classes = sorted(set(y))
    label2idx = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label2idx[c] for c in y])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── cross-validation ──────────────────────────────────────────────────
    print("\n  Cross-validating classifiers …")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=min(n_estimators, 150),
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            C=10,
            class_weight="balanced",
            probability=True,
            random_state=random_state
        ),
    }

    best_name, best_score, best_clf = None, -1, None
    for name, clf in candidates.items():
        cv_scores = cross_val_score(clf, X_scaled, y_int, cv=skf,
                                    scoring="f1_macro", n_jobs=-1)
        mean_score = cv_scores.mean()
        print(f"    {name:25s}  CV F1-macro = {mean_score:.4f} "
              f"(± {cv_scores.std():.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_name  = name
            best_clf   = clf

    print(f"\n  Best classifier: {best_name}  (CV F1={best_score:.4f})")

    # ── final train / test split ──────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_int, test_size=test_size,
        stratify=y_int, random_state=random_state
    )
    best_clf.fit(X_tr, y_tr)
    y_pred = best_clf.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    bacc = balanced_accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred,
                                   target_names=classes, zero_division=0)

    print(f"\n  Hold-out accuracy         : {acc:.4f}")
    print(f"  Hold-out balanced accuracy: {bacc:.4f}")
    print(f"\nClassification Report:\n{report}")

    # ── confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_te, y_pred)
    print("Confusion Matrix:")
    header = "       " + "  ".join(f"{c[:8]:>8}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {classes[i][:8]:>8}  " + "  ".join(f"{v:8d}" for v in row))

    # retrain on full data for deployment
    best_clf.fit(X_scaled, y_int)

    return best_clf, scaler, report, classes, {"accuracy": acc, "balanced_accuracy": bacc}


# ═════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ═════════════════════════════════════════════

def print_top_features(clf, n=15):
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:n]
    print(f"\n  Top {n} most important features:")
    for rank, idx in enumerate(indices, 1):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
        print(f"    {rank:2d}. {name:35s}  {importances[idx]:.4f}")


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train SLI classifier from LANNA database"
    )
    parser.add_argument(
        "--data_root", type=str, default="./Data",
        help="Path to the LANNA Data/ folder (contains Healthy/ and Patient/)"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./models",
        help="Directory to save model files"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=300,
        help="Number of trees for RandomForest / GradientBoosting"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data for hold-out evaluation"
    )
    parser.add_argument(
        "--severity", action="store_true",
        help="Train 4-class model (healthy/mild/moderate/severe) "
             "instead of binary (healthy/sli)"
    )
    parser.add_argument(
        "--no_severity", dest="severity", action="store_false",
        help="Train binary 2-class model"
    )
    parser.set_defaults(severity=True)

    args = parser.parse_args()

    print("=" * 60)
    print("  LANNA SLI Classifier Training")
    print("=" * 60)
    print(f"  Data root  : {args.data_root}")
    print(f"  Model dir  : {args.model_dir}")
    print(f"  Severity   : {args.severity}")
    print(f"  Estimators : {args.n_estimators}")
    print()

    # ── 1. Scan database ──────────────────────────────────────────────────
    print("Step 1 — Scanning database …")
    records = scan_database(args.data_root, use_severity=args.severity)

    if not records:
        print("\n[ERROR] No WAV files found!")
        print(f"  Expected structure under: {args.data_root}")
        print("  Data/")
        print("  ├── Healthy/  H26/  03_1SL/  *.wav")
        print("  └── Patient/  P01/  01_1SL/  *.wav")
        return

    label_counts = Counter(lbl for _, lbl in records)
    print(f"\n  Found {len(records)} WAV files:")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"    {lbl:20s}  {cnt:4d} files")

    # ── 2. Feature extraction ─────────────────────────────────────────────
    print("\nStep 2 — Extracting acoustic features …")
    X, y = build_feature_matrix(records)

    if len(X) < 10:
        print("[ERROR] Too few valid samples to train. Check your data path.")
        return

    label_counts_final = Counter(y)
    print(f"\n  Valid samples per class:")
    for lbl, cnt in sorted(label_counts_final.items()):
        print(f"    {lbl:20s}  {cnt:4d}")

    # ── 3. Train ──────────────────────────────────────────────────────────
    print("\nStep 3 — Training classifiers …")
    clf, scaler, report, classes, metrics = train(
        X, y,
        n_estimators=args.n_estimators,
        test_size=args.test_size
    )

    print_top_features(clf)

    # ── 4. Save ───────────────────────────────────────────────────────────
    print(f"\nStep 4 — Saving model to {args.model_dir}/ …")
    os.makedirs(args.model_dir, exist_ok=True)

    clf_path    = os.path.join(args.model_dir, "sli_classifier.pkl")
    scaler_path = os.path.join(args.model_dir, "sli_scaler.pkl")
    meta_path   = os.path.join(args.model_dir, "sli_meta.json")

    joblib.dump(clf,    clf_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "classes":              classes,
        "n_features":           int(X.shape[1]),
        "feature_names":        FEATURE_NAMES,
        "use_severity":         args.severity,
        "n_train_samples":      int(len(X)),
        "label_distribution":   {k: int(v) for k, v in label_counts_final.items()},
        "cv_accuracy":          metrics["accuracy"],
        "cv_balanced_accuracy": metrics["balanced_accuracy"],
        "classification_report": report,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✔ sli_classifier.pkl  saved → {clf_path}")
    print(f"  ✔ sli_scaler.pkl      saved → {scaler_path}")
    print(f"  ✔ sli_meta.json       saved → {meta_path}")
    print("\n  Training complete!  You can now run the Streamlit app.")


if __name__ == "__main__":
    main()