#!/usr/bin/env python3
"""
Silver AI Foundry — Inference Engine
Loads a saved model and runs prediction on provided features.
"""

import argparse
import json
import pickle
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = "/tmp/models"

def predict(args):
    job_id = args.job_id
    features = json.loads(args.features)

    model_path  = os.path.join(MODEL_DIR, f"{job_id}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{job_id}_scaler.pkl")

    if not os.path.exists(model_path):
        print(json.dumps({"error": "Model not found on disk."}))
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Handle different model types
    model_cls = type(model).__name__

    if model_cls == "KMeans":
        prediction = int(model.predict(X_scaled)[0])
        result = {"prediction": f"Cluster_{prediction}", "confidence": None}
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        pred_idx = int(np.argmax(proba))
        classes = getattr(model, "classes_", None)
        label = str(classes[pred_idx]) if classes is not None else f"Class_{pred_idx}"
        result = {
            "prediction": label,
            "confidence": round(float(proba[pred_idx]) * 100, 2),
            "all_classes": {
                str(c): round(float(p) * 100, 2)
                for c, p in zip(classes or range(len(proba)), proba)
            }
        }
    else:
        pred = float(model.predict(X_scaled)[0])
        result = {"prediction": round(pred, 4), "confidence": None}

    print(json.dumps(result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id",   required=True)
    parser.add_argument("--features", required=True, help="JSON array of feature values")
    predict(parser.parse_args())
