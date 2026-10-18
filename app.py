"""Bank note authentication API (Flask): web form + JSON API."""
import math
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent


def load_settings() -> dict:
    params_path = os.environ.get("PARAMS_PATH", BASE_DIR / "params.yaml")
    with open(params_path) as file:
        params = yaml.safe_load(file)
    return {
        "features": params["data"]["features"],
        "labels": {int(k): v for k, v in params["serving"]["class_labels"].items()},
        "ranges": params["serving"]["feature_ranges"],
    }


def parse_features(source, settings):
    """Read and validate the 4 numbers. Returns (values, error_message)."""
    values = {}
    for name in settings["features"]:
        raw = source.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, f"'{name}' must be a number"
        if not math.isfinite(value):
            return None, f"'{name}' must be a finite number"
        low, high = settings["ranges"][name]
        if not low <= value <= high:
            return None, f"'{name}' must be between {low} and {high}"
        values[name] = value
    return values, None


def create_app(model_path=None) -> Flask:
    app = Flask(__name__)
    settings = load_settings()
    path = model_path or os.environ.get(
        "MODEL_PATH", BASE_DIR / "models" / "banknote_auth_model.joblib"
    )
    model = joblib.load(path) if Path(path).exists() else None

    def run_model(values):
        frame = pd.DataFrame([values], columns=settings["features"])
        predicted = int(model.predict(frame)[0])
        confidence = float(model.predict_proba(frame)[0][predicted])
        return {
            "prediction": predicted,
            "label": settings["labels"][predicted],
            "confidence": round(confidence, 4),
        }

    @app.get("/")
    def home():
        return render_template(
            "index.html", features=settings["features"], ranges=settings["ranges"]
        )

    @app.get("/health")
    def health():
        status = 200 if model is not None else 503
        return jsonify(status="ok" if model else "model not loaded"), status

    @app.post("/predict")
    def predict_form():
        values, error = parse_features(request.form, settings)
        message = error
        if error is None:
            if model is None:
                message = "Model is not available"
            else:
                result = run_model(values)
                message = f"Bank note is: {result['label']} ({result['confidence']:.0%} confidence)"
        return render_template(
            "index.html",
            features=settings["features"],
            ranges=settings["ranges"],
            prediction_text=message,
            form_values=request.form,
        )

    @app.post("/api/predict")
    def predict_api():
        if model is None:
            return jsonify(error="Model is not available"), 503
        values, error = parse_features(request.get_json(silent=True) or {}, settings)
        if error:
            return jsonify(error=error), 400
        return jsonify(run_model(values))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
