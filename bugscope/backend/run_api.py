"""BugScope prediction API — zero heavy deps fallback + ML when available."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json

PORT = 8000
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ml" / "models"
EVAL_DIR = ROOT / "ml" / "evaluation"

def load_metrics():
    p = EVAL_DIR / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}

def load_eda():
    p = EVAL_DIR / "eda_stats.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}

def compute_factors(d):
    loc = float(d.get("loc", 500))
    cc = float(d.get("cyclomatic_complexity", 10))
    churn = float(d.get("code_churn", 10))
    review = float(d.get("review_coverage", 0.7))
    test = float(d.get("test_coverage", 0.65))
    exp = float(d.get("team_experience", 0.6))
    factors = [
        {"key": "loc", "label": "LOC", "raw": min(loc / 5000, 1) * 0.12, "direction": "up"},
        {"key": "cyclomatic_complexity", "label": "Cyclomatic complexity", "raw": min(cc / 40, 1) * 0.2, "direction": "up"},
        {"key": "code_churn", "label": "Code churn", "raw": min(churn / 80, 1) * 0.15, "direction": "up"},
        {"key": "review_coverage", "label": "Review coverage", "raw": (1 - review) * 0.2, "direction": "up" if review < 0.7 else "down"},
        {"key": "test_coverage", "label": "Test coverage", "raw": (1 - test) * 0.18, "direction": "up" if test < 0.65 else "down"},
        {"key": "team_experience", "label": "Team experience", "raw": (1 - exp) * 0.15, "direction": "up" if exp < 0.6 else "down"},
    ]
    for f in factors:
        f["pct"] = round(f["raw"] * 1000) / 10
    factors.sort(key=lambda x: x["raw"], reverse=True)
    return factors

def rule_predict(d):
    factors = compute_factors(d)
    score = sum(f["raw"] for f in factors)
    prob = max(0.05, min(0.95, score))
    level = "High" if prob >= 0.6 else ("Medium" if prob >= 0.35 else "Low")
    return {
        "defective": prob >= 0.5,
        "defect_probability": round(prob, 4),
        "probability": round(prob * 100, 1),
        "risk_band": level,
        "recommendation": (
            "Prioritize review, tests, and refactoring"
            if level == "High"
            else "Increase review coverage and targeted tests"
            if level == "Medium"
            else "Maintain current quality practices"
        ),
        "mode": "rule-based",
        "factors": factors,
    }

def ml_predict(d):
    try:
        import pandas as pd
        from joblib import load
    except ImportError:
        return None
    model_path = MODEL_DIR / "defect_model.joblib"
    if not model_path.exists():
        return None
    try:
        model = load(model_path)
        encoders = load(MODEL_DIR / "encoders.joblib")
        scaler = load(MODEL_DIR / "scaler.joblib")
        selector = load(MODEL_DIR / "selector.joblib")
        feature_cols = load(MODEL_DIR / "feature_cols.joblib")
        num_cols = load(MODEL_DIR / "num_cols.joblib")
        cat_cols = load(MODEL_DIR / "cat_cols.joblib")
        row = {c: d.get(c, 0) for c in feature_cols}
        df = pd.DataFrame([row])
        for c in cat_cols:
            le = encoders[c]
            val = str(df[c].iloc[0])
            if val not in set(le.classes_):
                val = le.classes_[0]
            df[c] = le.transform([val])
        df[num_cols] = scaler.transform(df[num_cols].astype(float))
        X = selector.transform(df[feature_cols].values)
        pred = bool(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        level = "High" if proba >= 0.6 else ("Medium" if proba >= 0.35 else "Low")
        return {
            "defective": pred,
            "defect_probability": round(proba, 4),
            "probability": round(proba * 100, 1),
            "risk_band": level,
            "recommendation": (
                "Prioritize review, tests, and refactoring"
                if level == "High"
                else "Increase review coverage and targeted tests"
                if level == "Medium"
                else "Maintain current quality practices"
            ),
            "mode": "ml",
            "factors": compute_factors(d),
        }
    except Exception as e:
        print("ML error:", e)
        return None

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def _send(self, code, data):
        body = json.dumps(data).encode() if data is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if code != 204:
            self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, None)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, {
                "status": "ok",
                "service": "BugScope API",
                "model_exists": (MODEL_DIR / "defect_model.joblib").exists(),
            })
        if self.path.startswith("/metrics"):
            return self._send(200, load_metrics())
        if self.path.startswith("/eda"):
            return self._send(200, load_eda())
        if self.path.startswith("/system"):
            sys_path = ROOT / "frontend" / "data" / "system_data.json"
            if sys_path.exists():
                return self._send(200, json.loads(sys_path.read_text()))
            return self._send(404, {"error": "system data not found"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n).decode() or "{}")
        except Exception:
            return self._send(400, {"error": "invalid json"})

        if self.path.startswith("/predict/batch"):
            items = body.get("items") or body.get("modules") or []
            if not isinstance(items, list):
                return self._send(400, {"error": "items must be a list"})
            out = []
            for d in items[:200]:
                out.append(ml_predict(d) or rule_predict(d))
            return self._send(200, {"results": out, "count": len(out)})

        if self.path.startswith("/predict"):
            result = ml_predict(body) or rule_predict(body)
            return self._send(200, result)

        self._send(404, {"error": "not found"})

if __name__ == "__main__":
    print(f"BugScope API → http://localhost:{PORT}")
    print("  GET  /health  /metrics  /eda  /system")
    print("  POST /predict  /predict/batch")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
