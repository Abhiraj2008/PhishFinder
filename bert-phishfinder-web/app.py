"""
BERT-PhishFinder Web Application
Backend Server using Flask
Provides REST APIs for real-time URL phishing detection,
interactive transformer pipeline visualization, batch testing,
and benchmark comparative analysis.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.inference_engine import PhishFinderInferenceEngine
from model.sample_data import PRESET_BENCHMARK_URLS, CUSTOM_TEST_URLS, PAPER_METRICS, BASELINE_COMPARISONS

app = Flask(__name__)

# Initialize the inference engine
weights_file = os.path.join(os.path.dirname(__file__), "weights", "phishfinder_mini.pth")
engine = PhishFinderInferenceEngine(weights_path=weights_file)


@app.route("/")
def index():
    """Renders the main dashboard interface."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict_url():
    """
    Analyzes an individual URL through the 5-stage BERT-PhishFinder pipeline:
    1. DistilBERT WordPiece Tokenizer
    2. Contextual Transformer Embeddings
    3. SpatialDropout1D & Dual Pooling (1536-dim)
    4. 8 Parallel Dense Classifier Heads
    5. Average Consensus Output (P_final)
    """
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "Please provide a valid URL"}), 400

    try:
        result = engine.analyze_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    """
    Evaluates a batch of URLs and returns comprehensive summary metrics.
    """
    data = request.get_json(force=True, silent=True) or {}
    urls = data.get("urls", [])

    if not urls or not isinstance(urls, list):
        return jsonify({"error": "Please provide a list of URLs under 'urls'"}), 400

    # Limit batch to 50 for fast interactive response
    urls_to_test = urls[:50]
    results = []
    phishing_count = 0
    legit_count = 0
    total_time = 0.0

    for item in urls_to_test:
        url_str = item if isinstance(item, str) else item.get("url", "")
        if not url_str:
            continue
        res = engine.analyze_url(url_str)
        if res["is_phishing"]:
            phishing_count += 1
        else:
            legit_count += 1
        total_time += res["inference_time_ms"]
        results.append({
            "url": res["url"],
            "verdict": res["verdict"],
            "is_phishing": res["is_phishing"],
            "phishing_probability": res["phishing_probability"],
            "risk_level": res["risk_level"],
            "inference_time_ms": res["inference_time_ms"],
            "token_count": res["pipeline"]["step1_tokenization"]["token_count"]
        })

    total = len(results)
    avg_latency = round(total_time / total, 2) if total > 0 else 0.0

    return jsonify({
        "total_scanned": total,
        "phishing_count": phishing_count,
        "legitimate_count": legit_count,
        "phishing_rate_pct": round((phishing_count / total) * 100, 1) if total > 0 else 0,
        "average_latency_ms": avg_latency,
        "results": results
    })


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    """
    Returns empirical benchmarks and comparative data from IEEE TDSC 2025 Paper.
    """
    return jsonify({
        "paper_metrics": PAPER_METRICS,
        "baseline_comparisons": BASELINE_COMPARISONS
    })


@app.route("/api/presets", methods=["GET"])
def get_presets():
    """
    Returns the 20 benchmark test URLs (10 legitimate, 10 phishing).
    """
    return jsonify({"presets": PRESET_BENCHMARK_URLS})


@app.route("/api/custom-tests", methods=["GET"])
def get_custom_tests():
    """
    Returns the 30 custom test URLs from the notebook.
    """
    return jsonify({"urls": CUSTOM_TEST_URLS})


@app.route("/api/status", methods=["GET"])
def get_status():
    """
    Returns system status and engine mode.
    """
    has_weights = os.path.exists(weights_file)
    return jsonify({
        "status": "online",
        "engine_mode": engine.engine_mode if hasattr(engine, "engine_mode") else ("Live PyTorch" if engine.is_live_pytorch else "BERT-PhishFinder Emulation Engine"),
        "weights_path": weights_file,
        "weights_found": has_weights,
        "device": getattr(engine, "device", "cpu"),
        "model_architecture": {
            "backbone": "DistilBERT (distilbert-base-uncased)",
            "layers": 6,
            "attention_heads": 12,
            "hidden_dimension": 768,
            "regularization": "SpatialDropout1D (p=0.1)",
            "pooling": "GlobalAveragePooling + GlobalMaxPooling (1,536 dim)",
            "classifier_heads": 8,
            "head_units": 64,
            "activation": "ReLU + Sigmoid",
            "consensus": "Average Pooling (threshold 0.50)"
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"=== BERT-PhishFinder Web Platform running on http://127.0.0.1:{port} ===")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
