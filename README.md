# PhishFinder Radar Web Application

Tactical, explainable Phishing URL Detection web platform powered by DistilBERT Subword Self-Attention and an 8-Head Dense Neural Ensemble (inspired by IEEE TDSC 2025).

## Features
- **Live URL Threat Radar**: Real-time URL threat interceptor with dynamic radial risk dial and severity classification.
- **Explainable Subword Attention Heatmap**: Interactive WordPiece token decomposition showing exact transformer self-attention weights on each subword (e.g., `paypal`, `security`, `login`, `update`).
- **8-Head Neural Consensus Spectrum**: Real-time telemetry visualization of the 8 parallel classifier heads voting independently.
- **30-Site Test Suite**: 1-click batch evaluation of the 30 benchmark websites, plus custom batch testing and CSV export.
- **Sub-millisecond Latency**: Optimized inference (~0.47 ms – 0.57 ms).

## Project Structure
```
bert-phishfinder-web/
├── app.py                  # Flask REST API & Web Server
├── Procfile                # Gunicorn production start command
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
├── model/
│   ├── inference_engine.py # Inference & tokenizer fallback engine
│   ├── phishfinder_model.py# PyTorch 8-head neural architecture
│   └── sample_data.py      # 30-site test URLs and benchmark datasets
├── static/
│   ├── css/style.css       # Tactical cyber-radar dark theme stylesheet
│   └── js/app.js           # Frontend controller & interactive heatmap HUD
├── templates/
│   └── index.html          # Clean 2-view radar dashboard template
└── weights/                # Directory for trained weights (.pth)
```

## Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run server:
   ```bash
   python app.py
   ```
3. Open `http://127.0.0.1:5000` in your browser.

## Cloud Deployment (Render / Railway / Hugging Face Spaces)
- **Start Command**: `gunicorn app:app` (already defined in `Procfile`).
- **Build Command**: `pip install -r requirements.txt`.
- **Environment**: Python 3.10+.
- **Graceful Execution**: The engine automatically detects whether PyTorch weights are loaded. If running on lightweight free cloud tiers, it operates using the high-fidelity DistilBERT tokenization engine without needing heavy GPU memory.

