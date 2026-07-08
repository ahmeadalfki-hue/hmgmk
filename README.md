# Physiological Deepfake Detection Engine

A research-driven engine for Deepfake detection focused on physiological inconsistencies (eye dynamics and remote photoplethysmography - rPPG). Built with Python, MediaPipe, and OpenCV, the project is designed to serve as a cloud-native API for integration with enterprise or governmental systems.

Why physiological signals?
- Deepfakes often fail to preserve subtle biometric dynamics (blink patterns, subtle eye movements, and pulse-related color fluctuations).
- Combining temporal physiological features with visual cues increases robustness against sophisticated GAN-based forgeries.

Key components
- Face & landmark extraction: MediaPipe FaceMesh (real-time, lightweight).
- Eye-dynamics: Eye Aspect Ratio (EAR), blink rate, temporal eye stability metrics.
- rPPG: skin-ROI green-channel extraction → detrend → bandpass → FFT/PSD (baseline). Extendable to POS/CHROM/ICA for production.
- Decision fusion: rule-based checks + plug-in ML/sequence models (LSTM/Transformer) for ensemble decisions.

Project layout (suggested)
- engine/               # core detection engine (processing, feature extraction)
  - engine.py           # core engine
  - rppg.py             # POS rPPG methods
  - storage.py          # cloud/local storage abstraction
- api/                  # FastAPI service wrapper
  - app.py
  - auth.py
  - celery_app.py
  - tasks.py
  - logging_config.py
  - schemas.py
- models/               # trained ML models or checkpoints
- tests/                # unit & integration tests
- Dockerfile
- docker-compose.yml
- .github/workflows/ci-cd.yml
- requirements.txt
- README.md
- TECHNICAL_COMPLIANCE.md

Quick start (local)
1. Create virtual environment
   python -m venv .venv
   source .venv/bin/activate
2. Install dependencies
   pip install -r requirements.txt
3. Run the demo (API)
   uvicorn api.app:app --host 0.0.0.0 --port 8080 --reload

Running on Google Colab (demo of engine)
- Install dependencies in a notebook cell:
  !pip install mediapipe opencv-python-headless numpy scipy matplotlib
- Upload or mount a sample video and run scripts as described in engine examples.
- Note: cv2.imshow doesn't work in Colab — use IPython.display for images or save annotated frames.

API Integration & Productionization
- FastAPI endpoints:
  - POST /analyze: accepts multipart video file or JSON with frame URLs. Returns JSON with:
    - per-segment physiological metrics (blink_rate, avg_EAR, estimated_HR, hr_snr)
    - anomaly_score and label (likely_real/likely_fake/uncertain)
  - GET /health, GET /metrics, GET /version
- Queue-based processing for large videos:
  - Uses Celery with Redis broker in this scaffold. Replace with managed queue if desired.
- Security & compliance:
  - JWT / OIDC for service access (demo JWT implemented; integrate enterprise IdP for production).
  - mTLS and IP allow-listing recommended for government integrations.
  - Audit logs and encrypted storage for sensitive media.

Performance & Reliability notes
- rPPG requires stable lighting, minimal motion, and at least 8–10 seconds of continuous face visibility for robust HR estimation.
- Use motion-compensation/face-stabilization to reduce movement artifacts.
- Add signal-quality metrics (SNR, signal periodicity) and fall back to visual-only detection when signal is unreliable.

Extensibility roadmap
- Replace/augment rule-based decision with a temporal ML classifier (LSTM/Transformer) trained on:
  - sequences of EAR, blink events, HR curve, landmark jitter metrics, color-synchrony features.
- Integrate additional modalities: audio-lip-sync inconsistencies, eye-reflection mismatch, lip micro-expressions.
- Harden for adversarial examples and synthetic augmentation.

Governance & Data Ethics
- Ensure legal compliance when processing biometric data.
- Minimize retention of raw videos — store extracted features and hashes.
- Provide transparent reporting for each decision (explainable signals).

Contact / Contributing
- Open to collaborations with security researchers, government teams, and academic partners. Follow standard PR/issue workflow; include unit tests for new features.
