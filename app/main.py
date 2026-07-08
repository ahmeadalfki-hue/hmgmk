"""
app/main.py

FastAPI service that exposes an endpoint to analyze an uploaded video file
using physiological inconsistency signals (rPPG + blink analysis).

Endpoint:
  POST /analyze
    - form field: file -> multipart file (video, mp4/webm)
    - optional query params:
        max_seconds: limit processing time (float, default 30.0)
        sample_fps: override sampling fps used by the engine (int, default: video fps)
    - returns JSON:
      {
        "id": "<uuid>",
        "verdict": "likely_real|likely_fake|inconclusive",
        "score": 0.0-1.0,
        "confidence": 0.0-1.0,
        "metrics": {bpms, snr, blink_count, frames_processed},
        "evidence": {
           "ppg_plot_base64": "...",
           "snapshot_base64": "..."
        }
      }

Docstrings: this file documents the API behavior for inclusion in README or auto-docs.
"""
import io
import os
import uuid
import base64
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
from app.core.physio_engine import PhysioEngine, bandpass_filter
from PIL import Image, ImageDraw

app = FastAPI(title="Deepfake Physio Engine API",
              description="Detects physiological inconsistencies (rPPG, blink patterns) from video.",
              version="0.1")

def image_bytes_to_base64_png(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode('ascii')

def pil_image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return image_bytes_to_base64(buf.getvalue())

def plot_ppg_and_blinks(times, raw_ppg, filtered_ppg, blinks, out_size=(1200, 400)):
    """
    Create PNG bytes for the PPG + blink plot.
    """
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(out_size[0]/100, out_size[1]/100))
    # normalize raw for visibility
    raw_norm = (raw_ppg - np.mean(raw_ppg)) / (np.std(raw_ppg) + 1e-8)
    ax.plot(times, raw_norm, label='Raw (normalized) green mean', alpha=0.6)
    ax.plot(times, filtered_ppg, label='Bandpass filtered (0.7-4.0 Hz)', linewidth=2)
    for b in blinks:
        ax.axvline(x=b, color='r', linestyle='--', alpha=0.7)
    ax.set_title("rPPG signal (green channel) and detected blinks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.legend()
    ax.grid(True)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def snapshot_with_landmarks(frame_bgr, landmarks):
    """
    Return PIL image bytes of frame with landmarks drawn (PNG bytes).
    landmarks: MediaPipe landmark list
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    h, w = frame_bgr.shape[:2]
    # draw small circles for each landmark (red)
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        r = max(1, int(min(w, h) * 0.005))
        draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 0, 0), width=2)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return buf.getvalue()

def compute_verdict_and_score(bpm, snr, blink_count):
    """
    Heuristic scoring:
      - Start with neutral score 0.5
      - Increase score when bpm in plausible physiological range and SNR is good
      - Decrease when SNR is very low or BPM is implausible (or missing)
      - Blink count of zero is suspicious (reduce a bit)
    Returns (verdict_str, score_float, confidence_float)
    Note: This is a simple rule-based scorer meant for demo / explainability.
    """
    score = 0.5
    if bpm is None:
        score -= 0.25
    else:
        # plausible bpm range 45-120 (rest/active)
        if 45 <= bpm <= 120:
            score += 0.2
        else:
            score -= 0.2
    if snr is None:
        score -= 0.2
    else:
        # map snr to contribution
        if snr > 6:
            score += 0.2
        elif snr > 3:
            score += 0.05
        else:
            score -= 0.2
    if blink_count == 0:
        score -= 0.1
    # clamp
    score = max(0.0, min(1.0, score))
    # simple confidence mapping from snr
    conf = 0.5
    if snr is not None:
        conf = float(min(1.0, max(0.1, min(1.0, snr / 8.0))))
    # verdict thresholding
    if score >= 0.7 and conf > 0.4:
        verdict = "likely_real"
    elif score <= 0.3:
        verdict = "likely_fake"
    else:
        verdict = "inconclusive"
    return verdict, score, conf

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...),
                        max_seconds: Optional[float] = Query(30.0, description="Max seconds of video to process"),
                        sample_fps: Optional[int] = Query(None, description="Override sampling FPS (default: video FPS)")):
    """
    Analyze uploaded video for physiological cues (rPPG + blinks).

    Request:
      - multipart/form-data with key 'file' containing the video.

    Optional query parameters:
      - max_seconds: limit the number of seconds processed (safety for large uploads).
      - sample_fps: override sampling fps (useful when video FPS is unreliable).

    Response JSON:
      id: unique analysis id
      verdict: 'likely_real' | 'likely_fake' | 'inconclusive'
      score: 0..1 (higher = more likely real)
      confidence: 0..1
      metrics: { estimated_bpm, snr, blink_count, frames_processed }
      evidence: { ppg_plot_base64, snapshot_base64 }

    Notes:
      - This endpoint is synchronous and intended for short clips. For longer videos,
        consider an async/queue-based workflow (e.g., upload -> process in background).
      - Evidence images are returned as Base64 PNG strings so the API consumer can
        store or display them without file I/O on the client side.
    """
    # Basic validation
    if not file.filename.lower().endswith(('.mp4', '.mov', '.webm', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use mp4/webm/mov/avi/mkv.")

    # Write upload to a temporary file
    suffix = os.path.splitext(file.filename)[1] or '.mp4'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        # Open video with OpenCV
        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps and total_frames else None

        use_fps = int(sample_fps) if sample_fps is not None else int(round(fps))
        engine = PhysioEngine(sampling_fps=use_fps, ppg_window_seconds=min(60, int(max_seconds or 30)))

        frame_idx = 0
        processed_frames = 0
        # read frames up to max_seconds
        max_frames = int((max_seconds or 30.0) * fps)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps
            if max_seconds and timestamp > max_seconds:
                break
            engine.process_frame(frame, timestamp)
            frame_idx += 1
            processed_frames += 1
        cap.release()

        # compute metrics
        bpm, snr, filtered = engine.estimate_bpm()
        blinks = engine.detect_blinks(ear_thresh=0.20, min_consec_frames=max(1, int(0.04 * use_fps)))
        blink_count = len(blinks)

        # create evidence images as PNG bytes then base64
        ppg_base64 = None
        snapshot_base64 = None
        if filtered is not None and len(engine.ppg_buffer) > 0:
            # align timestamps for ppg samples (engine.times covers all frames with faces)
            # take times length equal to len(ppg_buffer) by collecting times where mean_g existed
            # For simplicity we map first len(ppg_buffer) times
            times = np.array(engine.times[:len(engine.ppg_buffer)])
            raw_ppg = np.array(engine.ppg_buffer)
            # filtered may be shorter/longer; interpolate or pad to match times length
            if len(filtered) != len(raw_ppg):
                # try to resample filtered to raw length via linear interpolation of indices
                idx_src = np.linspace(0, len(filtered)-1, num=len(filtered))
                idx_dst = np.linspace(0, len(filtered)-1, num=len(raw_ppg))
                filtered_resampled = np.interp(idx_dst, idx_src, filtered)
            else:
                filtered_resampled = filtered
            png_bytes = plot_ppg_and_blinks(times, raw_ppg, filtered_resampled, blinks)
            ppg_base64 = image_bytes_to_base64_png(png_bytes)

        # snapshot with landmarks
        if engine.last_frame is not None and engine.last_landmarks is not None:
            png_bytes = snapshot_with_landmarks(engine.last_frame, engine.last_landmarks)
            snapshot_base64 = image_bytes_to_base64_png(png_bytes)

        verdict, score, conf = compute_verdict_and_score(bpm, snr, blink_count)

        response = {
            "id": str(uuid.uuid4()),
            "verdict": verdict,
            "score": float(score),
            "confidence": float(conf),
            "metrics": {
                "estimated_bpm": float(bpm) if bpm is not None else None,
                "bpm_snr": float(snr) if snr is not None else None,
                "blink_count": int(blink_count),
                "frames_processed": int(processed_frames)
            },
            "evidence": {
                "ppg_plot_base64": ppg_base64,
                "snapshot_base64": snapshot_base64
            }
        }
        return JSONResponse(content=response)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
