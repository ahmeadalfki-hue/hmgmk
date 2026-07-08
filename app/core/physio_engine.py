"""
app/core/physio_engine.py

PhysioEngine:
- Uses MediaPipe FaceMesh to extract facial landmarks per-frame.
- Computes a simple rPPG signal from the GREEN channel averaged over a forehead ROI.
  Why green? Hemoglobin absorption in the visible spectrum often produces a stronger
  pulsatile signature in the green channel for standard RGB cameras.
- Applies a bandpass filter (0.7 - 4.0 Hz) to isolate physiological heart-rate frequencies.
- Uses Eye Aspect Ratio (EAR) heuristic to detect blinks (interpretable and lightweight).
- Keeps last processed frame + landmarks for snapshot evidence.

This module is intentionally minimal and explainable; it is suitable for piping
into an API layer (see app/main.py).
"""
from collections import deque
import math
import numpy as np
from scipy import signal
import mediapipe as mp
import cv2

mp_face = mp.solutions.face_mesh

def bandpass_filter(sig, fs=30.0, low=0.7, high=4.0, order=4):
    """
    Zero-phase bandpass filter (filtfilt) to avoid phase shifts in physiological signals.
    Physiological band: ~0.7-4.0 Hz (42 - 240 bpm) — wide to accommodate noisy data.
    """
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = signal.butter(order, [lowcut, highcut], btype='band')
    return signal.filtfilt(b, a, sig)

def eye_aspect_ratio(coords):
    """
    coords: list of 6 (x,y) points around an eye (ordered)
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Lower EAR => eye closed.
    """
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    h = np.linalg.norm(np.array(coords[0]) - np.array(coords[3])) + 1e-6
    ear = (v1 + v2) / (2.0 * h)
    return ear

class PhysioEngine:
    """
    Simple, stateful engine that ingests frames and computes:
      - ppg_buffer: sequence of mean-green values from forehead ROI
      - ear_buffer: sequence of EAR values for blink detection
      - keeps last_frame and last_landmarks for snapshot evidence

    Public methods:
      - process_frame(frame_bgr, timestamp)
      - estimate_bpm() -> (bpm, snr, filtered_signal)
      - detect_blinks(ear_thresh, min_consec_frames) -> list of timestamps
    """
    def __init__(self, sampling_fps=30, ppg_window_seconds=15):
        self.sampling_fps = sampling_fps
        self.ppg_window = int(ppg_window_seconds * sampling_fps)
        self.ppg_buffer = []  # raw mean-green values
        self.ear_buffer = []
        self.times = []
        # to provide snapshot evidence
        self.last_frame = None
        self.last_landmarks = None

        # MediaPipe FaceMesh tuned for video (tracking)
        self.face_mesh = mp_face.FaceMesh(static_image_mode=False,
                                          max_num_faces=1,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
        # Landmark sets for EAR - indices chosen for MediaPipe face mesh
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [263, 387, 385, 362, 380, 373]

    def _face_bbox_from_landmarks(self, landmarks, w, h):
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = (max_x - min_x) * 0.2
        pad_y = (max_y - min_y) * 0.25
        x1 = int(max(0, (min_x - pad_x) * w))
        y1 = int(max(0, (min_y - pad_y) * h))
        x2 = int(min(w, (max_x + pad_x) * w))
        y2 = int(min(h, (max_y + pad_y) * h))
        return x1, y1, x2, y2

    def _forehead_roi(self, landmarks, w, h):
        """
        Choose a forehead ROI above the face bounding box and centered between temples.
        This ROI reduces contamination from lips/chin motion and focuses on areas with
        stronger skin signal.
        """
        x1, y1, x2, y2 = self._face_bbox_from_landmarks(landmarks, w, h)
        height = y2 - y1
        top = max(0, y1 - int(0.4 * height))
        roi_left = x1 + int(0.2 * (x2 - x1))
        roi_right = x2 - int(0.2 * (x2 - x1))
        roi_bottom = max(y1 - int(0.05 * height), top + 2)
        return (roi_left, top, roi_right, roi_bottom)

    def process_frame(self, frame_bgr, timestamp):
        """
        Process a single BGR frame (numpy array) and append features to buffers.
        Returns: dict with extracted features or None if no face detected.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            # keep time step for alignment (optional) - we don't append to buffers here
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        # keep last frame+landmarks for snapshot evidence
        self.last_frame = frame_bgr.copy()
        self.last_landmarks = landmarks

        # compute EAR
        left_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.LEFT_EYE]
        right_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.RIGHT_EYE]
        left_ear = eye_aspect_ratio(left_coords)
        right_ear = eye_aspect_ratio(right_coords)
        ear = (left_ear + right_ear) / 2.0

        # forehead ROI mean green
        x1, y1, x2, y2 = self._forehead_roi(landmarks, w, h)
        roi = frame_bgr[y1:y2, x1:x2]
        mean_g = None
        if roi.size != 0:
            mean_g = float(np.mean(roi[:, :, 1]))  # green channel

        # push buffers
        self.times.append(timestamp)
        self.ear_buffer.append(ear)
        if mean_g is not None:
            self.ppg_buffer.append(mean_g)

        return {
            "ear": ear,
            "mean_g": mean_g,
            "roi_bbox": (x1, y1, x2, y2),
            "landmarks": landmarks
        }

    def estimate_bpm(self):
        """
        Returns: (bpm, snr, filtered_signal)
        - bpm: estimated beats per minute (float) or None
        - snr: simple SNR-like metric (peak / median of FFT band) or None
        - filtered_signal: numpy array of filtered normalized PPG
        """
        if len(self.ppg_buffer) < max(4, int(0.5 * self.sampling_fps)):
            return None, None, None
        arr = np.array(self.ppg_buffer)
        arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
        filtered = bandpass_filter(arr, fs=self.sampling_fps, low=0.7, high=4.0)
        n = len(filtered)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sampling_fps)
        fft = np.abs(np.fft.rfft(filtered))
        mask = (freqs >= 0.7) & (freqs <= 4.0)
        if not mask.any():
            return None, None, filtered
        freqs_sel = freqs[mask]
        fft_sel = fft[mask]
        peak_idx = int(np.argmax(fft_sel))
        peak_freq = freqs_sel[peak_idx]
        bpm = float(peak_freq * 60.0)
        snr = float(fft_sel[peak_idx] / (np.median(fft_sel) + 1e-8))
        return bpm, snr, filtered

    def detect_blinks(self, ear_thresh=0.20, min_consec_frames=2):
        """
        Simple rule-based blink detector over the ear_buffer.
        Returns list of timestamps (seconds) where blinks were detected.
        """
        ears = np.array(self.ear_buffer)
        below = ears < ear_thresh
        blinks = []
        i = 0
        while i < len(below):
            if below[i]:
                j = i
                while j < len(below) and below[j]:
                    j += 1
                if (j - i) >= min_consec_frames:
                    mid = i + (j - i) // 2
                    if mid < len(self.times):
                        blinks.append(self.times[mid])
                i = j
            else:
                i += 1
        return blinks
