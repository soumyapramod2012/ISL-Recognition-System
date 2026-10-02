"""
Indian Sign Language Recognition - 136-Class V4.1 UI

Tkinter desktop UI built from the existing uploaded UI and upgraded to
the tested V4.1 inference pipeline.

V4.1 behavior:
- frozen targeted 136-class model
- 258 features / frame
- frame-position-preserving gesture buffer
- short no-hand gaps retained and interpolated
- minimum 30-frame gestures
- SequenceGenerator handles <60 by padding and >=60 by resampling
- automatic gesture finalization after 8 consecutive no-hand frames
- finalizes an active gesture on STOP, video end, or window close
- supports webcam and uploaded-video modes
- keeps CSV/JSON/annotated-video result saving

This file does NOT modify src/inference/realtime.py or the trained model.
"""

import sys
import time
import shutil
import csv
import json
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.realtime_136_v4_1 import (
    MODEL_PATH, SEQUENCE_LENGTH, FEATURES, CONFIDENCE_THRESHOLD,
    PREDICTION_INTERVAL, MIN_GESTURE_FRAMES, MAX_INTERPOLATION_GAP,
    GESTURE_END_GAP, MAX_GESTURE_FRAMES,
    LegacyLandmarkExtractor, hand_detected, confidence_quality, load_labels,
)
from src.training.hand_landmark_interpolator import HandLandmarkInterpolator
from src.training.landmark_normalizer import LandmarkNormalizer
from src.training.sequence_generator import SequenceGenerator
from src.inference.temporal_stabilizer import TemporalStabilizer

# V4.1 temporal stabilizer configuration.
# These are the same values used by the tested real-time V4.1 pipeline:
# history_size=5, initial_min_votes=2, transition_min_votes=3.
STABILIZER_HISTORY = 5
STABILIZER_INITIAL_VOTES = 2
STABILIZER_TRANSITION_VOTES = 3

WINDOW_TITLE = "Indian Sign Language Recognition - 136 Classes | V4.1"
VIDEO_DIR = PROJECT_ROOT / "outputs" / "external_tests"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


class ISLApplication:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1220x800")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#111827")
        self.mode = "webcam"
        self.running = False
        self.camera = None
        self.video_capture = None
        self.video_path = None
        self.video_fps = 30.0
        self.video_total = 0
        self.video_frame = 0
        self.after_id = None

        self.model = None
        self.labels = None
        self.extractor = None
        self.interpolator = None
        self.normalizer = None
        self.generator = None
        self.stabilizer = None

        # V4.1 gesture state:
        # retain EVERY frame while a gesture is active, including
        # short no-hand gaps at their original frame positions.
        self.buffer = []
        self.consecutive_no_hand = 0
        self.gesture_state = "WAITING"
        self.gesture_number = 0
        self.prediction_counter = 0
        self.raw_label = "Uncertain"
        self.raw_conf = 0.0
        self.stable_label = "Uncertain"
        self.stable_conf = 0.0
        self.inference_ms = 0.0
        self.fps = 0.0
        self.fps_start = time.perf_counter()
        self.fps_frames = 0
        self.history = []
        self.last_history = None
        self.last_history_time = 0.0

        self.result_dir = PROJECT_ROOT / "outputs" / "predictions"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.session_results = []
        self.session_started = None
        self.result_video_writer = None
        self.result_video_path = None
        self.save_csv_var = tk.BooleanVar(value=True)
        self.save_json_var = tk.BooleanVar(value=True)
        self.save_video_var = tk.BooleanVar(value=True)

        self.url_var = tk.StringVar()
        self.file_var = tk.StringVar(value="No video selected")
        self.status_var = tk.StringVar(value="LOADING MODEL...")
        self.hand_var = tk.StringVar(value="NO")
        self.buffer_var = tk.StringVar(value="0 / 60")
        self.raw_var = tk.StringVar(value="Uncertain")
        self.raw_conf_var = tk.StringVar(value="0.0%")
        self.progress_var = tk.StringVar(value="LIVE")
        self.fps_var = tk.StringVar(value="0.0")
        self.inference_var = tk.StringVar(value="0.0 ms")
        self.quality_var = tk.StringVar(value="UNCERTAIN")
        self.gesture_var = tk.StringVar(value="#0")

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.initialize)

    # ---------------- UI ----------------
    def build_ui(self):
        header = tk.Frame(self.root, bg="#111827")
        header.pack(fill="x", padx=20, pady=(14, 6))
        tk.Label(header, text="INDIAN SIGN LANGUAGE", font=("Segoe UI", 24, "bold"),
                 fg="white", bg="#111827").pack(side="left")
        tk.Label(header, text="136-CLASS AI RECOGNITION", font=("Segoe UI", 13, "bold"),
                 fg="#60a5fa", bg="#111827").pack(side="left", padx=14, pady=(8, 0))

        source = tk.Frame(self.root, bg="#1f2937", highlightbackground="#374151", highlightthickness=1)
        source.pack(fill="x", padx=20, pady=(0, 8))
        tk.Label(source, text="PREDICTION SOURCE", font=("Segoe UI", 10, "bold"),
                 fg="#9ca3af", bg="#1f2937").pack(side="left", padx=(14, 8), pady=9)
        self.webcam_btn = tk.Button(source, text="WEBCAM", command=lambda: self.select_mode("webcam"),
                                    font=("Segoe UI", 10, "bold"), fg="white", bg="#2563eb",
                                    relief="flat", padx=16, pady=7, cursor="hand2")
        self.webcam_btn.pack(side="left", padx=4)
        self.video_btn = tk.Button(source, text="VIDEO", command=lambda: self.select_mode("video"),
                                   font=("Segoe UI", 10, "bold"), fg="white", bg="#4b5563",
                                   relief="flat", padx=16, pady=7, cursor="hand2")
        self.video_btn.pack(side="left", padx=4)
        self.browse_btn = tk.Button(source, text="BROWSE SAVED VIDEO", command=self.browse_video,
                                    font=("Segoe UI", 9, "bold"), fg="white", bg="#059669",
                                    relief="flat", padx=12, pady=7, cursor="hand2", state="disabled")
        self.browse_btn.pack(side="left", padx=(14, 4))
        self.url_entry = tk.Entry(source, textvariable=self.url_var, width=38, font=("Segoe UI", 9),
                                  bg="#111827", fg="white", insertbackground="white", relief="flat",
                                  state="disabled")
        self.url_entry.pack(side="left", padx=(8, 4), ipady=6)
        self.url_btn = tk.Button(source, text="LOAD VIDEO LINK", command=self.load_url,
                                 font=("Segoe UI", 9, "bold"), fg="white", bg="#7c3aed",
                                 relief="flat", padx=12, pady=7, cursor="hand2", state="disabled")
        self.url_btn.pack(side="left", padx=4)

        main = tk.Frame(self.root, bg="#111827")
        main.pack(fill="both", expand=True, padx=20, pady=4)
        card = tk.Frame(main, bg="#1f2937", highlightbackground="#374151", highlightthickness=1)
        card.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.video_label = tk.Label(card, text="Camera preview will appear here", font=("Segoe UI", 16),
                                    fg="#9ca3af", bg="#0b1220")
        self.video_label.pack(fill="both", expand=True, padx=12, pady=12)
        info = tk.Frame(card, bg="#1f2937")
        info.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(info, text="VIDEO:", font=("Segoe UI", 8, "bold"), fg="#9ca3af", bg="#1f2937").pack(side="left")
        tk.Label(info, textvariable=self.file_var, font=("Segoe UI", 8), fg="#d1d5db", bg="#1f2937",
                 anchor="w").pack(side="left", padx=6, fill="x", expand=True)
        tk.Label(info, textvariable=self.progress_var, font=("Segoe UI", 8, "bold"),
                 fg="#60a5fa", bg="#1f2937").pack(side="right")

        side = tk.Frame(main, bg="#1f2937", width=350, highlightbackground="#374151", highlightthickness=1)
        side.pack(side="right", fill="y", padx=(10, 0)); side.pack_propagate(False)
        tk.Label(side, text="RECOGNIZED SIGN", font=("Segoe UI", 11, "bold"),
                 fg="#9ca3af", bg="#1f2937").pack(pady=(16, 3))
        self.sign_label = tk.Label(side, text="Uncertain", font=("Segoe UI", 24, "bold"),
                                   fg="#fbbf24", bg="#1f2937", wraplength=315, justify="center")
        self.sign_label.pack(padx=15, pady=(0, 7))
        self.conf_label = tk.Label(side, text="0.0%", font=("Segoe UI", 18, "bold"), fg="white", bg="#1f2937")
        self.conf_label.pack()
        self.quality_label = tk.Label(side, textvariable=self.quality_var, font=("Segoe UI", 10, "bold"),
                                      fg="#f87171", bg="#1f2937")
        self.quality_label.pack(pady=(1, 9))
        for name, var in [("HAND", self.hand_var), ("BUFFER", self.buffer_var), ("STATUS", self.status_var),
                          ("GESTURE", self.gesture_var),
                          ("RAW PREDICTION", self.raw_var), ("RAW CONFIDENCE", self.raw_conf_var),
                          ("VIDEO FRAME", self.progress_var), ("FPS", self.fps_var), ("INFERENCE", self.inference_var)]:
            self.metric(side, name, var)
        tk.Label(side, text="RECENT PREDICTIONS", font=("Segoe UI", 10, "bold"),
                 fg="#9ca3af", bg="#1f2937").pack(anchor="w", padx=18, pady=(9, 4))
        self.history_text = tk.Text(side, height=5, font=("Consolas", 9), fg="#d1d5db", bg="#111827",
                                    relief="flat", state="disabled", wrap="word")
        self.history_text.pack(fill="x", padx=18, pady=(0, 8))

        result_bar = tk.Frame(self.root, bg="#1f2937",
                                  highlightbackground="#374151", highlightthickness=1)
        result_bar.pack(fill="x", padx=20, pady=(2, 5))
        tk.Label(result_bar, text="SAVE RESULTS", font=("Segoe UI", 9, "bold"),
                 fg="#9ca3af", bg="#1f2937").pack(side="left", padx=(12, 8), pady=7)
        for text, var in [("CSV", self.save_csv_var), ("JSON", self.save_json_var),
                          ("ANNOTATED VIDEO", self.save_video_var)]:
            tk.Checkbutton(result_bar, text=text, variable=var,
                           font=("Segoe UI", 8, "bold"), fg="white", bg="#1f2937",
                           activebackground="#1f2937", activeforeground="white",
                           selectcolor="#111827").pack(side="left", padx=3)
        self.save_btn = tk.Button(result_bar, text="SAVE RESULTS", command=self.save_results,
                                  font=("Segoe UI", 8, "bold"), fg="white", bg="#0891b2",
                                  activebackground="#0e7490", relief="flat", padx=12, pady=6)
        self.save_btn.pack(side="left", padx=(10, 5))
        self.result_status_var = tk.StringVar(value="No results recorded")
        tk.Label(result_bar, textvariable=self.result_status_var, font=("Segoe UI", 8),
                 fg="#9ca3af", bg="#1f2937").pack(side="right", padx=12)

        controls = tk.Frame(self.root, bg="#111827")
        controls.pack(fill="x", padx=20, pady=(5, 13))
        self.start_btn = tk.Button(controls, text="START", command=self.start_source, font=("Segoe UI", 10, "bold"),
                                   fg="white", bg="#2563eb", relief="flat", padx=18, pady=9, state="disabled")
        self.start_btn.pack(side="left", padx=(0, 7))
        self.stop_btn = tk.Button(controls, text="STOP", command=self.stop_source, font=("Segoe UI", 10, "bold"),
                                  fg="white", bg="#dc2626", relief="flat", padx=18, pady=9, state="disabled")
        self.stop_btn.pack(side="left", padx=7)
        tk.Button(controls, text="CLEAR / RESET", command=self.reset_prediction, font=("Segoe UI", 10, "bold"),
                  fg="white", bg="#4b5563", relief="flat", padx=18, pady=9).pack(side="left", padx=7)
        tk.Label(controls, text="136 classes • 258 features • V4.1 gap-tolerant pipeline",
                 font=("Segoe UI", 8), fg="#9ca3af", bg="#111827").pack(side="right", pady=10)

    def metric(self, parent, name, var):
        row = tk.Frame(parent, bg="#1f2937"); row.pack(fill="x", padx=18, pady=1)
        tk.Label(row, text=name, font=("Segoe UI", 8, "bold"), fg="#9ca3af", bg="#1f2937",
                 width=17, anchor="w").pack(side="left")
        tk.Label(row, textvariable=var, font=("Segoe UI", 9, "bold"), fg="white", bg="#1f2937").pack(side="right")

    # ---------------- Initialization ----------------
    def initialize(self):
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            if self.model.input_shape[-2:] != (SEQUENCE_LENGTH, FEATURES):
                raise RuntimeError(f"Unexpected model input shape: {self.model.input_shape}")
            if self.model.output_shape[-1] != 136:
                raise RuntimeError(f"Expected 136 outputs, got {self.model.output_shape}")
            self.labels = load_labels()
            self.extractor = LegacyLandmarkExtractor()
            self.interpolator = HandLandmarkInterpolator(max_gap=5)
            self.normalizer = LandmarkNormalizer()
            self.generator = SequenceGenerator()
            self.make_stabilizer()
            self.status_var.set("READY")
            self.start_btn.config(state="normal")
        except Exception as e:
            self.status_var.set("ERROR")
            messagebox.showerror("Initialization Error", str(e))

    def make_stabilizer(self):
        self.stabilizer = TemporalStabilizer(history_size=STABILIZER_HISTORY,
                                             initial_min_votes=STABILIZER_INITIAL_VOTES,
                                             transition_min_votes=STABILIZER_TRANSITION_VOTES)

    # ---------------- Result recording ----------------

    def start_result_session(self):
        self.session_results = []
        self.session_started = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.result_video_writer is not None:
            self.result_video_writer.release()
        self.result_video_writer = None
        self.result_video_path = None
        self.result_status_var.set("Recording predictions...")

    def record_prediction(self):
        if self.session_started is None:
            self.start_result_session()
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Webcam" if self.mode == "webcam" else "Uploaded Video",
            "video": self.video_path.name if self.video_path else "Webcam",
            "frame": self.video_frame if self.mode == "video" else "",
            "raw_prediction": self.raw_label,
            "raw_confidence": round(float(self.raw_conf), 6),
            "stable_prediction": self.stable_label,
            "stable_confidence": round(float(self.stable_conf), 6),
            "quality": confidence_quality(self.stable_conf, self.stable_label != "Uncertain"),
            "inference_ms": round(float(self.inference_ms), 3),
        }
        self.session_results.append(row)
        self.result_status_var.set(f"{len(self.session_results)} prediction records")

    def ensure_result_video_writer(self, frame):
        if self.mode != "video" or not self.save_video_var.get() or self.result_video_writer is not None:
            return
        stamp=time.strftime("%Y%m%d_%H%M%S")
        self.result_video_path=self.result_dir/f"isl_prediction_{stamp}.mp4"
        h,w=frame.shape[:2]
        writer=cv2.VideoWriter(str(self.result_video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                               self.video_fps if self.video_fps > 0 else 30.0, (w,h))
        if writer.isOpened():
            self.result_video_writer=writer
        else:
            writer.release(); self.result_video_path=None

    def write_annotated_frame(self, frame, has_hand, status):
        if self.mode != "video" or not self.save_video_var.get():
            return
        self.ensure_result_video_writer(frame)
        if self.result_video_writer is None:
            return
        annotated=frame.copy()
        cv2.rectangle(annotated,(10,10),(annotated.shape[1]-10,116),(17,24,39),-1)
        cv2.putText(annotated,"INDIAN SIGN LANGUAGE - 136 CLASSES",(24,38),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        cv2.putText(annotated,f"Prediction: {self.stable_label}",(24,65),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        cv2.putText(annotated,
                    f"Confidence: {self.stable_conf*100:.1f}%  Status: {status}  "
                    f"Frame: {self.video_frame}/{self.video_total}",(24,91),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(96,165,250),1)
        cv2.putText(annotated,f"Hand: {'YES' if has_hand else 'NO'}  Buffer: {len(self.buffer)}/60",
                    (24,108),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
        self.result_video_writer.write(annotated)

    def save_results(self):
        if not self.session_results:
            messagebox.showinfo("Save Results","No prediction results have been recorded yet.")
            return
        stamp=time.strftime("%Y%m%d_%H%M%S")
        base=self.result_dir/f"isl_results_{stamp}"
        saved=[]
        if self.save_csv_var.get():
            p=base.with_suffix(".csv")
            fields=list(self.session_results[0].keys())
            with p.open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(self.session_results)
            saved.append(str(p))
        if self.save_json_var.get():
            p=base.with_suffix(".json")
            payload={
                "project":"Indian Sign Language Recognition",
                "model":"isl_lstm_generalized_filtered_136_targeted.keras",
                "classes":136,"features_per_frame":FEATURES,
                "sequence_length":SEQUENCE_LENGTH,"confidence_threshold":CONFIDENCE_THRESHOLD,
                "session_started":self.session_started,
                "source":"Webcam" if self.mode=="webcam" else "Uploaded Video",
                "video":self.video_path.name if self.video_path else "Webcam",
                "records":self.session_results
            }
            with p.open("w",encoding="utf-8") as f: json.dump(payload,f,indent=2)
            saved.append(str(p))
        if self.result_video_path is not None:
            saved.append(str(self.result_video_path))
        self.result_status_var.set(f"Saved {len(self.session_results)} prediction records")
        messagebox.showinfo("Results Saved","Results saved successfully:\n\n"+"\n".join(saved))

    # ---------------- Mode / video loading ----------------
    def select_mode(self, mode):
        if self.running: self.stop_source()
        self.mode = mode
        if mode == "webcam":
            self.webcam_btn.config(bg="#2563eb"); self.video_btn.config(bg="#4b5563")
            self.browse_btn.config(state="disabled"); self.url_btn.config(state="disabled"); self.url_entry.config(state="disabled")
            self.file_var.set("No video selected"); self.progress_var.set("LIVE"); self.status_var.set("READY")
        else:
            self.webcam_btn.config(bg="#4b5563"); self.video_btn.config(bg="#2563eb")
            self.browse_btn.config(state="normal"); self.url_btn.config(state="normal"); self.url_entry.config(state="normal")
            self.progress_var.set("0 / 0"); self.status_var.set("SELECT VIDEO")
        self.reset_prediction(clear_status=False)

    def browse_video(self):
        path = filedialog.askopenfilename(title="Select a sign-language video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")])
        if path: self.set_video_path(Path(path))

    def set_video_path(self, path):
        if not path.exists():
            messagebox.showerror("Video Error", f"File not found:\n{path}"); return
        self.video_path = path
        self.file_var.set(path.name)
        self.status_var.set("VIDEO READY")
        self.reset_prediction(clear_status=False)
        self.start_btn.config(state="normal")

    def load_url(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("Video Link", "Paste a video or YouTube link first."); return
        self.start_btn.config(state="disabled"); self.browse_btn.config(state="disabled"); self.url_btn.config(state="disabled")
        self.status_var.set("DOWNLOADING VIDEO...")
        threading.Thread(target=self.download_url, args=(url,), daemon=True).start()

    def download_url(self, url):
        # Always use the Python environment running this application.
        python_exe = sys.executable

        try:
            subprocess.run(
                [python_exe, "-m", "yt_dlp", "--version"],
                check=True, capture_output=True, text=True, timeout=30
            )
        except Exception as exc:
            self.root.after(
                0,
                lambda: self.url_error(
                    "yt-dlp is not available in this Python environment.\n\n"
                    "Run in PowerShell:\n"
                    "python -m pip install -U \"yt-dlp[default]\"\n\n"
                    f"Details: {exc}"
                )
            )
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        template = str(VIDEO_DIR / f"isl_url_{stamp}.%(ext)s")

        # Deno is required for current YouTube extraction. If FFmpeg is
        # installed, allow yt-dlp to merge the best separate streams.
        ffmpeg_available = shutil.which("ffmpeg") is not None
        format_selector = (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
            if ffmpeg_available else "best[ext=mp4]/best"
        )

        cmd = [
            python_exe, "-m", "yt_dlp",
            "--js-runtimes", "deno",
            "--no-playlist",
            "-f", format_selector,
            "--merge-output-format", "mp4",
            "-o", template, url
        ]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                self.root.after(
                    0,
                    lambda: self.url_error(
                        "Download failed:\n\n" + (r.stderr or r.stdout or "Unknown error")[-2500:]
                    )
                )
                return

            files = sorted(
                [p for p in VIDEO_DIR.glob(f"isl_url_{stamp}.*")
                 if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi"}],
                key=lambda p: p.stat().st_mtime, reverse=True
            )
            if not files:
                self.root.after(0, lambda: self.url_error(
                    "Download completed but no video file was found."
                ))
                return

            self.root.after(0, lambda p=files[0]: self.url_success(p))
        except subprocess.TimeoutExpired:
            self.root.after(0, lambda: self.url_error(
                "Video download timed out after 15 minutes."
            ))
        except Exception as exc:
            self.root.after(0, lambda: self.url_error(str(exc)))

    def url_success(self, path):
        self.video_path = path; self.file_var.set(path.name); self.status_var.set("VIDEO READY")
        self.browse_btn.config(state="normal"); self.url_btn.config(state="normal"); self.start_btn.config(state="normal")
        messagebox.showinfo("Video Ready", f"Video downloaded successfully.\n\n{path.name}")

    def url_error(self, message):
        self.status_var.set("URL ERROR"); self.browse_btn.config(state="normal"); self.url_btn.config(state="normal"); self.start_btn.config(state="normal")
        messagebox.showerror("Video Link Error", message)

    # ---------------- Start / stop ----------------
    def start_source(self):
        if self.running: return
        if self.mode == "webcam": self.start_webcam()
        else: self.start_video()

    def start_webcam(self):
        # Start every webcam session with a clean V4.1 state.
        self.reset_prediction(clear_status=False)
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera.release(); self.camera = None; messagebox.showerror("Camera Error", "Could not open webcam."); return
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800); self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.running = True; self.start_result_session(); self.start_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self.status_var.set("STARTING..."); self.fps_start = time.perf_counter(); self.fps_frames = 0; self.update_frame()

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("Video", "Browse for a saved video or load a video link first."); return
        # Start every uploaded-video session with a clean V4.1 state.
        self.reset_prediction(clear_status=False)
        self.video_capture = cv2.VideoCapture(str(self.video_path))
        if not self.video_capture.isOpened():
            self.video_capture.release(); self.video_capture = None; messagebox.showerror("Video Error", "Could not open video."); return
        self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_total = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_frame = 0; self.running = True
        self.start_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self.status_var.set("PLAYING VIDEO..."); self.fps_start = time.perf_counter(); self.fps_frames = 0; self.update_frame()

    def stop_source(self):
        # Match V4.1: finish an active gesture before stopping.
        if self.running:
            try:
                self.finalize_active_gesture()
            except Exception as exc:
                print(f"Stop finalization warning: {exc}")

        self.running = False
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        if self.result_video_writer is not None:
            self.result_video_writer.release()
            self.result_video_writer = None
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")
        self.status_var.set("STOPPED")

    # ---------------- Prediction ----------------
    # ---------------- V4.1 Prediction Engine ----------------
    def reset_prediction(self, clear_status=True):
        """Reset the complete V4.1 gesture state."""
        self.buffer = []
        self.consecutive_no_hand = 0
        self.gesture_state = "WAITING"
        self.gesture_number = 0
        self.prediction_counter = 0
        self.raw_label = "Uncertain"
        self.raw_conf = 0.0
        self.stable_label = "Uncertain"
        self.stable_conf = 0.0
        self.inference_ms = 0.0
        self.make_stabilizer()
        self.history.clear()
        self.last_history = None
        self.last_history_time = 0.0

        self.raw_var.set("Uncertain")
        self.raw_conf_var.set("0.0%")
        self.buffer_var.set("0 frames")
        self.gesture_var.set(f"#{self.gesture_number}")
        self.inference_var.set("0.0 ms")
        self.quality_var.set("UNCERTAIN")
        self.update_history()

        if clear_status:
            self.status_var.set("WARMING UP" if self.running else "READY")
        self.update_sign()

    def _classify_buffer(self, final=False):
        """
        V4.1 classification:
        - keep original frame positions
        - interpolate short hand gaps
        - normalize
        - SequenceGenerator resamples >=60 or zero-pads <60
        """
        if len(self.buffer) < MIN_GESTURE_FRAMES:
            return False

        raw_data = np.asarray(self.buffer, dtype=np.float32)

        # Match V4.1 live behavior: prevent an unlimited live
        # classification span, but finalization uses the complete buffer.
        if not final and len(raw_data) > MAX_GESTURE_FRAMES:
            raw_data = raw_data[-MAX_GESTURE_FRAMES:]

        interpolated = self.interpolator.interpolate(raw_data)
        normalized = self.normalizer.normalize(interpolated)
        sequence = self.generator.generate(normalized)

        if sequence.shape != (SEQUENCE_LENGTH, FEATURES):
            raise RuntimeError(
                f"Unexpected sequence shape: {sequence.shape}"
            )

        start = time.perf_counter()
        probabilities = self.model.predict(
            np.expand_dims(sequence, axis=0),
            verbose=0
        )[0]
        self.inference_ms = (time.perf_counter() - start) * 1000.0

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        label = (
            self.labels[index]
            if confidence >= CONFIDENCE_THRESHOLD
            else "Uncertain"
        )

        self.raw_label = label
        self.raw_conf = confidence
        self.stable_label, self.stable_conf = self.stabilizer.update(
            label,
            confidence,
            hand_detected=True,
        )

        self.record_prediction()
        self.add_history()
        return True

    def finalize_active_gesture(self):
        """
        Finalize the active gesture before resetting it.

        Used for:
        - V4.1 automatic end-gap finalization
        - STOP
        - end of uploaded video
        - application close
        """
        if (
            self.gesture_state != "ACTIVE"
            or len(self.buffer) < MIN_GESTURE_FRAMES
        ):
            return False

        try:
            ok = self._classify_buffer(final=True)
        except Exception as exc:
            self.result_status_var.set("Finalization error")
            print(f"\nGesture finalization warning: {exc}")
            ok = False

        # Reset only after final classification.
        self.buffer = []
        self.consecutive_no_hand = 0
        self.prediction_counter = 0
        self.gesture_state = "WAITING"
        self.gesture_var.set(f"#{self.gesture_number}")
        return ok

    def process_frame(self, frame):
        """
        Exact V4.1 gesture-state behavior adapted to the Tkinter UI.

        Every frame is retained during an active gesture, even if no
        hand is detected. Short gaps are interpolated later instead of
        immediately resetting the gesture.
        """
        landmarks = self.extractor.extract(frame)

        if landmarks.shape != (FEATURES,):
            raise RuntimeError(
                f"Unexpected landmark shape: {landmarks.shape}"
            )

        has_hand = hand_detected(landmarks)

        if self.gesture_state == "WAITING":
            if has_hand:
                self.gesture_number += 1
                self.gesture_state = "ACTIVE"
                self.buffer = []
                self.consecutive_no_hand = 0
                self.prediction_counter = 0
                self.raw_label = "Uncertain"
                self.raw_conf = 0.0
                self.buffer.append(landmarks.copy())
        else:
            # CRITICAL V4.1 behavior: retain EVERY frame.
            self.buffer.append(landmarks.copy())

            if has_hand:
                self.consecutive_no_hand = 0
            else:
                self.consecutive_no_hand += 1

            self.prediction_counter += 1

            # Longer no-hand gap = gesture ended.
            if self.consecutive_no_hand >= GESTURE_END_GAP:
                self.finalize_active_gesture()

            # Live partial prediction while hand is present.
            elif (
                has_hand
                and len(self.buffer) >= MIN_GESTURE_FRAMES
                and self.prediction_counter >= PREDICTION_INTERVAL
            ):
                try:
                    self._classify_buffer(final=False)
                except Exception as exc:
                    print(f"\nInference warning: {exc}")
                self.prediction_counter = 0

        return has_hand

    # ---------------- Display / loop ----------------
    def update_history(self):
        self.history_text.config(state="normal"); self.history_text.delete("1.0", "end")
        if not self.history: self.history_text.insert("end", "No recognized signs yet.")
        else:
            for label, conf, stamp in self.history: self.history_text.insert("end", f"{stamp}  {label}  ({conf*100:.1f}%)\n")
        self.history_text.config(state="disabled")

    def update_sign(self):
        quality = confidence_quality(self.stable_conf, self.stable_label != "Uncertain")
        self.sign_label.config(text=self.stable_label, fg="#34d399" if quality == "HIGH" else "#fbbf24")
        self.conf_label.config(text=f"{self.stable_conf*100:.1f}%")
        self.quality_var.set(quality); self.quality_label.config(fg="#34d399" if quality == "HIGH" else "#fbbf24" if quality == "MEDIUM" else "#f87171")

    def update_frame(self):
        if not self.running: return
        cap = self.camera if self.mode == "webcam" else self.video_capture
        ok, frame = cap.read()
        if not ok:
            if self.mode == "video": self.finish_video()
            else: self.status_var.set("CAMERA READ ERROR")
            return
        if self.mode == "webcam": frame = cv2.flip(frame, 1)
        else: self.video_frame += 1
        self.fps_frames += 1
        try: has_hand = self.process_frame(frame)
        except Exception as e:
            self.stop_source(); messagebox.showerror("Inference Error", str(e)); return
        elapsed = time.perf_counter() - self.fps_start
        if elapsed >= 1.0: self.fps = self.fps_frames / elapsed; self.fps_frames = 0; self.fps_start = time.perf_counter()
        n = len(self.buffer)

        if self.gesture_state == "WAITING":
            status = "NO HAND / WAITING"
        elif self.consecutive_no_hand > 0:
            status = f"GAP {self.consecutive_no_hand}/{GESTURE_END_GAP}"
        elif n < MIN_GESTURE_FRAMES:
            status = "COLLECTING"
        else:
            status = confidence_quality(self.stable_conf, self.stable_label)

        self.hand_var.set("YES" if has_hand else "NO")
        self.buffer_var.set(f"{n} frames")
        self.status_var.set(status)
        self.gesture_var.set(f"#{self.gesture_number}")
        self.raw_var.set(self.raw_label)
        self.raw_conf_var.set(f"{self.raw_conf*100:.1f}%")
        self.fps_var.set(f"{self.fps:.1f}")
        self.inference_var.set(f"{self.inference_ms:.1f} ms")
        self.progress_var.set(f"{self.video_frame} / {self.video_total}" if self.mode == "video" else "LIVE"); self.update_sign()
        self.write_annotated_frame(frame, has_hand, status)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(rgb, (10,10), (rgb.shape[1]-10,80), (17,24,39), -1)
        cv2.putText(rgb, "INDIAN SIGN LANGUAGE - 136 CLASSES", (24,38), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        source = "WEBCAM" if self.mode == "webcam" else "UPLOADED VIDEO"
        cv2.putText(
            rgb,
            f"{source}  Hand: {'YES' if has_hand else 'NO'}  "
            f"Buffer: {n}  Status: {status}  Gesture: #{self.gesture_number}",
            (24,66), cv2.FONT_HERSHEY_SIMPLEX, .43, (96,165,250), 1
        )
        image = Image.fromarray(rgb); image.thumbnail((max(self.video_label.winfo_width()-20,640), max(self.video_label.winfo_height()-20,480)), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image); self.video_label.configure(image=photo, text=""); self.video_label.image = photo
        delay = max(10, int(1000/max(self.video_fps,1))) if self.mode == "video" else 10
        self.after_id = self.root.after(delay, self.update_frame)

    def finish_video(self):
        # If the uploaded clip ends during an active gesture, classify it
        # using the same V4.1 finalization path before resetting.
        if self.gesture_state == "ACTIVE":
            try:
                self.finalize_active_gesture()
            except Exception as exc:
                print(f"Video-end finalization warning: {exc}")

        self.running = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        if self.result_video_writer is not None:
            self.result_video_writer.release()
            self.result_video_writer = None
        self.stop_btn.config(state="disabled")
        self.start_btn.config(state="normal")
        self.status_var.set("VIDEO COMPLETE")
        self.result_status_var.set(
            f"{len(self.session_results)} prediction records ready to save"
        )
        self.progress_var.set(f"{self.video_total} / {self.video_total}")

    def close(self):
        if self.running and self.gesture_state == "ACTIVE":
            try:
                self.finalize_active_gesture()
            except Exception:
                pass

        self.running = False
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
        if self.camera is not None:
            self.camera.release()
        if self.video_capture is not None:
            self.video_capture.release()
        if self.result_video_writer is not None:
            self.result_video_writer.release()
        if self.extractor is not None:
            try:
                self.extractor.close()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk(); ISLApplication(root); root.mainloop()


if __name__ == "__main__": main()
