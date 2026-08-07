from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"

FACE_MODEL = MODELS_DIR / "face_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"