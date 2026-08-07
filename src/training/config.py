"""
Training Configuration
----------------------
All configurable parameters for model training are defined here.
"""

# ==========================
# Dataset
# ==========================

SEQUENCE_LENGTH = 60
LANDMARK_SIZE = 258

# ==========================
# Dataset Split
# ==========================

TEST_SIZE = 0.20
RANDOM_STATE = 42

# ==========================
# Model Architecture
# ==========================

LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64

DENSE_UNITS = 64

DROPOUT_1 = 0.30
DROPOUT_2 = 0.30
DROPOUT_3 = 0.20

# ==========================
# Training
# ==========================

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

# ==========================
# Checkpoints
# ==========================

MODEL_NAME = "isl_lstm.keras"

CHECKPOINT_DIR = "saved_models"

LOG_DIR = "logs"

LABEL_MAPPING = "outputs/label_mapping.json"

# ==========================
# Saving
# ==========================

MODEL_DIR = "saved_models"

BEST_MODEL = "saved_models/isl_lstm.keras"

MODEL_SUMMARY = "saved_models/model_summary.txt"

MODEL_PLOT = "saved_models/model.png"

TRAINING_HISTORY = "saved_models/history.json"

ACCURACY_PLOT = "saved_models/accuracy.png"

LOSS_PLOT = "saved_models/loss.png"

# ==========================
# TensorBoard
# ==========================

TENSORBOARD_LOGS = "logs"

VALIDATION_SIZE = 0.25

EXPERIMENT_LOG = "outputs/experiments.csv"
PROJECT_VERSION = "v1.1.0"

EXPERIMENTS_DIR = "outputs/experiments"

