Indian Sign Language Recognition — Project Progress Checkpoint

Checkpoint date: 2026-10-02
Git branch: feature/v1.9-temporal-stability
Current committed V4.1 checkpoint: 9d5bb1d — Add V4.1 136-class real-time UI

1. Current project state

The project has reached a working end-to-end 136-class AI application.

Working components

136-class targeted LSTM model

MediaPipe Holistic 0.10.21

258 features per frame

60-frame model input

V4.1 frame-position-preserving gesture buffer

Short no-hand-gap tolerance

Landmark interpolation

Landmark normalization

SequenceGenerator

Temporal stabilization

Tkinter desktop UI

Webcam inference

Uploaded-video inference

Prediction confidence display

Recent prediction history

CSV/JSON result saving

Annotated-video result saving

2. Frozen production model

Model:

saved_models/isl_lstm_generalized_filtered_136_targeted.keras

Label mapping:

outputs/generalized_filtered_136_label_mapping.json

Architecture/input:

Classes: 136

Features/frame: 258

Sequence length: 60 frames

Confidence threshold: 0.50

The targeted model is frozen for final validation. No retraining is planned unless final validation identifies a reproducible blocking defect.

3. V4.1 inference behavior

The important correction made in V4.1 was to stop treating every temporary hand-detection loss as the end of a gesture.

Current behavior:

Camera
  -> MediaPipe Holistic
  -> 258 features/frame
  -> retain original frame positions
  -> short-gap interpolation
  -> normalization
  -> SequenceGenerator
  -> 60-frame model sequence
  -> frozen 136-class model
  -> temporal stabilization
  -> prediction

Gesture settings:

Minimum gesture length: 30 frames

Maximum interpolation gap: 5 frames

Gesture end gap: 8 frames

Maximum live gesture buffer: 180 frames

Stabilizer history: 5

Initial minimum votes: 2

Transition minimum votes: 3

The tested V4.1 behavior demonstrated that short hand-loss gaps do not immediately reset the gesture.

4. UI checkpoint

Working application:

src/inference/isl_app_136_v4_1.py

Inference engine:

src/inference/realtime_136_v4_1.py

Both were committed in:

9d5bb1d Add V4.1 136-class real-time UI

The V4.1 UI was tested successfully with the webcam.

5. Dataset/model evidence already completed

The final filtered 136-class dataset contains:

Train: 1873

Validation: 610

Test: 610

Total valid samples: 3093

Previously completed model comparison established the targeted model as the frozen production model.

The final targeted model achieved approximately:

73.44% Top-1 on the 136-class test set

89.02% Top-3

93.93% Top-5

The original 24-class subset was also evaluated separately.

6. Real-video/inference diagnostics already completed

Known-video testing established the importance of preserving original frame positions and allowing short hand-detection gaps.

Examples already tested:

Season

Clock

Laptop

Summer

Multiple webcam gestures

The V4/V4.1 recordings demonstrated:

gesture buffering

short-gap handling

live prediction

temporal stabilization

multiple consecutive gestures

automatic gesture finalization

7. Git state

The V4.1 application/inference checkpoint is already committed and pushed.

Commit:

9d5bb1d Add V4.1 136-class real-time UI

The following files are currently known to have additional uncommitted changes and should NOT be mixed into the V4.1 UI checkpoint without review:

dataset/split_generalized_legacy.json
src/preprocessing/prepare_generalized_legacy_dataset.py

There are also many untracked experimental/audit/training scripts.

8. IMPORTANT: 136-class reproducibility code still needs preservation

Before final validation, the project must preserve the source code used to build/evaluate the 136-class pipeline.

At minimum, review and preserve these categories:

Dataset preparation

src/preprocessing/create_filtered_generalized_dataset.py

any final 136-class split-generation script actually used for the frozen dataset

the final filtered split file:
dataset/split_generalized_filtered_legacy.json

136-class training

src/training/train_generalized_filtered_136_legacy.py

src/training/train_generalized_filtered_136_targeted.py

src/training/train_generalized_filtered_136_targeted_nosdweight.py

136-class evaluation/comparison

src/training/compare_targeted_vs_baseline_136.py

scripts/analyze_generalized_filtered_136_test.py

relevant 136-class audit scripts

136-class inference/testing

src/inference/realtime_136_v4_1.py

src/inference/isl_app_136_v4_1.py

src/inference/video_136.py

src/inference/video_136_gaptest.py

src/inference/video_136_v4_framepreserve_test.py

src/inference/evaluate_sign_clip_136.py

Experimental scripts should be clearly separated from production/reproducibility scripts rather than blindly committing every untracked file.

9. Next stage — final validation

After the reproducibility code is preserved:

Verify that the final model file and 136-class label mapping are present and documented.

Run controlled final validation on known signs.

Record expected label, predicted label, confidence and result.

Separate model test-set accuracy from real-video demonstration accuracy.

Prepare the final report.

Prepare the final presentation.

Prepare the demonstration/viva material.

10. Freeze rule

Do not modify:

src/inference/realtime.py

the frozen targeted 136-class model

the final label mapping

unless a reproducible blocking defect is discovered.

The immediate objective is now reproducibility preservation followed by final validation, not further model experimentation.