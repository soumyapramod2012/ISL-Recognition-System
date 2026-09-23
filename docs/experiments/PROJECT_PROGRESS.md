Indian Sign Language Recognition — Project Progress

1. Project Objective

Build an Indian Sign Language (ISL) recognition system that can:

recognize signs from video/webcam input;

use the official AI4Bharat INCLUDE dataset;

support a substantially larger vocabulary than the original 24-class model;

eventually accept an uploaded sign-language video and predict the recognized sign.

2. Git / Development Safety

Active development branch: feature/v1.9-temporal-stability

main must remain untouched.

src/inference/realtime.py must remain untouched.

Experimental work is kept separate from production files.

Avoid git add .; stage only explicitly approved files.

Latest pushed checkpoint: 4703c66 Add test quality and Clock Laptop diagnostics

Feature branch is pushed to GitHub and tracks origin/feature/v1.9-temporal-stability.

3. Environments

Legacy environment: venv_mp_legacy

MediaPipe 0.10.21

TensorFlow/Keras 2.15

258-dimensional MediaPipe landmark representation

Per frame:

Pose: 33 landmarks × 4 = 132

Left hand: 21 landmarks × 3 = 63

Right hand: 21 landmarks × 3 = 63

Total: 258 features

Sequence length: 60 frames

4. Original 24-Class Baseline

Existing dataset:

24 classes

391 samples after expanded Bird addition

Train: 225

Validation: 80

Test: 81

Original baseline test accuracy: 64.20%

Legacy-compatible model parity:

81 test samples

Compatible accuracy: 64.20%

Maximum probability difference: approximately 3e-7

Predicted classes identical: 100%

5. INCLUDE Dataset Integration

Categories already downloaded/processed:

Animals

Clothes

Electronics

Greetings

Home

Jobs

Means_of_Transportation

Places

Seasons

Downloader supports:

category-wise size reporting;

resume/retry;

expected-size verification;

HTTP Range / 206 handling;

restart when the server ignores Range requests.

Official downloader reported:

15 categories

44 ZIP files

approximately 52.85 GB total dataset

Remaining categories:

Adjectives

Colours

Days_and_Time

People

Pronouns

Society

6. Legacy Landmark Processing

Legacy extractor:

MediaPipe Holistic 0.10.21

model complexity 1

smooth landmarks enabled

detection/tracking confidence 0.5

INCLUDE landmark processing:

1,968 videos successfully processed

258 features per frame

no NaN/Inf values

stray Extra files corrected

legitimate Ex. Monsoon class retained

7. Generalized Dataset

Generalized legacy dataset:

114 classes

2,359 samples

Stratified split:

Train: 1,373

Validation: 479

Test: 507

Validated arrays:

X_train: (1373, 60, 258)

X_val: (479, 60, 258)

X_test: (507, 60, 258)

valid samples: 2,359

invalid samples: 0

8. Generalized 114-Class Model

Architecture:

Input (60,258)

Bidirectional LSTM 128, return sequences

Dropout 0.30

Bidirectional LSTM 64

Dropout 0.30

Dense 64 ReLU

Dropout 0.20

Dense 114 Softmax

Training:

Adam, learning rate 0.001

Sparse categorical crossentropy

Balanced class weights

Early stopping

Results:

Best validation loss: 1.1832

Best validation accuracy: 73.49%, epoch 76

Test accuracy: 72.98%

9. Generalized Model Analysis

Overall:

Top-1: 72.98%

Top-3: 88.95%

Top-5: 93.29%

Original 24-class subset within generalized test set:

Top-1: 85.89%

Top-3: 96.32%

Top-5: 97.55%

This is not treated as a strict apples-to-apples improvement over 64.20% because the split and training setup differ.

Stored targeted Clock/Laptop test:

Clock: 6/6 correct

Laptop: 6/6 correct

No Clock/Laptop cross-confusion

10. Webcam Investigation — Final Finding

A separate experimental realtime implementation was tested without modifying src/inference/realtime.py.

Recorded webcam Clock:

66 frames

258 features/frame

generalized model repeatedly predicted Truck

Completed diagnostics included temporal windows, gesture segments, mirror/hand-swap tests, landmark similarity, model feature-space comparison, hand detection, and pose/hand distribution comparison.

Conclusion:

Evidence indicates a webcam-domain / landmark-representation mismatch, especially in hand landmarks.

Stored Clock samples are recognized correctly by the generalized model.

Do not modify the production realtime file or retrain solely for this webcam recording.

11. Experiments Not Adopted

Experimental work not adopted into production includes:

ensemble experiment;

alternative training strategies;

geometry augmentation;

motion-feature experiments;

pose/hand fusion;

targeted retraining;

temporary model conversion/export scripts.

12. Current Git Checkpoint

Branch:
feature/v1.9-temporal-stability

Latest commit:
4703c66 Add test quality and Clock Laptop diagnostics

The branch has been successfully pushed to GitHub.

Untracked experimental scripts are intentionally not part of this checkpoint.

13. Next Planned Stage

Continue downloading remaining INCLUDE categories.

Verify the complete dataset.

Process the complete dataset into the compatible 258-feature legacy landmark representation.

Prepare the expanded vocabulary.

Decide whether the existing generalized model is sufficient or a new full-vocabulary model is required.

Build/test uploaded-video prediction.

Validate predictions on videos not used for training.

14. Documentation Rule

Record important milestone results, final model metrics, dataset statistics, major decisions, and Git checkpoints.

Do not record every temporary traceback or exploratory console output unless it becomes relevant to a final technical decision.