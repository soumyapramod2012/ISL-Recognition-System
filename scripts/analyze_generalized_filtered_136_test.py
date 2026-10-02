import json
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from src.training.train_dataset import TrainDataset

DATASET_ROOT = Path("dataset/processed/generalized_filtered_legacy")
SPLIT_FILE = Path("dataset/split_generalized_filtered_legacy.json")
MODEL_FILE = Path("saved_models/isl_lstm_generalized_filtered_136_legacy.keras")
MAPPING_FILE = Path("outputs/generalized_filtered_136_label_mapping.json")

OUT_JSON = Path("outputs/generalized_filtered_136_test_analysis.json")
OUT_PER_CLASS = Path("outputs/generalized_filtered_136_per_class.csv")
OUT_CONFUSIONS = Path("outputs/generalized_filtered_136_confusions.csv")

ORIGINAL_24 = {
    "1. Dog","2. Cat","3. Fish","4. Bird","5. Cow","51. Clock",
    "52. Lamp","53. Fan","54. Cell phone","55. Computer","56. Laptop",
    "57. Screen","58. Camera","59. Television","6. Mouse","60. Radio",
    "61. Summer","62. Spring","63. Winter","64. Fall","65. Season",
    "7. Horse","8. Animal","Ex. Monsoon"
}
TARGETS = ["47. Red","58. Son","51. Clock","56. Laptop","48. Hello"]

def top_k_accuracy(y_true, probs, k):
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean(np.any(topk == y_true[:, None], axis=1)))

def main():
    for p in [DATASET_ROOT, SPLIT_FILE, MODEL_FILE, MAPPING_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    with open(SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)
    with open(MAPPING_FILE, encoding="utf-8") as f:
        mapping = json.load(f)

    if mapping and all(str(k).isdigit() for k in mapping):
        index_to_label = {int(k): v for k, v in mapping.items()}
        label_to_index = {v: k for k, v in index_to_label.items()}
    else:
        label_to_index = {k: int(v) for k, v in mapping.items()}
        index_to_label = {v: k for k, v in label_to_index.items()}

    model = tf.keras.models.load_model(MODEL_FILE, compile=False)

    dataset = TrainDataset(str(DATASET_ROOT))
    dataset.split_file = SPLIT_FILE

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        encoder,
    ) = dataset.build()
    probs = model.predict(X_test, verbose=0)
    pred = np.argmax(probs, axis=1)

    overall = {
        "test_samples": int(len(y_test)),
        "classes": int(len(label_to_index)),
        "top1_accuracy_percent": round(100 * top_k_accuracy(y_test, probs, 1), 4),
        "top3_accuracy_percent": round(100 * top_k_accuracy(y_test, probs, 3), 4),
        "top5_accuracy_percent": round(100 * top_k_accuracy(y_test, probs, 5), 4),
    }

    original_indices = {label_to_index[x] for x in ORIGINAL_24 if x in label_to_index}
    mask = np.array([y in original_indices for y in y_test])
    original24 = {
        "classes_present": len(original_indices),
        "test_samples": int(mask.sum()),
        "top1_accuracy_percent": round(100 * top_k_accuracy(y_test[mask], probs[mask], 1), 4) if mask.any() else None,
        "top3_accuracy_percent": round(100 * top_k_accuracy(y_test[mask], probs[mask], 3), 4) if mask.any() else None,
        "top5_accuracy_percent": round(100 * top_k_accuracy(y_test[mask], probs[mask], 5), 4) if mask.any() else None,
    }

    rows = []
    for idx, label in sorted(index_to_label.items()):
        m = y_test == idx
        if not m.any():
            continue
        cp = probs[m]
        yt = y_test[m]
        rows.append({
            "index": idx, "label": label, "test_samples": int(m.sum()),
            "top1_percent": round(100*np.mean(pred[m] == idx), 4),
            "top3_percent": round(100*np.mean(np.any(np.argsort(cp, axis=1)[:, -3:] == yt[:,None], axis=1)), 4),
            "top5_percent": round(100*np.mean(np.any(np.argsort(cp, axis=1)[:, -5:] == yt[:,None], axis=1)), 4),
            "avg_true_confidence_percent": round(100*np.mean(cp[:,idx]), 4),
            "avg_pred_confidence_percent": round(100*np.mean(np.max(cp, axis=1)), 4),
        })
    per_class = pd.DataFrame(rows)
    per_class.to_csv(OUT_PER_CLASS, index=False)

    confusion = {}
    for t, p in zip(y_test, pred):
        if t != p:
            confusion[(int(t), int(p))] = confusion.get((int(t), int(p)), 0) + 1
    conf_rows = [{
        "true_index": t, "true_label": index_to_label[t],
        "pred_index": p, "pred_label": index_to_label[p], "count": n
    } for (t,p), n in sorted(confusion.items(), key=lambda x: (-x[1], x[0]))]
    pd.DataFrame(conf_rows).to_csv(OUT_CONFUSIONS, index=False)

    targets, absent = {}, []
    for label in TARGETS:
        if label not in label_to_index:
            absent.append(label)
        else:
            r = per_class[per_class.label == label]
            targets[label] = r.iloc[0].to_dict() if not r.empty else {"test_samples": 0}

    report = {
        "model": str(MODEL_FILE), "dataset": str(DATASET_ROOT),
        "split": str(SPLIT_FILE), "overall": overall,
        "original_24_subset": original24, "target_classes": targets,
        "target_classes_absent_from_136_mapping": absent,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*70)
    print("FILTERED 136-CLASS TEST ANALYSIS")
    print("="*70)
    print(f"Test samples : {overall['test_samples']}")
    print(f"Classes      : {overall['classes']}")
    print(f"Top-1        : {overall['top1_accuracy_percent']:.2f}%")
    print(f"Top-3        : {overall['top3_accuracy_percent']:.2f}%")
    print(f"Top-5        : {overall['top5_accuracy_percent']:.2f}%")
    print("\nOriginal 24-class subset")
    print(f"Test samples : {original24['test_samples']}")
    print(f"Top-1        : {original24['top1_accuracy_percent']:.2f}%")
    print(f"Top-3        : {original24['top3_accuracy_percent']:.2f}%")
    print(f"Top-5        : {original24['top5_accuracy_percent']:.2f}%")
    print("\nTarget classes present:")
    for label, r in targets.items():
        print(f"{label:20s} n={int(r.get('test_samples',0)):2d}  "
              f"Top1={r.get('top1_percent',0):6.2f}%  "
              f"Top3={r.get('top3_percent',0):6.2f}%  "
              f"Top5={r.get('top5_percent',0):6.2f}%")
    if absent:
        print("\nTargets NOT present in 136 mapping:")
        for x in absent:
            print("-", x)
    print("\nTop 20 confusion pairs:")
    for r in conf_rows[:20]:
        print(f"{r['true_label']} -> {r['pred_label']} : {r['count']}")
    print("\nAnalysis completed successfully.")

if __name__ == "__main__":
    main()
