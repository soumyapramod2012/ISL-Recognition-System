import json, sys
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.train_dataset import TrainDataset

DATASET = "dataset/processed/generalized_filtered_legacy"
SPLIT = Path("dataset/split_generalized_filtered_legacy.json")
BASE = Path("saved_models/isl_lstm_generalized_filtered_136_legacy.keras")
TARGET = Path("saved_models/isl_lstm_generalized_filtered_136_targeted.keras")
BASE_MAP = Path("outputs/generalized_filtered_136_label_mapping.json")
TARGET_MAP = Path("outputs/generalized_filtered_136_targeted_label_mapping.json")
OUT = Path("outputs/generalized_filtered_136_targeted_comparison.txt")
OUT_JSON = Path("outputs/generalized_filtered_136_targeted_comparison.json")

ORIGINAL24 = {
"1. Dog","2. Cat","3. Fish","4. Bird","5. Cow","51. Clock","52. Lamp",
"53. Fan","54. Cell phone","55. Computer","56. Laptop","57. Screen",
"58. Camera","59. Television","6. Mouse","60. Radio","61. Summer",
"62. Spring","63. Winter","64. Fall","65. Season","7. Horse","8. Animal",
"Ex. Monsoon"
}

def calc(p, y):
    pred = np.argmax(p, axis=1)
    top3 = np.argsort(p, axis=1)[:, -3:]
    top5 = np.argsort(p, axis=1)[:, -5:]
    return (
        float(np.mean(pred == y)),
        float(np.mean([y0 in r for y0,r in zip(y,top3)])),
        float(np.mean([y0 in r for y0,r in zip(y,top5)]))
    )

def mapping(path):
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Support both mapping formats:
    # 1. {"class name": index}
    # 2. {"index": "class name"}
    if all(str(k).isdigit() for k in raw.keys()):
        return {int(k): v for k, v in raw.items()}

    return {int(v): k for k, v in raw.items()}

def main():
    print("="*80)
    print("READ-ONLY TARGETED 136 VS BASELINE 136 COMPARISON")
    print("="*80)

    ds = TrainDataset(DATASET)
    ds.split_file = SPLIT
    Xtr, Xv, Xt, ytr, yv, yt, enc = ds.build()
    print("Test:", Xt.shape)

    bm, tm = mapping(BASE_MAP), mapping(TARGET_MAP)
    if bm != tm:
        raise RuntimeError("Label mappings differ.")

    print("Loading baseline...")
    bmodel = tf.keras.models.load_model(BASE, compile=False)
    print("Loading targeted...")
    tmodel = tf.keras.models.load_model(TARGET, compile=False)

    print("Predicting baseline...")
    bp = bmodel.predict(Xt, verbose=0)
    print("Predicting targeted...")
    tp = tmodel.predict(Xt, verbose=0)

    b = calc(bp, yt)
    t = calc(tp, yt)

    idx24 = {i for i,n in bm.items() if n in ORIGINAL24}
    mask = np.array([int(y) in idx24 for y in yt])
    y24 = yt[mask]
    b24, t24 = calc(bp[mask], y24), calc(tp[mask], y24)

    rows=[]
    for i,n in sorted(bm.items(), key=lambda x:x[1]):
        if n not in ORIGINAL24: continue
        m = y24 == i
        if not np.any(m): continue
        old = float(np.mean(np.argmax(bp[mask][m],axis=1)==i))
        new = float(np.mean(np.argmax(tp[mask][m],axis=1)==i))
        rows.append((n,int(m.sum()),old,new,new-old))

    result = {
      "test_samples":len(yt),
      "original24_test_samples":int(mask.sum()),
      "overall":{"baseline":b,"targeted":t,
                 "change":[t[i]-b[i] for i in range(3)]},
      "original24":{"baseline":b24,"targeted":t24,
                    "change":[t24[i]-b24[i] for i in range(3)]},
      "per_class":[
        {"class":n,"test_samples":num,"baseline_top1":old,
         "targeted_top1":new,"change_top1":delta}
        for n,num,old,new,delta in rows
      ]
    }

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT_JSON.write_text(json.dumps(result,indent=2),encoding="utf-8")

    lines=[
      "="*80,
      "TARGETED 136-CLASS MODEL COMPARISON","="*80,"",
      f"Test samples       : {len(yt)}",
      f"Original-24 samples: {int(mask.sum())}","",
      "OVERALL 136-CLASS","-"*80,
      f"Baseline Top-1: {b[0]*100:.4f}%",
      f"Targeted Top-1: {t[0]*100:.4f}%",
      f"Change: {(t[0]-b[0])*100:+.4f} pp","",
      f"Baseline Top-3: {b[1]*100:.4f}%",
      f"Targeted Top-3: {t[1]*100:.4f}%",
      f"Change: {(t[1]-b[1])*100:+.4f} pp","",
      f"Baseline Top-5: {b[2]*100:.4f}%",
      f"Targeted Top-5: {t[2]*100:.4f}%",
      f"Change: {(t[2]-b[2])*100:+.4f} pp","",
      "ORIGINAL 24-CLASS SUBSET","-"*80,
      f"Baseline Top-1: {b24[0]*100:.4f}%",
      f"Targeted Top-1: {t24[0]*100:.4f}%",
      f"Change: {(t24[0]-b24[0])*100:+.4f} pp","",
      f"Baseline Top-3: {b24[1]*100:.4f}%",
      f"Targeted Top-3: {t24[1]*100:.4f}%",
      f"Change: {(t24[1]-b24[1])*100:+.4f} pp","",
      f"Baseline Top-5: {b24[2]*100:.4f}%",
      f"Targeted Top-5: {t24[2]*100:.4f}%",
      f"Change: {(t24[2]-b24[2])*100:+.4f} pp","",
      "ORIGINAL 24 CLASS TOP-1","-"*80,
      f"{'Class':25s}{'N':>4s}{'Base':>9s}{'Target':>9s}{'Change':>10s}"
    ]
    for n,num,old,new,delta in rows:
        lines.append(f"{n:25s}{num:4d}{old*100:8.2f}%{new*100:8.2f}%{delta*100:+9.2f} pp")
    lines += ["",f"TXT: {OUT}",f"JSON: {OUT_JSON}"]
    OUT.write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__":
    main()
