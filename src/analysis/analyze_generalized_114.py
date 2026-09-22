import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from src.training.train_dataset import TrainDataset

DATASET_PATH = Path('dataset/processed/generalized_landmarks_legacy')
SPLIT_PATH = Path('dataset/split_generalized_legacy_stratified.json')
MODEL_PATH = Path('saved_models/isl_lstm_generalized_114_legacy.keras')
OUTPUT_DIR = Path('outputs/generalized_114_analysis')

ORIGINAL_24 = [
    '1. Dog','2. Cat','3. Fish','4. Bird','5. Cow','51. Clock',
    '52. Lamp','53. Fan','54. Cell phone','55. Computer','56. Laptop',
    '57. Screen','58. Camera','59. Television','6. Mouse','60. Radio',
    '61. Summer','62. Spring','63. Winter','64. Fall','65. Season',
    'Ex. Monsoon','7. Horse','8. Animal'
]

def top_k_accuracy(y_true, probs, k):
    top = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean(np.any(top == y_true[:, None], axis=1)))

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = TrainDataset(DATASET_PATH)
    dataset.split_file = SPLIT_PATH
    _, _, X_test, _, _, y_test, encoder = dataset.build()
    class_names = list(encoder.classes)
    if len(class_names) != 114:
        raise RuntimeError(f'Expected 114 classes, found {len(class_names)}')
    print('Loading model...')
    model = tf.keras.models.load_model(MODEL_PATH)
    print('Running predictions...')
    probs = model.predict(X_test, batch_size=32, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    top1 = top_k_accuracy(y_test, probs, 1)
    top3 = top_k_accuracy(y_test, probs, 3)
    top5 = top_k_accuracy(y_test, probs, 5)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(114))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(OUTPUT_DIR/'confusion_matrix.csv', encoding='utf-8-sig')
    rows=[]
    for i,name in enumerate(class_names):
        actual=int(np.sum(y_test==i)); correct=int(cm[i,i]); other=cm[i].copy(); other[i]=0
        j=int(np.argmax(other)) if other.sum() else -1
        rows.append({'class_id':i,'class':name,'test_samples':actual,'correct':correct,'incorrect':actual-correct,'accuracy_percent':100*correct/actual if actual else 0,'most_common_confusion':class_names[j] if j>=0 else '','confusion_count':int(other[j]) if j>=0 else 0})
    per_class=pd.DataFrame(rows).sort_values('accuracy_percent')
    per_class.to_csv(OUTPUT_DIR/'per_class_accuracy.csv',index=False,encoding='utf-8-sig')
    pairs=[]
    for a in range(114):
        for p in range(114):
            if a!=p and cm[a,p]>0: pairs.append({'actual':class_names[a],'predicted':class_names[p],'count':int(cm[a,p])})
    pairs_df=pd.DataFrame(pairs).sort_values('count',ascending=False)
    pairs_df.to_csv(OUTPUT_DIR/'most_confused_pairs.csv',index=False,encoding='utf-8-sig')
    ids=[class_names.index(x) for x in ORIGINAL_24 if x in class_names]
    mask=np.isin(y_test,ids)
    orig={'test_samples':int(mask.sum()),'top1':top_k_accuracy(y_test[mask],probs[mask],1),'top3':top_k_accuracy(y_test[mask],probs[mask],3),'top5':top_k_accuracy(y_test[mask],probs[mask],5),'missing_classes':[x for x in ORIGINAL_24 if x not in class_names]}
    (OUTPUT_DIR/'original_24_class_performance.json').write_text(json.dumps(orig,indent=2),encoding='utf-8')
    clock=class_names.index('51. Clock'); laptop=class_names.index('56. Laptop'); clock_mask=y_test==clock; laptop_mask=y_test==laptop
    clock_result={'clock_test_samples':int(clock_mask.sum()),'clock_accuracy':float(np.mean(y_pred[clock_mask]==clock)),'clock_to_laptop':int(cm[clock,laptop]),'laptop_test_samples':int(laptop_mask.sum()),'laptop_accuracy':float(np.mean(y_pred[laptop_mask]==laptop)),'laptop_to_clock':int(cm[laptop,clock])}
    (OUTPUT_DIR/'clock_laptop_analysis.json').write_text(json.dumps(clock_result,indent=2),encoding='utf-8')
    report=classification_report(y_test,y_pred,labels=np.arange(114),target_names=class_names,output_dict=True,zero_division=0)
    (OUTPUT_DIR/'classification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    summary={'test_samples':len(y_test),'classes':114,'top1_accuracy':top1,'top3_accuracy':top3,'top5_accuracy':top5,'original_24':orig,'clock_laptop':clock_result}
    (OUTPUT_DIR/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\n'+'='*60); print('OVERALL TEST PERFORMANCE'); print('='*60)
    print(f'Top-1: {top1*100:.2f}%'); print(f'Top-3: {top3*100:.2f}%'); print(f'Top-5: {top5*100:.2f}%')
    print('\n'+'='*60); print('ORIGINAL 24-CLASS SUBSET'); print('='*60)
    print(f"Samples: {orig['test_samples']}"); print(f"Top-1: {orig['top1']*100:.2f}%"); print(f"Top-3: {orig['top3']*100:.2f}%"); print(f"Top-5: {orig['top5']*100:.2f}%")
    print('\n'+'='*60); print('CLOCK / LAPTOP'); print('='*60)
    print(f"Clock accuracy: {clock_result['clock_accuracy']*100:.2f}%"); print(f"Clock -> Laptop: {clock_result['clock_to_laptop']}"); print(f"Laptop accuracy: {clock_result['laptop_accuracy']*100:.2f}%"); print(f"Laptop -> Clock: {clock_result['laptop_to_clock']}")
    print('\nLOWEST-ACCURACY CLASSES'); print(per_class[['class','test_samples','correct','accuracy_percent','most_common_confusion','confusion_count']].head(15).to_string(index=False))
    print('\nMOST COMMON CONFUSIONS'); print(pairs_df.head(20).to_string(index=False))
    print('\nAnalysis complete:',OUTPUT_DIR)

if __name__=='__main__': main()
