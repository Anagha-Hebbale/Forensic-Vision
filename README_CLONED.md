# NexJam Model-Quality Clone

This cloned workspace contains a stricter training/evaluation flow to reduce wrong predictions.

## What was improved

- `train_model.py`
  - data augmentation
  - class weights for imbalance
  - early stopping + LR reduction
  - fine-tuning of upper MobileNetV2 layers
  - saves model metadata (`models/model_metadata.json`) with:
    - class order
    - threshold
    - image size
    - validation metrics
- `eval_model.py`
  - confusion matrix
  - precision/recall/F1 report
  - ROC AUC (if both classes present)
  - saves `models/eval_report.json`
- `data_audit.py`
  - per-split class counts
  - corrupted image detection
  - imbalance warning
  - saves `models/data_audit_report.json`
- `heatmap.py`, `app.py`, `app_cool.py`
  - inference reads class mapping + threshold + image size from metadata
  - avoids hardcoded label mapping at prediction time
- `predict_cnn.py`
  - metadata-aware CLI predictor for the CNN model

## Run order

1. Audit dataset quality
```bash
python3 data_audit.py
```

2. Retrain model
```bash
python3 train_model.py
```

3. Evaluate model
```bash
python3 eval_model.py --data_dir dataset/valid
```
Optional test split:
```bash
python3 eval_model.py --data_dir dataset/test --report_path models/eval_report_test.json
```

4. Run UI
```bash
streamlit run app_cool.py
```

5. CLI sanity check
```bash
python3 predict_cnn.py test.jpeg
```

## Required dataset layout

```text
dataset/
  train/
    FAKE/
    REAL/
  valid/
    FAKE/
    REAL/
  test/
    FAKE/
    REAL/
```
