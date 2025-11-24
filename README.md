# PII Named Entity Recognition (NER) System

This project implements a lightweight, high-performance Named Entity Recognition (NER) model for detecting **Personally Identifiable Information (PII)**.  
The goal is to achieve strong span-level accuracy **while maintaining a latency requirement of p95 ≤ 20 ms on CPU**.

## Project Overview

The system identifies the following PII entity types:

- **EMAIL**
- **PHONE**
- **PERSON_NAME**
- **DATE**
- **CITY**
- **LOCATION**
- **CREDIT_CARD**

All labels follow the **BIO tagging scheme**, and span-level predictions are returned in JSON format.

## Repository Structure

```
pii_ner_assignment/
│
├── data/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
│
├── src/
│   ├── dataset.py
│   ├── labels.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── eval_span_f1.py
│   ├── measure_latency.py
│   └── run_full_experiment.py
│
├── out_minilm/
├── tune_logs/
│
├── test_pred.json
└── metrics_report.json
```

## 🔧 Models Evaluated

| Model Name | Checkpoint | Purpose |
|-----------|------------|---------|
| **DistilBERT** | distilbert-base-uncased | Baseline |
| **MiniLM-L6-H384** | nreimers/MiniLM-L6-H384-uncased | ⭐ Best model |
| **MobileBERT** | google/mobilebert-uncased | Mobile-optimized |
| **BERT-Tiny** | prajjwal1/bert-tiny | Ultra-fast |
| **MiniLM-L6-v2** | microsoft/MiniLM-L6-v2 | Compact general model |

## 🏆 Best Model Summary

**Best Model:** `nreimers/MiniLM-L6-H384-uncased`

Training Hyperparameters:
- learning rate: 3e-5
- batch size: 16
- epochs: 5
- max_length: 256
- device: CPU

## 📊 Final Dev Set Metrics

Macro-F1: **0.473**  
PII-only F1: **0.464**  
Non-PII F1: **0.497**

Per-Entity F1:
- CITY: 0.392
- CREDIT_CARD: 0.262
- DATE: 0.500
- EMAIL: 0.440
- LOCATION: 0.667
- PERSON_NAME: 0.632
- PHONE: 0.421

## ⚡ Latency (CPU)

p50: **11.79 ms**  
p95: **14.64 ms**

✔ Meets requirement: p95 ≤ 20 ms

## 📥 Submission Files

- **test_pred.json**
- **metrics_report.json**

## 🛠️ Train

```
python src/train.py --model_name nreimers/MiniLM-L6-H384-uncased --train data/train.jsonl --dev data/dev.jsonl --out_dir out_minilm --batch_size 16 --epochs 5 --lr 3e-5 --device cpu
```

## 🧪 Predict

```
python src/predict.py --model_dir out_minilm --input data/test.jsonl --output test_pred.json --device cpu
```

## 📈 Evaluate

```
python src/eval_span_f1.py --gold data/dev.jsonl --pred out_minilm/dev_pred.json
```

## ⚡ Latency

```
python src/measure_latency.py --model_dir out_minilm --input data/dev.jsonl --runs 50 --device cpu
```

## 🧪 Optional Tuning

```
python src/run_full_experiment.py
```

## 📌 Conclusion

The **MiniLM-L6-H384** model achieves:
- Macro-F1 = 0.473
- Very low latency (14.64 ms p95)
- Strong PII extraction performance

Meeting all assignment requirements.
