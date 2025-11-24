import json
import subprocess

BEST_DIR = "tune_logs/nreimers_MiniLM-L6-H384-uncased_lr3.455735963606937e-05_bs8"
DEV = "data/dev.jsonl"
PRED = BEST_DIR + "/dev_pred.json"

# Run eval
eval_out = subprocess.run(
    f"python src/eval_span_f1.py --gold {DEV} --pred {PRED}",
    shell=True, capture_output=True, text=True
).stdout

# Run latency
lat_out = subprocess.run(
    f"python src/measure_latency.py --model_dir {BEST_DIR} --input {DEV} --runs 50 --device cpu",
    shell=True, capture_output=True, text=True
).stdout

print("\n===== FULL METRICS =====")
print(eval_out)
print(lat_out)
