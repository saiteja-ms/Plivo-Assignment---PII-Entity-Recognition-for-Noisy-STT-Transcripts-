# src/run_full_experiment.py
"""
FAST tuning experiment (CPU-friendly, <15 minutes total)

Uses the ORIGINAL train.py and model.py with NO modifications.

Models tried:
- nreimers/MiniLM-L6-H384-uncased (2 trials)
- distilbert-base-uncased (2 trials)

Saves:
- tune_logs/best_config.json
- tune_logs/best_dev_pred.json
- final_submission.json
"""

import os
import json
import subprocess
from skopt import forest_minimize
from skopt.space import Real, Categorical

# -----------------------------------------------------
# Very small, CPU-friendly experiment
# -----------------------------------------------------
MODEL_BUDGET = {
    "nreimers/MiniLM-L6-H384-uncased": 2,
    "distilbert-base-uncased": 2,
    "google/mobilebert-uncased": 2,
    "microsoft/MiniLM-L6-v2": 2
}

MODELS = list(MODEL_BUDGET.keys())

# -----------------------------------------------------
# Paths
# -----------------------------------------------------
TRAIN = "data/train.jsonl"
DEV = "data/dev.jsonl"
TEST = "data/test.jsonl"

ROOT = "tune_logs"
os.makedirs(ROOT, exist_ok=True)

BEST_CONFIG = os.path.join(ROOT, "best_config.json")
BEST_PRED = os.path.join(ROOT, "best_dev_pred.json")

FINAL_SUBMISSION = "final_submission.json"

# -----------------------------------------------------
# Hyperparameter Search Space
# -----------------------------------------------------
space = [
    Real(2e-5, 5e-5, prior="log-uniform", name="lr"),
    Categorical([8, 16], name="batch_size"),
    Categorical([3], name="epochs"),   # FIXED to 3 for speed
]

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------
def run(cmd):
    print(f"\n[RUN] {cmd}\n")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(r.stderr)
    return r.stdout + r.stderr


def extract_f1(text):
    for line in text.split("\n"):
        if "Macro-F1" in line:
            return float(line.split(":")[1])
    return 0.0


def objective_factory(model_name, best):
    """Returns a function that trains, predicts, evaluates, updates best."""
    def objective(params):
        lr, batch_size, epochs = params

        out_dir = os.path.join(
            ROOT, f"{model_name.replace('/', '_')}_lr{lr}_bs{batch_size}"
        )
        os.makedirs(out_dir, exist_ok=True)

        # TRAIN
        run(
            f"python src/train.py "
            f"--model_name {model_name} "
            f"--train {TRAIN} --dev {DEV} "
            f"--out_dir {out_dir} "
            f"--batch_size {batch_size} --epochs {epochs} "
            f"--lr {lr} --max_length 256 --device cpu"
        )

        # PREDICT
        pred_file = os.path.join(out_dir, "dev_pred.json")
        run(
            f"python src/predict.py "
            f"--model_dir {out_dir} "
            f"--input {DEV} "
            f"--output {pred_file} "
            f"--device cpu"
        )

        # EVAL
        out = run(
            f"python src/eval_span_f1.py "
            f"--gold {DEV} "
            f"--pred {pred_file}"
        )
        f1 = extract_f1(out)

        print(f"→ F1 = {f1:.3f}")

        # UPDATE BEST
        if f1 > best["f1"]:
            print("\n🔥 NEW BEST MODEL FOUND!\n")
            best.update({
                "model": model_name,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "f1": f1,
                "dev_pred_file": pred_file
            })

            # Save best_config.json
            with open(BEST_CONFIG, "w") as f:
                json.dump(best, f, indent=2)

            # Save best_dev_pred.json
            with open(pred_file, "r") as src, open(BEST_PRED, "w") as dst:
                dst.write(src.read())

        return -f1

    return objective

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
def main():
    best = {"model": None, "lr": None, "batch_size": None,
            "epochs": None, "f1": -1.0, "dev_pred_file": None}

    # TUNE BOTH MODELS
    for model in MODELS:
        budget = MODEL_BUDGET[model]
        if budget == 0:
            continue

        print("\n=====================================")
        print(f"Tuning: {model} (Trials={budget})")
        print("=====================================\n")

        forest_minimize(
            func=objective_factory(model, best),
            dimensions=space,
            n_calls=budget,
            n_initial_points=budget,
            random_state=42,
            verbose=True
        )

    print("\n=============== BEST MODEL ===============")
    print(json.dumps(best, indent=2))

    # FINAL RETRAIN
    final_out = os.path.join(
        ROOT,
        f"FINAL_{best['model'].replace('/', '_')}_lr{best['lr']}_bs{best['batch_size']}"
    )
    os.makedirs(final_out, exist_ok=True)

    run(
        f"python src/train.py "
        f"--model_name {best['model']} "
        f"--train {TRAIN} --dev {DEV} "
        f"--out_dir {final_out} "
        f"--batch_size {best['batch_size']} "
        f"--epochs {best['epochs']} "
        f"--lr {best['lr']} "
        f"--max_length 256 --device cpu"
    )

    # FINAL TEST PREDICTIONS
    final_test_file = os.path.join(final_out, "test_pred.json")
    run(
        f"python src/predict.py "
        f"--model_dir {final_out} "
        f"--input {TEST} "
        f"--output {final_test_file} "
        f"--device cpu"
    )

    # LATENCY
    out = run(
        f"python src/measure_latency.py "
        f"--model_dir {final_out} "
        f"--input {DEV} "
        f"--runs 50 --device cpu"
    )

    latency = {
        "p50": None,
        "p95": None
    }
    for line in out.split("\n"):
        if "p50:" in line:
            latency["p50"] = float(line.split(":")[1].replace("ms", ""))
        if "p95:" in line:
            latency["p95"] = float(line.split(":")[1].replace("ms", ""))

    # FINAL SUBMISSION
    payload = {
        "best_model": best["model"],
        "hyperparameters": {
            "lr": best["lr"],
            "batch_size": best["batch_size"],
            "epochs": best["epochs"]
        },
        "dev_f1": best["f1"],
        "latency": latency,
        "test_predictions": final_test_file
    }

    with open(FINAL_SUBMISSION, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n================ FINAL SUBMISSION CREATED ================")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
