import csv
import random
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }


def rss_bytes():
    return psutil.Process().memory_info().rss


def measure_training(model_factory, x_train, y_train):
    warmup = model_factory()
    warmup.fit(x_train, y_train)

    times_ms = []
    memory_deltas = []
    last_model = None

    for _ in range(5):
        model = model_factory()
        before_mem = rss_bytes()
        start = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = (time.perf_counter() - start) * 1000
        after_mem = rss_bytes()

        times_ms.append(elapsed)
        memory_deltas.append(max(0, after_mem - before_mem))
        last_model = model

    return last_model, statistics.median(times_ms), max(memory_deltas)


def measure_inference(model, sample):
    model.predict(sample)  # warm-up
    times_ms = []
    memory_deltas = []

    for _ in range(100):
        before_mem = rss_bytes()
        start = time.perf_counter()
        model.predict(sample)
        elapsed = (time.perf_counter() - start) * 1000
        after_mem = rss_bytes()
        times_ms.append(elapsed)
        memory_deltas.append(max(0, after_mem - before_mem))

    return statistics.median(times_ms), max(memory_deltas)


def deployment_fit(latency_ms, model_kb, memory_bytes):
    memory_mb = memory_bytes / (1024 * 1024)
    budgets = {
        "Cloud": memory_mb >= 1024 and latency_ms <= 100 and model_kb <= 500 * 1024,
        "Edge": 256 <= memory_mb <= 1024 and latency_ms <= 50 and model_kb <= 50 * 1024,
        "Mobile": 64 <= memory_mb <= 256 and latency_ms <= 20 and model_kb <= 10 * 1024,
        "TinyML": memory_bytes <= 256 * 1024 and latency_ms <= 10 and model_kb <= 100,
    }
    return budgets


def main():
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=42
    )

    factories = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    }

    accuracy_rows = []
    measurement_rows = []
    deployment_rows = []

    for name, factory in factories.items():
        model, training_ms, training_memory = measure_training(factory, x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        inference_ms, inference_memory = measure_inference(model, x_test[:1])

        model_path = RESULTS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        model_bytes = model_path.stat().st_size
        model_kb = model_bytes / 1024
        peak_memory = max(training_memory, inference_memory)

        accuracy_rows.append([name, f"{accuracy:.4f}"])
        measurement_rows.append([
            name,
            f"{training_ms:.6f}",
            f"{inference_ms:.6f}",
            model_bytes,
            f"{model_kb:.3f}",
            peak_memory,
            f"{peak_memory / (1024 * 1024):.3f}",
        ])

        fits = deployment_fit(inference_ms, model_kb, peak_memory)
        for platform_name, fits_budget in fits.items():
            deployment_rows.append([name, platform_name, "YES" if fits_budget else "NO"])

    with (RESULTS_DIR / "baseline_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_accuracy"])
        writer.writerows(accuracy_rows)

    with (RESULTS_DIR / "measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "training_median_ms", "inference_median_ms",
            "model_bytes", "model_kb", "peak_memory_bytes", "peak_memory_mb"
        ])
        writer.writerows(measurement_rows)

    with (RESULTS_DIR / "deployment_fit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "platform", "fits_measured_budget"])
        writer.writerows(deployment_rows)

    print("Lab 01 completed.")
    print(f"Results saved to: {RESULTS_DIR}")
    print("Open baseline_accuracy.csv and measurements.csv and use the measured values in report.md.")


if __name__ == "__main__":
    main()
