# Lab 1: Environment and First System Measurements

## 1. Goal

The goal of this laboratory was to create a reproducible Python environment, train two baseline machine-learning models, measure their system costs, and compare the measurements with deployment resource budgets.

## 2. Method

The Breast Cancer Wisconsin dataset was loaded with `load_breast_cancer(return_X_y=True)`.
The dataset was split into 70% training data and 30% test data using a stratified split and `random_state=42`.

Two models were tested:
- Logistic Regression with `max_iter=1000` and `random_state=42`.
- Random Forest with 100 estimators and `random_state=42`.

Training time was measured after one warm-up run and then repeated five times. The median time was used. Single-sample inference was repeated 100 times and the median was recorded. The models were saved with joblib to measure their file size. Process memory was measured with psutil.

## 3. Results

Run the scripts first, then copy the values from `results/baseline_accuracy.csv` and `results/measurements.csv` into the table below.

| Model | Accuracy | Training median (ms) | Inference median (ms) | Model size (KB) | Peak memory (MB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | TODO | TODO | TODO | TODO | TODO |
| Random Forest | TODO | TODO | TODO | TODO | TODO |

### Deployment budgets

| Platform | Memory | Maximum latency | Maximum model size |
|---|---|---:|---:|
| Cloud | >= 1 GB | 100 ms | 500 MB |
| Edge | 256-1024 MB | 50 ms | 50 MB |
| Mobile | 64-256 MB | 20 ms | 10 MB |
| TinyML | <= 256 KB | 10 ms | 100 KB |

### Deployment suitability

Use the actual measurements above to explain where each model can run. Do not claim a platform fits unless the measured memory, latency, and model size satisfy the laboratory budget.

## 4. Conclusions

1. TODO: Write one conclusion comparing the accuracy of the two models.
2. TODO: Write one conclusion comparing their training/inference cost and model size.
3. TODO: Write one conclusion about which deployment targets are realistic according to the measured results.
