# Lab 5 — Frameworks and automatic differentiation

Daniil Krylou · 72119 · Group 02

## Reproduce

From LAB5, create a Python environment and run the experiment script.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/main.py
python -m unittest discover -s tests
```

The script regenerates CSV tables, plots, model files and report.md. It runs on CPU and needs no dataset download. See results/versions.json for the measured environment.

## Files

| Requirement | Evidence |
| --- | --- |
| Value node and operations | src/engine.py |
| Topological backward and accumulation | src/engine.py, tests/test_engine.py |
| PyTorch and numerical validation | results/gradient_check.csv |
| Tiny model, five training runs | results/training_timings.csv, loss_history.csv, loss_curves.png |
| Eager versus traced timings | results/execution_timings.csv |
| Graph inspection | results/traced_graph.txt |
| Shape, dtype, strides and contiguity | results/tensor_layout.csv |
| Benchmark settings | results/benchmark_setup.json |
| Analysis and conclusions | report.md |

The model is 2 → 8 → 1 with ReLU then sigmoid. Both implementations start from identical weights and use the same float64 BCE objective. The engine resets gradients for each backward call and accumulates contributions from all paths within the graph. Tracing is the execution-mode alternative explicitly allowed by the manual.
