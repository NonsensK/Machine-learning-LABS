# Lab 6 — Training pipeline and project proposal

Daniil Krylou · 72119 · Group 02

This lab uses the 20,490-parameter CNN from Lab 3. The original weight file is included in `src/lab3_cnn_state.pt`. Each study starts from a fresh initialization with the same seed so changes to training settings can be compared. Fashion-MNIST uses a reproducible 55,000/5,000 training/validation split and the same normalization as Lab 3.

## Run

From the `LAB6` directory, use Python 3.13 on macOS or Linux.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/main.py --mode study --output new_results
```

To use an existing Fashion-MNIST download, add `--data /absolute/path/to/data`. The directory should contain `FashionMNIST/raw`.

The YAML config specifies the seed, data, device, batch, epochs, workers, optimizer, schedules, learning rates and experiment grids. The provided study uses CPU so that training times and profiler CPU spans can be compared directly. Use a different output directory for a new study.

To repeat only the worker study in a regular terminal:

```bash
python src/main.py --mode workers --output worker_check
```

Use the same `--data` path if reusing an existing dataset. This study uses the same transformations for all five worker counts and transfers collated batches as NumPy arrays so that PyTorch worker subprocesses need no Torch storage sharing.

## Recover an interrupted run

```bash
python src/main.py --output recovery_demo --stop-after 3
python src/main.py --output recovery_demo --resume
```

The checkpoint stores model, optimizer and scheduler state, history, RNG and sampling state, and run ID. The study compares epochs 1–5 with an uninterrupted baseline. Epoch numbers shown are one-based.

## Open MLflow

The script logs configuration, epoch metrics, histories, checkpoints and study artifacts directly to MLflow when installed. If an existing results folder contains `tracking/*.jsonl`, import those recorded runs first. This takes the measured results and requires no retraining.

```bash
python src/import_mlflow.py --results results
cd results
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns
```

Open [the local MLflow UI](http://127.0.0.1:5000). For new direct MLflow runs, change into the output directory you passed to the study and start the UI there.

## Evidence

| Task | Output |
| --- | --- |
| Reproducible YAML training | `config.yaml`, baseline history and checkpoint |
| One-epoch CPU profile | `results/profile.csv`, per-batch profiler spans |
| DataLoader worker comparison | `results/workers.csv`, `results/workers.png` |
| Batch-size comparison | `results/batches.csv` |
| Optimizer and LR schedule comparison | `results/schedules.csv`, accuracy and LR plots |
| Gradient accumulation | `results/accumulation.csv`, loss plot |
| Recovery after epoch 3 | `results/recovery.json`, validation loss plot |
| Individual project | `project_proposal.md` (approval pending) |

Worker throughput uses the second training epoch to avoid worker startup. Batch-size throughput aggregates training time over five epochs. Peak memory is sampled RSS of the main training process in MiB. All study accuracies are on the fixed validation split. The project proposal reserves the official test split for final evaluation.

The tests check the Lab 3 checkpoint against the CNN architecture and compare direct batching with gradient accumulation for a partial final update.

```bash
python -m unittest discover -s tests
```

See `report.md` for the interpretation of the measurements.
