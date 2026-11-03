# Lab 3 — Neural Network Architectures and Costs

Compare MLP, CNN and patch-attention models on Fashion-MNIST (5 epochs, seed 42).

## Structure
- `src/main.py`: experiment source code.
- `results/`: saved metrics, plots and model weights from the original archive.
- `report.md`: experiment report.

## Run from the LAB3 directory
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/main.py
```

The script downloads Fashion-MNIST into `data/`, writes artifacts to `results/`, and regenerates `report.md`. Results depend on hardware and package versions. The included results were copied from the uploaded archive; training was not re-run during restructuring.

## GitHub
Commit `src/`, `results/`, `README.md`, `report.md`, and `requirements.txt`. Do not commit `.venv/`, `.idea/`, `.git/`, or downloaded Fashion-MNIST data.
