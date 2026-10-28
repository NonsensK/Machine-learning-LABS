# Lab 3 — Architectures and their costs

Run on macOS/Linux from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

`main.py` downloads Fashion-MNIST, trains MLP/CNN/Attention for 5 epochs, profiles FLOPs, times inference, and regenerates `report.md` and plots.

**Included `outputs/` and `report.md` use actual measurements supplied from the student's Mac terminal run.** The uploaded ZIP contained source only, not saved model weights or generated output files. The included results are reconstructed from the terminal output and not independently re-run here. Running `main.py` overwrites these files with new measurements.

To commit:

```bash
git init
git add main.py requirements.txt README.md report.md outputs/results.json outputs/*.png
git commit -m "Complete Lab 3 architecture comparison"
```
