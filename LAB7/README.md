# Lab 7 — Compression, pruning and distillation

Daniil Krylou · 72119 · Group 02

## Reproduce

Run from LAB7 with Python and the packages in requirements.txt.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/main.py
python -m unittest discover -s tests
```

Use `--data /absolute/path/to/data` to reuse Fashion-MNIST. Use `--reuse` to reuse completed checkpoints and repeat the measurements. Reuse requires the same configuration. An interrupted training trial restarts that trial from its original initialization. Completed trials can be reused.

The full study trains a teacher, 11 unstructured-pruning variants including a shared zero-pruning control, three physically compacted variants and two student models. All pruning variants receive three fine-tuning epochs. The teacher and each student receive five epochs, a reproducible budget chosen for this lab.

## Evidence

| Requirement | Files |
| --- | --- |
| Baseline accuracy, parameters, storage and latency | results/comparison.csv |
| Global and layer-wise masks | src/models.py, results/global_*/fixed_masks.pt, results/layer_*/fixed_masks.pt |
| Pruning curves | results/magnitude_sparsity.png, results/structured_pruning.png |
| Per-epoch losses, validation accuracy and duration | results/*/history.csv |
| Dense and COO storage | results/*/model.pt, results/*/sparse_coo.pt |
| Distillation and plain student comparison | results/comparison.csv, report.md |
| Seed, split hash, configuration and environment | results/environment.json |
| Implementation checks | tests/test_compression.py |

Models use CPU with four PyTorch threads. Latency measurements use 20 warm-up calls and the median of 100 timed calls for batches 1 and 128. Dense forward MAC counts include convolution and linear layers and exclude bias, activation and pooling operations. FLOPs count two operations per MAC.

Energy in joules is marked unmeasured because no hardware power measurement was available. The report includes training time and compute as separate observations.

The distillation objective follows the manual's KL(student || teacher) direction with T=4 and alpha=0.5. Structured pruning physically slices output filters, dependent input channels and linear input columns.
