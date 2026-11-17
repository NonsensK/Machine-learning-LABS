# Lab 6 — Training pipelines and project proposal

**Daniil Krylou · 72119 · Group 02**

## Setup

The 20,490-parameter CNN from Lab 3 was trained on Fashion-MNIST with a fixed 55,000/5,000 split and seed 42. The experiments ran on the MacBook Air M2 CPU with four PyTorch threads. All accuracy figures in this report use the same validation split. The baseline uses Adam at 0.001 and batch size 128.

The baseline reached 89.42% after 5 epochs.

## CPU profile

| Stage | Time (ms) | Share |
| --- | ---: | ---: |
| data loading | 3,663 | 22.0% |
| forward | 4,048 | 24.4% |
| backward | 8,667 | 52.2% |
| optimizer | 211 | 1.3% |
| other | 24 | 0.1% |

Backward was the largest labeled span. Data loading accounted for 22.0% of profiled training time. The spans include the profiler's instrumentation overhead.

## DataLoader workers

| Workers | Samples/s | Status |
| ---: | ---: | --- |
| 0 | 3,365 | measured |
| 1 | 3,813 | measured |
| 2 | 4,714 | measured |
| 4 | 3,774 | measured |
| 8 | 4,795 | measured |

The highest observed throughput used 8 workers. The main process spent 1.5% of that epoch waiting for batches and converting them to tensors. Each setting uses the same NumPy batch transport and the second epoch excludes worker startup. A separate optimizer run shared the CPU during this sweep, so absolute throughput is specific to that load.

## Batch-size study

| Batch | Samples/s | Peak RSS (MiB) | Accuracy after 5 epochs |
| ---: | ---: | ---: | ---: |
| 32 | 4,199 | 501.0 | 91.06% |
| 64 | 4,479 | 519.3 | 90.28% |
| 128 | 5,172 | 612.3 | 89.42% |
| 256 | 4,990 | 564.0 | 89.22% |
| 512 | 5,492 | 601.2 | 88.04% |

Batch 512 produced the highest throughput. Batch 32 produced the best five-epoch accuracy. Each configuration ran for the same number of epochs, so larger batches performed fewer optimizer steps. Peak RSS is sampled process memory and includes allocator state and dataset memory.

## Optimizers and learning-rate schedules

Each run used a 10-epoch budget and batch size 128. Adam started at 0.001 and SGD at 0.01 with momentum 0.9.

| Optimizer and schedule | Final accuracy | Best accuracy | Final validation loss |
| --- | ---: | ---: | ---: |
| sgd / step | 90.12% | 90.18% | 0.2767 |
| sgd / cosine | 90.38% | 90.38% | 0.2726 |
| sgd / plateau | 89.96% | 90.76% | 0.2722 |
| adam / step | 90.58% | 90.82% | 0.2667 |
| adam / cosine | 90.88% | 90.88% | 0.2636 |
| adam / plateau | 90.82% | 90.96% | 0.2578 |

adam / cosine had the highest final accuracy for these learning rates. ReduceLROnPlateau observes validation loss, StepLR decays at fixed intervals, and cosine annealing changes the rate on every epoch.

## Gradient accumulation

Direct batch 256 and four microbatches of 64 used the same sample order and effective batch size, including the final partial update. The maximum validation accuracy difference was 0.080 percentage points and the maximum training loss difference was 0.000080.

## Checkpoint recovery

Training stopped after epoch 3 and resumed for epochs 4, 5. The largest absolute difference between the uninterrupted and resumed loss, accuracy or learning rate was 0. See the overlaid validation-loss curves.

## Tracking and proposal

The recorded metrics and artifacts can be imported into MLflow using the command in the README. MLflow could not be installed on the measurement host.

The [individual project proposal](project_proposal.md) targets Fashion-MNIST classification with INT8 quantization, a maximum model size of 5 MB and p95 CPU latency of at most 50 ms. Instructor approval is pending.
