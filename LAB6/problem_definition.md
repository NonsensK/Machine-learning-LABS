# Problem definition

Given one 28 × 28 grayscale Fashion-MNIST image, predict one of ten garment categories. Use the CNN architecture selected in Lab 3. Build a reproducible training pipeline and measure how data loading, batch size, optimizer and learning-rate schedule affect the training process. Verify that gradient accumulation approximates direct training with the same effective batch and that training can resume from a saved checkpoint.

Report validation accuracy, training throughput and sampled peak CPU process memory. For the proposed deployment project, evaluate test accuracy and compare the FP32 and INT8 model sizes and batch-one CPU latency. See [the individual project proposal](project_proposal.md) for targets and the measurement plan.
