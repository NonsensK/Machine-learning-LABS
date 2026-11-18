# Lab 7 — Compression, pruning and distillation

Daniil Krylou · Student ID 72119 · Group 02

## Protocol and data

Fashion-MNIST uses 55,000 training images and 5,000 validation images from the official training set, with a seed-42 split. The official 10,000 test images are used for the final table. Images are converted to float32 and normalized with mean 0.2860 and standard deviation 0.3530. Input tensors are prepared in memory before timing training.

The baseline has 1,015,370 parameters and is trained for 5 epochs. Each pruning trial starts from that same checkpoint. Global and layer-wise magnitude pruning use targets 0, 25, 50, 75, 90 and 95 percent. Fixed masks are applied to gradients and parameters throughout three fine-tuning epochs. The zero-pruning fine-tune is shared between the two curves because the procedures are identical at zero sparsity.

Structured pruning ranks convolution filters by L2 norm, removes filters and their dependent input channels, and slices the first linear layer accordingly. Targets 10, 25 and 50 percent are rounded to valid integer channel counts. These models also receive three fine-tuning epochs. The parameter-reduction column measures physical size reduction. Weight sparsity measures zeros inside the remaining tensors.

## Efficiency pillars

Data is identical across runs (55,000 training, 5,000 validation, 10,000 test). The model columns report accuracy, parameters and serialized dense/COO storage. Compute columns report batch-one CPU latency and dense multiply-accumulate operations per image. Energy is unavailable because no hardware power meter was used. Training duration is included in comparison.csv and is not converted into joules.

| Model | Accuracy | Parameters | Dense KiB | COO KiB | CPU ms (b1) | MACs/image | Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 90.52% | 1,015,370 | 3969.3 | 19898.7 | 0.177 | 2,025,984 | Unmeasured |
| unpruned_finetuned | 90.58% | 1,015,370 | 3969.3 | 19898.7 | 0.141 | 2,025,984 | Unmeasured |
| global_0.25 | 90.64% | 1,015,370 | 3969.3 | 14938.4 | 0.140 | 2,025,984 | Unmeasured |
| global_0.5 | 90.72% | 1,015,370 | 3969.3 | 9978.0 | 0.170 | 2,025,984 | Unmeasured |
| global_0.75 | 90.91% | 1,015,370 | 3969.3 | 5015.1 | 0.166 | 2,025,984 | Unmeasured |
| global_0.9 | 91.00% | 1,015,370 | 3969.3 | 2030.5 | 0.140 | 2,025,984 | Unmeasured |
| global_0.95 | 90.65% | 1,015,370 | 3969.3 | 1031.1 | 0.140 | 2,025,984 | Unmeasured |
| layer_0.25 | 90.70% | 1,015,370 | 3969.3 | 14925.7 | 0.169 | 2,025,984 | Unmeasured |
| layer_0.5 | 90.68% | 1,015,370 | 3969.3 | 9952.7 | 0.161 | 2,025,984 | Unmeasured |
| layer_0.75 | 90.19% | 1,015,370 | 3969.3 | 4979.7 | 0.141 | 2,025,984 | Unmeasured |
| layer_0.9 | 87.44% | 1,015,370 | 3969.3 | 1995.9 | 0.141 | 2,025,984 | Unmeasured |
| layer_0.95 | 84.84% | 1,015,370 | 3969.3 | 1001.4 | 0.155 | 2,025,984 | Unmeasured |
| structured_0.1 | 90.56% | 888,826 | 3475.0 | 17410.1 | 0.193 | 1,674,752 | Unmeasured |
| structured_0.25 | 90.43% | 762,426 | 2981.3 | 14926.6 | 0.137 | 1,351,744 | Unmeasured |
| structured_0.5 | 89.75% | 510,058 | 1995.5 | 9974.6 | 0.114 | 790,400 | Unmeasured |
| student_plain | 88.07% | 103,018 | 405.5 | 2032.6 | 0.112 | 383,872 | Unmeasured |
| student_distilled | 88.31% | 103,018 | 405.5 | 2032.6 | 0.115 | 383,872 | Unmeasured |

## Storage and latency

Each latency is the median of 100 calls after 20 warm-up calls on CPU with 4 threads. Batch sizes 1 and 128 are recorded. Input generation, disk loading and preprocessing are excluded. All models use dense PyTorch kernels for these timings.

Unstructured pruning leaves tensor shapes unchanged, so dense MAC counts remain unchanged and zeros still occupy dense storage. COO stores nonzero values and integer coordinates, making it larger at low sparsity. The sparse checkpoint is validated by reconstructing every tensor. It is a storage comparison, while inference uses dense tensors. Physical filter removal reduces channels and matrix dimensions, which can improve latency. Thread scheduling, cache behavior and kernel selection also affect measured speed.

## Knowledge distillation

The student has 103,018 parameters. Plain and distilled students start with the same weights and use the same minibatch order, optimizer, learning rate and 5-epoch budget. The teacher is the frozen, unpruned baseline. T=4 and alpha=0.5.

The implementation follows the manual's written KL(student || teacher) direction explicitly, using sum(p_student * (log p_student - log p_teacher)), with T squared scaling. It combines this with hard-label cross entropy.

The plain student reached 88.07%, and the distilled student reached 88.31%. The difference is +0.24 percentage points. This is a single-seed, fixed-budget comparison. The student's serialized size and inference architecture are the same in both training methods. Distillation adds teacher computation during training.

## Conclusion

Use the accuracy/sparsity plots to select a tolerable pruning level. Dense checkpoint size and dense inference latency do not automatically improve when weights become zero. Compact filters and the smaller student alter the executed architecture. The measured table supports comparing these changes against accuracy loss. Energy efficiency remains unverified without direct power measurements.
