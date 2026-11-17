# Lab 5 — Frameworks and automatic differentiation

Daniil Krylou · 72119 · Group 02

## Automatic differentiation

The scalar Value stores a float, gradient, parent nodes and a local backward rule. Iterative depth-first traversal creates a topological order. Backward resets reachable gradients, seeds the output with 1 and traverses the reversed order. Each local rule adds contributions, including when a parent is reused. Separate backward calls compute fresh gradients.

Addition, multiplication, constant powers and ReLU construct graphs. Sigmoid and log support training. The vector expression in the manual is represented with four scalar terms and summed. Across 30 seeded cases, the maximum error against PyTorch was 0, and against central differences was 5.49e-10. Both satisfy the 1e-6 threshold.

## Tiny model

A 2 → 8 → 1 network uses hidden ReLU and output sigmoid. The synthetic dataset has 200 points with noise 0.15, split 70/30 with stratification. Both implementations use float64, identical initial weights, full batches, SGD at 0.2 and 60 epochs. BCE uses the same tiny probability smoothing in both versions.

Median training time over five fresh runs was 7.018 s for the scalar engine and 0.0097 s for PyTorch. Test accuracy was 91.67% and 91.67%, respectively. Maximum final weight difference was 2.22e-16. Scalar timing includes graph allocation, backpropagation and garbage collection. PyTorch batches computations in native kernels.

## Execution modes and graph

TorchScript tracing is the execution-mode option selected from the manual. The saved inlined graph contains linear, ReLU and sigmoid operations and their parameter accesses. It captures tensor operations for this model, which has no input-dependent Python branches. The benchmark uses CPU, one thread, 20 warm-up calls and median latency over 100 calls. Trace construction took 56.23 ms and is excluded from steady-state timings.

| Batch | Eager ms | Traced ms | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.00913 | 0.00708 | 1.29× |
| 32 | 0.01473 | 0.01208 | 1.22× |
| 128 | 0.01556 | 0.01315 | 1.18× |
| 1024 | 0.02788 | 0.02271 | 1.23× |

Compilation or tracing can help repeated inference by reducing Python dispatch and enabling graph optimization. The setup cost must be amortized over enough calls. Small workloads can be dominated by framework overhead. Larger matrix operations may be dominated by kernel time, reducing the relative gain. Tracing captures the executed path and requires care with data-dependent control flow.

## Tensor representation

Both layouts contain identical values, shape [512,512] and float32 dtype. The strided view selects every second column of a [512,1024] tensor. Its strides are (1024,2), while the contiguous copy has strides (512,1). Allocation of the contiguous reference is excluded from operation timings. Copy latency is recorded separately in benchmark_setup.json.

| Layout | Matmul ms | ReLU ms |
| --- | ---: | ---: |
| non_contiguous | 0.2731 | 0.0946 |
| contiguous | 0.2139 | 0.0227 |

Strided access can require extra packing or less efficient memory access. The cost depends on the operation and backend. Copying can pay off when reused across many operations. Measurements are specific to this CPU and process load.
