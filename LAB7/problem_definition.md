# Problem definition

Classify Fashion-MNIST images into ten clothing categories while measuring the trade-off between classification quality, representation size and CPU inference cost. Compare a CNN near one million parameters against magnitude-pruned, physically filter-pruned and approximately 100,000-parameter student models.

Evaluate top-1 test accuracy, parameter count, actual weight sparsity, serialized dense and sparse COO sizes, median batch-one and batch-128 latency, dense MACs/FLOPs and training duration. Compare identical student architectures trained with hard labels alone and with a frozen teacher. Energy requires a power measurement and is explicitly marked unmeasured.
