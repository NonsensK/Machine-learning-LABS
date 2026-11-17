# Individual Project Proposal

**Daniil Krylou · Student ID 72119**  
**Machine Learning · Group 02**

## Offline clothing image classification

**Problem.** Build a classifier that assigns a single garment image to one of ten clothing categories. The prototype will run locally on a laptop CPU and accept one grayscale image at a time.

**Dataset.** Fashion-MNIST contains labeled 28 × 28 grayscale images in ten categories. Split the official training set into 55,000 training and 5,000 validation images using seed 42. Reserve the official 10,000-image test split for final evaluation. Use the validation split for model and quantization decisions.

**Baseline and success metric.** Use the two-convolution CNN architecture from Lab 3 as the FP32 baseline. The primary metric is classification accuracy on the official test split. Aim for at least 89% test accuracy and a quantization-related drop of at most 1 percentage point relative to the FP32 baseline. Report per-class errors in a confusion matrix.

**Deployment constraints.** The serialized inference model must be at most 5 MB. Measure p95 end-to-end inference latency on a MacBook Air M2 CPU at batch size 1, including input normalization and class selection, with a target of at most 50 ms. Use 50 warm-up calls followed by 500 timed calls, and use the same inputs and CPU thread count for both models.

**Planned optimization.** Apply static post-training INT8 quantization to the convolutional and linear layers using a representative, fixed calibration subset from the training data. Compare FP32 and INT8 accuracy, serialized size and p95 latency. Select the quantized model if it meets the quality and deployment targets.

**Deliverables.** Reproducible training and evaluation scripts, exported inference models, and a comparison table with accuracy, size and latency.
