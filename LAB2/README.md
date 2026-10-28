# Lab 2 — A Neural Network from Scratch

**Course lab:** Chapter 3 — Neural Network Fundamentals  
**Implementation:** Python, NumPy, scikit-learn (dataset and splitting only), Matplotlib  
**Goal:** Implement and validate a two-hidden-layer classifier for handwritten digits **without using a neural-network framework**.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 lab02.py
```

The script trains all six required configurations and writes:

- `results/gradient_check.csv` — numerical versus analytical derivatives.
- `results/metrics.csv` — final losses, test accuracies, and timing medians.
- `results/loss_curves.png` — training-loss curves for all six runs.

`report.md` contains a separate narrative report. The values below are taken from the **included `results/*.csv` files**; timing may vary when rerunning on a different computer.

---

## Task 1 — Dataset preparation (10 points)

**Requirement:** Load `sklearn.datasets.load_digits`, normalize inputs by dividing by 16, one-hot encode class labels, and perform a stratified 70/30 train/test split with `random_state=42`.

**What was implemented (`data()` in `lab02.py`):**

```python
digits = load_digits()
x = digits.data.astype(np.float64) / 16.0
y = digits.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, stratify=y, random_state=42
)
y_train_one_hot = np.eye(10)[y_train]
y_test_one_hot = np.eye(10)[y_test]
```

Each digit image is **8 × 8 = 64** pixels, flattened into a 64-dimensional input vector. Pixel values originally lie between 0 and 16, so normalization maps them to [0, 1]:

\[
x_{\mathrm{scaled}} = \frac{x_{\mathrm{original}}}{16}.
\]

There are **1,797** digit samples. The 70/30 split produces **1,257 training samples** and **540 test samples** (because 30% of 1,797 rounds to 540). `stratify=y` preserves approximately the same class proportions in both subsets. One-hot encoding maps a label, for example 3, to `[0,0,0,1,0,0,0,0,0,0]`.

**Expected shapes:** `x_train: (1257, 64)`, `x_test: (540, 64)`, `y_train_one_hot: (1257, 10)`, `y_test_one_hot: (540, 10)`.

## Task 2 — Network architecture and parameter dimensions (15 points)

**Requirement:** Design a network with two hidden ReLU layers, widths **W = 16** and **W = 64**, and a 10-class softmax output. Initialize weights from N(0, 0.01).

**Architecture:**

```text
Input image (64 features)
       |
       v
Dense 1: 64 -> W   [ReLU]
       |
       v
Dense 2: W -> W    [ReLU]
       |
       v
Dense 3: W -> 10  [Softmax]
       |
       v
10 class probabilities (digits 0–9)
```

Here `W` is either 16 or 64. In the `initialize()` function, every weight is sampled independently from a normal distribution with **mean 0 and standard deviation 0.01**; all biases start at zero.

| Parameter | General shape | W = 16 | W = 64 |
|---|---|---|---|
| `W1` | `(64, W)` | `(64, 16)` | `(64, 64)` |
| `b1` | `(W,)` | `(16,)` | `(64,)` |
| `W2` | `(W, W)` | `(16, 16)` | `(64, 64)` |
| `b2` | `(W,)` | `(16,)` | `(64,)` |
| `W3` | `(W, 10)` | `(16, 10)` | `(64, 10)` |
| `b3` | `(10,)` | `(10,)` | `(10,)` |

**Trainable-parameter count:**

\[
N = (64W+W)+(W^2+W)+(10W+10)=W^2+76W+10.
\]

- **W = 16:** 256 + 1,216 + 10 = **1,482 parameters**.
- **W = 64:** 4,096 + 4,864 + 10 = **8,970 parameters**.

The second model has approximately **6.05 times** as many trainable parameters. The assignment also asks for an **on-paper drawing** of this architecture; the text diagram above can be used as a guide, but it does not replace the handwritten requirement.

## Task 3 — Forward propagation (20 points)

**Requirement:** Compute the forward pass layer by layer. For a batch of size `m`, check that the output shape is `(m, 10)` and that each row sums to 1 within `1e-6`.

**Implemented in `forward(x, p)`:**

Let `X` be a batch with shape `(m, 64)` and let `@` denote matrix multiplication.

\[
Z_1 = XW_1+b_1, \qquad A_1=\operatorname{ReLU}(Z_1)=\max(0,Z_1)
\]
\[
Z_2=A_1W_2+b_2, \qquad A_2=\operatorname{ReLU}(Z_2)
\]
\[
Z_3=A_2W_3+b_3.
\]

For each example `i` and class `k`, softmax gives:

\[
P_{ik}=\frac{\exp(Z_{3,ik}-c_i)}{\sum_{j=1}^{10}\exp(Z_{3,ij}-c_i)},
\qquad c_i=\max_j Z_{3,ij}.
\]

Subtracting the row maximum improves numerical stability without changing the probabilities. For a 32-example batch the intermediate shapes are:

| Tensor | Shape |
|---|---|
| `X` | `(32, 64)` |
| `Z1`, `A1` | `(32, W)` |
| `Z2`, `A2` | `(32, W)` |
| `Z3`, `P` | `(32, 10)` |

**Actual assertions in `main()`:**

```python
assert probs.shape == (32, 10)
assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
```

Thus the output has the required dimensions and each example's ten class probabilities sum to approximately 1.

## Task 4 — Cross-entropy loss (10 points)

**Requirement:** Implement multiclass cross-entropy and check that an untrained 10-class model has loss approximately `ln(10) = 2.302585`.

**Implemented in `loss(probs, y)`:**

\[
L=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{10}Y_{ik}\ln(P_{ik}).
\]

`Y` contains one-hot targets. Only the probability assigned to the correct class contributes to each example's loss. The code clips probabilities to `[1e-15, 1]` before taking logarithms to avoid `log(0)`.

If the network predicts 0.1 for each of ten classes, the per-example loss is:

\[
-\ln(0.1)=\ln(10)\approx2.302585093.
\]

**Observed initial loss in the supplied run log:** `2.302595128619221`, very close to the theoretical value. This confirms the expected starting behavior, not good predictive performance.

## Task 5 — Backpropagation and numerical gradient checking (30 points)

**Requirement:** Derive gradients using the chain rule; implement backpropagation; compare 10 selected parameters against central finite differences using `epsilon = 1e-5`. At least 9 of 10 relative errors must be below `1e-6`.

**Derivation (`backward()`):**

For softmax combined with cross-entropy, the derivative with respect to output logits simplifies to:

\[
\frac{\partial L}{\partial Z_3}=\delta_3=\frac{P-Y}{m}.
\]

The final dense layer gradients follow from the matrix chain rule:

\[
\frac{\partial L}{\partial W_3}=A_2^T\delta_3,\qquad
\frac{\partial L}{\partial b_3}=\sum_{i=1}^{m}\delta_{3,i}.
\]

ReLU has derivative `1` for positive pre-activations and `0` for negative ones. The implementation uses `0` at exactly zero:

\[
\delta_2=(\delta_3 W_3^T)\odot\mathbf{1}[Z_2>0],
\]
\[
\frac{\partial L}{\partial W_2}=A_1^T\delta_2,\qquad
\frac{\partial L}{\partial b_2}=\sum_{i=1}^{m}\delta_{2,i},
\]
\[
\delta_1=(\delta_2 W_2^T)\odot\mathbf{1}[Z_1>0],
\]
\[
\frac{\partial L}{\partial W_1}=X^T\delta_1,\qquad
\frac{\partial L}{\partial b_1}=\sum_{i=1}^{m}\delta_{1,i}.
\]

`\odot` means elementwise multiplication. The `1/m` factor is already included in `delta3`, so it must **not** be applied a second time.

**Numerical check (`gradient_check()`):** For a selected scalar parameter `theta`, calculate:

\[
g_{\mathrm{numeric}}=\frac{L(\theta+10^{-5})-L(\theta-10^{-5})}{2\times10^{-5}}.
\]

Compare against the analytic backpropagation derivative:

\[
\mathrm{relative\ error}=\frac{|g_{\mathrm{numeric}}-g_{\mathrm{analytic}}|}{\max(10^{-8},|g_{\mathrm{numeric}}|+|g_{\mathrm{analytic}}|)}.
\]

**Observed result: 10/10 checks passed**, exceeding the required 9/10. The largest relative error in the supplied CSV is approximately **5.91 × 10^-10**, well below `1e-6`.

The checker samples **10 random scalar entries across weight matrices and bias vectors** (`W1`, `b1`, `W2`, `b2`, `W3`, `b3`); the saved sample happens to include 7 weight entries and 3 bias entries. Some checked derivatives are zero; both numerical and analytical values agree for those entries. For a more informative check, the script temporarily multiplies weight matrices by 30 in a **separate checking model**, while actual training still uses the required `std=0.01` initialization.

For every perturbation, the original parameter is restored before continuing. Full individual results are in `results/gradient_check.csv`.

## Task 6 — Training experiments, loss curves, and accuracy (20 points)

**Requirement:** Shuffle training samples every epoch, train using mini-batches of 32 for 50 epochs, log every epoch, and test the Cartesian product of widths `[16, 64]` and learning rates `[0.001, 0.01, 0.1]`. Save the plot to `results/loss_curves.png` and report final test accuracy.

**Implemented in `train()`:**

1. Initialize the model parameters using seed 42.
2. Shuffle training indices using `rng.permutation(...)` each epoch.
3. Divide training examples into consecutive batches of up to 32 examples (the final batch may be smaller).
4. Run forward propagation and compute backpropagation gradients on each batch.
5. Apply mini-batch stochastic gradient descent (SGD) to every parameter:

   \[
   \theta\leftarrow\theta-\eta\frac{\partial L}{\partial\theta},
   \]

   where `eta` is the learning rate.
6. Compute full-training-set cross-entropy after each epoch, append it to the history, and print the epoch's loss.
7. After epoch 50, predict the labels of the held-out test set and calculate accuracy:

   \[
   \mathrm{accuracy}=\frac{\#\mathrm{correct\ predictions}}{\#\mathrm{test\ examples}}.
   \]

**Actual experiment results from `results/metrics.csv`:**

| Hidden width | Learning rate | Final training loss | Test accuracy |
|---:|---:|---:|---:|
| 16 | 0.001 | 2.302550 | 10.19% |
| 16 | 0.010 | 2.302476 | 10.19% |
| 16 | 0.100 | 1.597017 | 32.59% |
| 64 | 0.001 | 2.302526 | 10.19% |
| 64 | 0.010 | 2.302343 | 10.19% |
| **64** | **0.100** | **0.168405** | **93.89%** |

**Interpretation:** Four configurations remain close to chance-level performance (about 10% accuracy for ten balanced classes), with loss near `ln(10)`. A width of 16 with learning rate 0.1 begins learning but reaches only 32.59%. The best tested combination, **width 64 and learning rate 0.1**, achieves **93.89% test accuracy**. The very small required weight initialization slows learning substantially at smaller learning rates. These outcomes are observations from the six prescribed configurations, not a claim that these hyperparameters are globally optimal.

**Loss plot:**

![Training loss for six configurations](results/loss_curves.png)

The figure plots the **full training-set loss after each epoch**; it is not a test-loss curve. Because all six curves share one axis, runs remaining near 2.30 may look almost flat compared with the rapidly improving best run.

## Task 7 — Forward and backward timing (10 points)

**Requirement:** Measure forward and backward propagation separately; report the median of 100 runs and explain the difference using matrix multiplication.

**Implemented in `timing()`:** For each trained configuration, run `forward()` and `backward()` separately **100 times** on a 32-sample training batch, measuring elapsed time with `time.perf_counter()`. Convert seconds to milliseconds and compute each median with `np.median()`.

**Recorded timing values from the included `results/metrics.csv` (milliseconds):**

| Width | Learning rate | Forward median (ms) | Backward median (ms) | Backward / forward |
|---:|---:|---:|---:|---:|
| 16 | 0.001 | 0.02030 | 0.02213 | 1.09× |
| 16 | 0.010 | 0.02001 | 0.02192 | 1.10× |
| 16 | 0.100 | 0.01990 | 0.02196 | 1.10× |
| 64 | 0.001 | 0.02994 | 0.03740 | 1.25× |
| 64 | 0.010 | 0.03023 | 0.03859 | 1.28× |
| 64 | 0.100 | 0.03071 | 0.03865 | 1.26× |

**Why backward is slower:** The forward pass uses one matrix multiplication per dense layer: `X @ W1`, `A1 @ W2`, and `A2 @ W3`, plus activations and softmax. The backward pass computes both the gradients of the weight matrices (`A.T @ delta`) and the gradients propagated to earlier layers (`delta @ W.T`), plus ReLU masks and bias-gradient sums. Thus backward generally performs more matrix-multiplication work. A wider hidden layer also increases the cost of multiplying matrices of shape `(m, W)` and `(W, W)`; the middle layer scales approximately with `m * W^2`.

**Measurement limitations:** These are short, CPU-level wall-clock timings and can change with processor, operating system, NumPy/BLAS build, background activity, and cache state. Medians reduce the influence of occasional slow runs; they do not remove all measurement noise. The included CSV and the previously recorded Mac terminal log can show different millisecond values from different executions while preserving the same overall conclusion.

---

## Final conclusion

The lab demonstrates the full neural-network pipeline from scratch: preparing the digits dataset, defining a dimensionally valid 64–W–W–10 architecture, computing stable softmax probabilities and cross-entropy loss, deriving and checking backpropagation gradients, training six SGD configurations, and measuring execution time. The gradient check passed **10/10** sampled entries; the best of the required configurations reached **93.89% test accuracy**. The experiments also show that network width and learning rate strongly affect optimization under the specified small-weight initialization.

## File structure

```text
lab02_ai_73875/
├── lab02.py
├── requirements.txt
├── README.md
├── report.md
├── run.log
└── results/
    ├── gradient_check.csv
    ├── metrics.csv
    └── loss_curves.png
```

**Submission note:** The lab manual explicitly requests that the architecture be designed **on paper** and the gradients be derived **by hand**. The formulas and diagram in this README document the work and provide a guide for that handwritten component. The lab manual does not specify a particular ZIP or repository submission format.
