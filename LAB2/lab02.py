import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
SEED = 42


def data():
    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, stratify=y, random_state=SEED)
    return x_train, x_test, np.eye(10)[y_train], np.eye(10)[y_test], y_train, y_test


def initialize(width, seed=SEED):
    rng = np.random.default_rng(seed)
    return {'W1': rng.normal(0, .01, (64, width)), 'b1': np.zeros(width),
            'W2': rng.normal(0, .01, (width, width)), 'b2': np.zeros(width),
            'W3': rng.normal(0, .01, (width, 10)), 'b3': np.zeros(10)}


def forward(x, p):
    z1 = x @ p['W1'] + p['b1']
    a1 = np.maximum(z1, 0)
    z2 = a1 @ p['W2'] + p['b2']
    a2 = np.maximum(z2, 0)
    logits = a2 @ p['W3'] + p['b3']
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs, (x, z1, a1, z2, a2)


def loss(probs, y):
    return -np.mean(np.sum(y * np.log(np.clip(probs, 1e-15, 1)), axis=1))


def backward(probs, y, cache, p):
    x, z1, a1, z2, a2 = cache
    dz3 = (probs - y) / len(x)
    grads = {'W3': a2.T @ dz3, 'b3': dz3.sum(axis=0)}
    dz2 = (dz3 @ p['W3'].T) * (z2 > 0)
    grads['W2'] = a1.T @ dz2
    grads['b2'] = dz2.sum(axis=0)
    dz1 = (dz2 @ p['W2'].T) * (z1 > 0)
    grads['W1'] = x.T @ dz1
    grads['b1'] = dz1.sum(axis=0)
    return grads


def gradient_check(x, y, p, n=10, eps=1e-5):
    probs, cache = forward(x, p)
    grads = backward(probs, y, cache, p)
    rng = np.random.default_rng(123)
    keys = list(p)
    errors = []
    for _ in range(n):
        key = keys[int(rng.integers(len(keys)))]
        idx = tuple(int(rng.integers(s)) for s in p[key].shape)
        original = p[key][idx]
        p[key][idx] = original + eps
        plus = loss(forward(x, p)[0], y)
        p[key][idx] = original - eps
        minus = loss(forward(x, p)[0], y)
        p[key][idx] = original
        numerical = (plus - minus) / (2 * eps)
        analytical = grads[key][idx]
        relative = abs(numerical - analytical) / max(1e-8, abs(numerical) + abs(analytical))
        errors.append((key, str(idx), analytical, numerical, relative))
    return errors


def train(x, y, width, lr, epochs=50, batch_size=32):
    p = initialize(width)
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(epochs):
        indices = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            batch = indices[start:start + batch_size]
            probs, cache = forward(x[batch], p)
            grads = backward(probs, y[batch], cache, p)
            for key in p:
                p[key] -= lr * grads[key]
        current = loss(forward(x, p)[0], y)
        history.append(current)
        print(f'width={width:2d} lr={lr:.3f} epoch={epoch+1:02d}/{epochs} loss={current:.6f}')
    return p, history


def timing(x, y, p, runs=100):
    forward_times, backward_times = [], []
    for _ in range(runs):
        t = time.perf_counter()
        probs, cache = forward(x, p)
        forward_times.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        backward(probs, y, cache, p)
        backward_times.append((time.perf_counter() - t) * 1000)
    return float(np.median(forward_times)), float(np.median(backward_times))


def main():
    x_train, x_test, y_train, y_test, labels_train, labels_test = data()
    p = initialize(16)
    probs, _ = forward(x_train[:32], p)
    assert probs.shape == (32, 10)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
    print('Untrained loss:', loss(probs, y_train[:32]))
    # Nonzero inputs and moderate parameter scale make gradient checking informative.
    check_params = initialize(16)
    for key in ('W1', 'W2', 'W3'):
        check_params[key] *= 30
    checks = gradient_check(x_train[:16], y_train[:16], check_params)
    with (RESULTS / 'gradient_check.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'index', 'analytical', 'numerical', 'relative_error'])
        writer.writerows(checks)
    passing = sum(row[-1] < 1e-6 for row in checks)
    print(f'Gradient check: {passing}/10 passed')
    if passing < 9:
        raise RuntimeError('Gradient checking failed; review derivatives before training.')

    fig, ax = plt.subplots(figsize=(10, 6))
    rows = []
    for width in (16, 64):
        for lr in (.001, .01, .1):
            model, history = train(x_train, y_train, width, lr)
            predicted = forward(x_test, model)[0].argmax(axis=1)
            accuracy = float(np.mean(predicted == labels_test))
            fwd, bwd = timing(x_train[:32], y_train[:32], model)
            rows.append((width, lr, history[-1], accuracy, fwd, bwd))
            ax.plot(range(1, 51), history, label=f'W={width}, lr={lr}')
            print(f'FINAL width={width} lr={lr}: test accuracy={accuracy:.4f}, forward={fwd:.3f}ms, backward={bwd:.3f}ms')
    ax.set(xlabel='Epoch', ylabel='Training cross-entropy', title='Training loss (50 epochs)')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTS / 'loss_curves.png', dpi=160)
    plt.close(fig)
    with (RESULTS / 'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['width', 'learning_rate', 'final_train_loss', 'test_accuracy', 'forward_median_ms', 'backward_median_ms'])
        writer.writerows(rows)
    print('Saved results to', RESULTS)


if __name__ == '__main__':
    main()
