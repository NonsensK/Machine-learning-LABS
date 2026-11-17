"""Reproduce all Lab 5 measurements on CPU."""
import csv
import gc
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import yaml
from engine import Value

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def save_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gradient_check():
    rng = np.random.default_rng(42)
    rows = []
    for case in range(30):
        values = rng.uniform(-1, 1, (3, 4))
        nodes = [[Value(x) for x in row] for row in values]
        loss = sum((nodes[0][i] * nodes[1][i] + nodes[2][i]) ** 2
                   for i in range(4)).relu()
        loss.backward()
        tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        ((tensor[0] * tensor[1] + tensor[2]) ** 2).relu().sum().backward()
        def objective(x):
            return np.maximum((x[0] * x[1] + x[2]) ** 2, 0).sum()
        for row in range(3):
            for col in range(4):
                plus, minus = values.copy(), values.copy()
                plus[row, col] += 1e-6
                minus[row, col] -= 1e-6
                numeric = (objective(plus) - objective(minus)) / 2e-6
                actual = nodes[row][col].grad
                expected = tensor.grad[row, col].item()
                rows.append(dict(case=case, variable=f"{row}_{col}", engine=actual,
                                 pytorch=expected, numeric=numeric,
                                 torch_error=abs(actual-expected), numeric_error=abs(actual-numeric)))
    save_csv("gradient_check.csv", rows)
    assert max(max(row["torch_error"], row["numeric_error"]) for row in rows) < 1e-6
    return rows


def train_scalar(weights, x, y, config):
    w1, b1, w2, b2 = [[Value(v) for v in array.ravel()] for array in weights]
    params = w1 + b1 + w2 + b2
    history = []
    start = perf_counter()
    for epoch in range(config["epochs"]):
        predictions = []
        for sample in x:
            hidden = [sum(w1[j*2+k] * sample[k] for k in range(2)) + b1[j] for j in range(8)]
            logit = sum(w2[j] * hidden[j].relu() for j in range(8)) + b2[0]
            predictions.append(logit.sigmoid())
        # Smoothing keeps both implementations finite near sigmoid saturation.
        safe = [p * (1-2e-12) + 1e-12 for p in predictions]
        loss = sum(-(target*p.log() + (1-target)*(1-p).log()) for p,target in zip(safe,y)) * (1/len(y))
        loss.backward()
        for param in params:
            param.data -= config["lr"] * param.grad
        history.append(loss.data)
        # Local derivative closures form cycles, collected between epochs.
        del loss, predictions, safe, hidden, logit
        gc.collect()
    seconds = perf_counter() - start
    arrays = [np.array([v.data for v in group]).reshape(shape)
              for group, shape in zip([w1,b1,w2,b2],[(8,2),(8,),(1,8),(1,)])]
    return seconds, history, arrays


def model_from(weights):
    model = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,1), nn.Sigmoid()).double()
    with torch.no_grad():
        for param, value in zip(model.parameters(), weights):
            param.copy_(torch.from_numpy(value))
    return model


def train_torch(weights, x, y, config):
    model = model_from(weights)
    inputs = torch.tensor(x, dtype=torch.float64)
    labels = torch.tensor(y[:,None], dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    history = []
    start = perf_counter()
    for epoch in range(config["epochs"]):
        optimizer.zero_grad()
        p = model(inputs) * (1-2e-12) + 1e-12
        loss = -(labels*p.log() + (1-labels)*(1-p).log()).mean()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return perf_counter()-start, history, model


def median_ms(operation, repetitions=100):
    for _ in range(20):
        operation()
    measurements = []
    for _ in range(repetitions):
        start = perf_counter()
        operation()
        measurements.append((perf_counter()-start)*1000)
    return statistics.median(measurements)


def benchmark(model):
    rows = []
    sample = torch.randn(128,2,dtype=torch.float64)
    start = perf_counter()
    traced = torch.jit.trace(model.eval(), sample)
    build_ms = (perf_counter()-start)*1000
    (OUT / "traced_graph.txt").write_text(str(traced.inlined_graph))
    traced.save(str(OUT / "traced_model.pt"))
    with torch.inference_mode():
        for batch in [1,32,128,1024]:
            x = torch.randn(batch,2,dtype=torch.float64)
            torch.testing.assert_close(model(x), traced(x))
            eager = median_ms(lambda: model(x))
            jit = median_ms(lambda: traced(x))
            rows.append(dict(batch=batch,eager_ms=eager,traced_ms=jit,speedup=eager/jit))
    save_csv("execution_timings.csv",rows)
    tensors = []
    base = torch.randn(512,1024)
    view = base[:,::2]
    contiguous = view.contiguous()
    rhs = torch.randn(512,512)
    for name, tensor in [("non_contiguous",view),("contiguous",contiguous)]:
        tensors.append(dict(layout=name,shape=str(list(tensor.shape)),dtype=str(tensor.dtype),
                            strides=str(tensor.stride()),contiguous=tensor.is_contiguous(),
                            matmul_ms=median_ms(lambda: tensor @ rhs),
                            relu_ms=median_ms(lambda: tensor.relu())))
    torch.testing.assert_close(view @ rhs, contiguous @ rhs)
    save_csv("tensor_layout.csv",tensors)
    copy_ms = median_ms(lambda: view.contiguous())
    (OUT / "benchmark_setup.json").write_text(json.dumps(
        dict(device="cpu",threads=1,warmup=20,repetitions=100,trace_build_ms=build_ms,
             contiguous_copy_ms=copy_ms),indent=2))
    return rows, tensors, build_ms


def main():
    OUT.mkdir(exist_ok=True)
    config = yaml.safe_load((ROOT/"config.yaml").read_text())
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    checks = gradient_check()
    x,y = make_moons(n_samples=config["samples"],noise=config["noise"],random_state=config["seed"])
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=.3,stratify=y,random_state=42)
    initial = nn.Sequential(nn.Linear(2,8),nn.ReLU(),nn.Linear(8,1),nn.Sigmoid()).double()
    weights = [p.detach().numpy().copy() for p in initial.parameters()]
    timing = []
    # Both frameworks get an unmeasured warm-up.
    warmup = dict(config,epochs=2)
    train_scalar(weights,train_x,train_y,warmup)
    train_torch(weights,train_x,train_y,warmup)
    for repeat in range(5):
        seconds,scalar_loss,arrays = train_scalar(weights,train_x,train_y,config)
        torch_seconds,torch_loss,model = train_torch(weights,train_x,train_y,config)
        scalar_model = model_from(arrays)
        with torch.inference_mode():
            inputs = torch.tensor(test_x,dtype=torch.float64)
            scalar_accuracy = ((scalar_model(inputs).numpy().ravel()>=.5)==test_y).mean()
            torch_accuracy = ((model(inputs).numpy().ravel()>=.5)==test_y).mean()
        difference = max(np.max(np.abs(a-p.detach().numpy())) for a,p in zip(arrays,model.parameters()))
        assert difference < 1e-8
        timing.append(dict(run=repeat+1,scalar_seconds=seconds,torch_seconds=torch_seconds,
                           scalar_accuracy=scalar_accuracy,torch_accuracy=torch_accuracy,
                           max_weight_difference=difference))
        print(f"Run {repeat+1}/5 completed, scalar {seconds:.2f}s, torch {torch_seconds:.3f}s",flush=True)
    save_csv("training_timings.csv",timing)
    save_csv("loss_history.csv",[dict(epoch=i+1,scalar=a,pytorch=b) for i,(a,b) in enumerate(zip(scalar_loss,torch_loss))])
    plt.plot(scalar_loss,label="Scalar engine")
    plt.plot(torch_loss,"--",label="PyTorch")
    plt.xlabel("Epoch"); plt.ylabel("Binary cross entropy"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT/"loss_curves.png",dpi=160); plt.close()
    torch.save(model.state_dict(),OUT/"model_state.pt")
    execution,layouts,build = benchmark(model)
    versions = dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,
                    numpy=np.__version__,sklearn=sklearn.__version__)
    (OUT/"versions.json").write_text(json.dumps(versions,indent=2))
    scalar_time = statistics.median(row["scalar_seconds"] for row in timing)
    torch_time = statistics.median(row["torch_seconds"] for row in timing)
    report = f"""# Lab 5 — Frameworks and automatic differentiation

Daniil Krylou · 72119 · Group 02

## Automatic differentiation

The scalar Value stores a float, gradient, parent nodes and a local backward rule. Iterative depth-first traversal creates a topological order. Backward resets reachable gradients, seeds the output with 1 and traverses the reversed order. Each local rule adds contributions, including when a parent is reused. Separate backward calls compute fresh gradients.

Addition, multiplication, constant powers and ReLU construct graphs. Sigmoid and log support training. The vector expression in the manual is represented with four scalar terms and summed. Across 30 seeded cases, the maximum error against PyTorch was {max(r['torch_error'] for r in checks):.3g}, and against central differences was {max(r['numeric_error'] for r in checks):.3g}. Both satisfy the 1e-6 threshold.

## Tiny model

A 2 → 8 → 1 network uses hidden ReLU and output sigmoid. The synthetic dataset has {config['samples']} points with noise {config['noise']}, split 70/30 with stratification. Both implementations use float64, identical initial weights, full batches, SGD at {config['lr']} and {config['epochs']} epochs. BCE uses the same tiny probability smoothing in both versions.

Median training time over five fresh runs was {scalar_time:.3f} s for the scalar engine and {torch_time:.4f} s for PyTorch. Test accuracy was {scalar_accuracy:.2%} and {torch_accuracy:.2%}, respectively. Maximum final weight difference was {max(r['max_weight_difference'] for r in timing):.3g}. Scalar timing includes graph allocation, backpropagation and garbage collection. PyTorch batches computations in native kernels.

## Execution modes and graph

TorchScript tracing is the execution-mode option selected from the manual. The saved inlined graph contains linear, ReLU and sigmoid operations and their parameter accesses. It captures tensor operations for this model, which has no input-dependent Python branches. The benchmark uses CPU, one thread, 20 warm-up calls and median latency over 100 calls. Trace construction took {build:.2f} ms and is excluded from steady-state timings.

| Batch | Eager ms | Traced ms | Speedup |
| ---: | ---: | ---: | ---: |
"""
    for row in execution:
        report += f"| {row['batch']} | {row['eager_ms']:.5f} | {row['traced_ms']:.5f} | {row['speedup']:.2f}× |\n"
    report += """
Compilation or tracing can help repeated inference by reducing Python dispatch and enabling graph optimization. The setup cost must be amortized over enough calls. Small workloads can be dominated by framework overhead. Larger matrix operations may be dominated by kernel time, reducing the relative gain. Tracing captures the executed path and requires care with data-dependent control flow.

## Tensor representation

Both layouts contain identical values, shape [512,512] and float32 dtype. The strided view selects every second column of a [512,1024] tensor. Its strides are (1024,2), while the contiguous copy has strides (512,1). Allocation of the contiguous reference is excluded from operation timings. Copy latency is recorded separately in benchmark_setup.json.

| Layout | Matmul ms | ReLU ms |
| --- | ---: | ---: |
"""
    for row in layouts:
        report += f"| {row['layout']} | {row['matmul_ms']:.4f} | {row['relu_ms']:.4f} |\n"
    report += "\nStrided access can require extra packing or less efficient memory access. The cost depends on the operation and backend. Copying can pay off when reused across many operations. Measurements are specific to this CPU and process load.\n"
    (ROOT/"report.md").write_text(report)


if __name__ == "__main__":
    main()
