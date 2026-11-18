"""Run all Lab 7 experiments on Fashion-MNIST."""
import argparse
import copy
import hashlib
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import FashionMNIST
import yaml

from models import CNN, apply_masks, compact_filters, distillation_loss, macs, prune_weights

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def accuracy(model, x, y):
    model.eval()
    correct = 0
    for start in range(0,len(y),512):
        correct += (model(x[start:start+512]).argmax(1) == y[start:start+512]).sum().item()
    return correct/len(y)


def train(name, model, data, config, output, epochs, masks=None, teacher=None, reuse=False):
    folder = output/name
    folder.mkdir(exist_ok=True)
    checkpoint = folder/"model.pt"
    if reuse and checkpoint.exists() and (folder/"history.csv").exists():
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        return model
    x,y,vx,vy = data
    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"])
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y),generator=generator)
        total_loss = 0.0
        start = perf_counter()
        for indices in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[indices])
            if teacher is None:
                loss = torch.nn.functional.cross_entropy(logits,y[indices])
            else:
                with torch.no_grad():
                    target = teacher(x[indices])
                loss = distillation_loss(logits,target,y[indices],config["alpha"],config["temperature"])
            loss.backward()
            if masks:
                for key, parameter in model.named_parameters():
                    if key in masks:
                        parameter.grad.mul_(masks[key])
            optimizer.step()
            if masks:
                apply_masks(model,masks)
            total_loss += loss.item()*len(indices)
        elapsed = perf_counter()-start
        val = accuracy(model,vx,vy)
        history.append(dict(epoch=epoch+1,loss=total_loss/len(y),validation_accuracy=val,
                            training_seconds=elapsed))
        print(f"{name} {epoch+1}/{epochs}: val={val:.4f}, {elapsed:.1f}s",flush=True)
    pd.DataFrame(history).to_csv(folder/"history.csv",index=False)
    torch.save(model.state_dict(),checkpoint)
    if masks:
        torch.save(masks,folder/"fixed_masks.pt")
    return model


@torch.inference_mode()
def latency(model, batch):
    model.eval()
    inputs = torch.randn(batch,1,28,28)
    for _ in range(20):
        model(inputs)
    times = []
    for _ in range(100):
        start = perf_counter()
        model(inputs)
        times.append((perf_counter()-start)*1000)
    return statistics.median(times)


def measure(name, model, test, output, baseline_params, target=0, kind="baseline"):
    folder = output/name
    dense = folder/"model.pt"
    state = model.state_dict()
    # COO tensors preserve the original shape and values, including dense biases.
    sparse = {key: value.to_sparse().coalesce() if value.ndim>1 else value
              for key,value in state.items()}
    sparse_path = folder/"sparse_coo.pt"
    torch.save(sparse,sparse_path)
    restored = torch.load(sparse_path,weights_only=True)
    for key,value in restored.items():
        torch.testing.assert_close(value.to_dense() if value.is_sparse else value,state[key])
    parameters = sum(p.numel() for p in model.parameters())
    weights = [p for name,p in model.named_parameters() if name.endswith("weight")]
    zeros = sum((p==0).sum().item() for p in weights)
    weights_count = sum(p.numel() for p in weights)
    history = pd.read_csv(folder/"history.csv")
    return dict(model=name,kind=kind,target=target,test_accuracy=accuracy(model,*test),
                parameters=parameters,parameter_reduction=1-parameters/baseline_params,
                weight_sparsity=zeros/weights_count,dense_bytes=dense.stat().st_size,
                sparse_coo_bytes=sparse_path.stat().st_size,
                latency_batch1_ms=latency(model,1),latency_batch128_ms=latency(model,128),
                dense_macs=macs(model),dense_flops=2*macs(model),
                training_seconds=history.training_seconds.sum(),
                energy_joules="not measured")


def report(output, config):
    table = pd.read_csv(output/"comparison.csv")
    baseline = table.iloc[0]
    plain = table[table.model=="student_plain"].iloc[0]
    distilled = table[table.model=="student_distilled"].iloc[0]
    text = f"""# Lab 7 — Compression, pruning and distillation

Daniil Krylou · Student ID 72119 · Group 02

## Protocol and data

Fashion-MNIST uses 55,000 training images and 5,000 validation images from the official training set, with a seed-42 split. The official 10,000 test images are used for the final table. Images are converted to float32 and normalized with mean 0.2860 and standard deviation 0.3530. Input tensors are prepared in memory before timing training.

The baseline has {int(baseline.parameters):,} parameters and is trained for {config['baseline_epochs']} epochs. Each pruning trial starts from that same checkpoint. Global and layer-wise magnitude pruning use targets 0, 25, 50, 75, 90 and 95 percent. Fixed masks are applied to gradients and parameters throughout three fine-tuning epochs. The zero-pruning fine-tune is shared between the two curves because the procedures are identical at zero sparsity.

Structured pruning ranks convolution filters by L2 norm, removes filters and their dependent input channels, and slices the first linear layer accordingly. Targets 10, 25 and 50 percent are rounded to valid integer channel counts. These models also receive three fine-tuning epochs. The parameter-reduction column measures physical size reduction. Weight sparsity measures zeros inside the remaining tensors.

## Efficiency pillars

Data is identical across runs (55,000 training, 5,000 validation, 10,000 test). The model columns report accuracy, parameters and serialized dense/COO storage. Compute columns report batch-one CPU latency and dense multiply-accumulate operations per image. Energy is unavailable because no hardware power meter was used. Training duration is included in comparison.csv and is not converted into joules.

| Model | Accuracy | Parameters | Dense KiB | COO KiB | CPU ms (b1) | MACs/image | Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
"""
    for row in table.itertuples():
        text += f"| {row.model} | {row.test_accuracy:.2%} | {row.parameters:,} | {row.dense_bytes/1024:.1f} | {row.sparse_coo_bytes/1024:.1f} | {row.latency_batch1_ms:.3f} | {row.dense_macs:,} | Unmeasured |\n"
    text += f"""
## Storage and latency

Each latency is the median of 100 calls after 20 warm-up calls on CPU with {config['threads']} threads. Batch sizes 1 and 128 are recorded. Input generation, disk loading and preprocessing are excluded. All models use dense PyTorch kernels for these timings.

Unstructured pruning leaves tensor shapes unchanged, so dense MAC counts remain unchanged and zeros still occupy dense storage. COO stores nonzero values and integer coordinates, making it larger at low sparsity. The sparse checkpoint is validated by reconstructing every tensor. It is a storage comparison, while inference uses dense tensors. Physical filter removal reduces channels and matrix dimensions, which can improve latency. Thread scheduling, cache behavior and kernel selection also affect measured speed.

## Knowledge distillation

The student has {int(plain.parameters):,} parameters. Plain and distilled students start with the same weights and use the same minibatch order, optimizer, learning rate and {config['student_epochs']}-epoch budget. The teacher is the frozen, unpruned baseline. T=4 and alpha=0.5.

The implementation follows the manual's written KL(student || teacher) direction explicitly, using sum(p_student * (log p_student - log p_teacher)), with T squared scaling. It combines this with hard-label cross entropy.

The plain student reached {plain.test_accuracy:.2%}, and the distilled student reached {distilled.test_accuracy:.2%}. The difference is {(distilled.test_accuracy-plain.test_accuracy)*100:+.2f} percentage points. This is a single-seed, fixed-budget comparison. The student's serialized size and inference architecture are the same in both training methods. Distillation adds teacher computation during training.

## Conclusion

Use the accuracy/sparsity plots to select a tolerable pruning level. Dense checkpoint size and dense inference latency do not automatically improve when weights become zero. Compact filters and the smaller student alter the executed architecture. The measured table supports comparing these changes against accuracy loss. Energy efficiency remains unverified without direct power measurements.
"""
    (ROOT/"report.md").write_text(text)
    for kind in ["global","layer"]:
        subset = table[(table.kind==kind)|(table.model=="unpruned_finetuned")]
        plt.plot(subset.weight_sparsity,subset.test_accuracy,marker="o",label=kind)
    plt.xlabel("Weight sparsity"); plt.ylabel("Test accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(output/"magnitude_sparsity.png",dpi=160); plt.close()
    subset = table[table.kind=="structured"]
    plt.plot([0,*subset.parameter_reduction],[baseline.test_accuracy,*subset.test_accuracy],marker="o")
    plt.xlabel("Physical parameter reduction"); plt.ylabel("Test accuracy")
    plt.tight_layout(); plt.savefig(output/"structured_pruning.png",dpi=160); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",default=str(ROOT/"data"))
    parser.add_argument("--reuse",action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT/"config.yaml").read_text())
    output = ROOT/"results"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    train_set = FashionMNIST(args.data,train=True,download=True)
    test_set = FashionMNIST(args.data,train=False,download=True)
    x = (train_set.data[:,None].float()/255-0.2860)/0.3530
    tx = (test_set.data[:,None].float()/255-0.2860)/0.3530
    indices = torch.randperm(len(x),generator=torch.Generator().manual_seed(config["seed"]))
    training,validation = indices[:55000],indices[55000:]
    data = x[training],train_set.targets[training],x[validation],train_set.targets[validation]
    test = tx,test_set.targets
    metadata = dict(config=config,python=platform.python_version(),torch=str(torch.__version__),
                    platform=platform.platform(),device="cpu",
                    split_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest())
    metadata_path = output/"environment.json"
    if args.reuse and metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old["config"] != config:
            raise ValueError("Reuse requires identical config.")
    metadata_path.write_text(json.dumps(metadata,indent=2))
    teacher = train("baseline",CNN(),data,config,output,config["baseline_epochs"],reuse=args.reuse)
    total_params = sum(p.numel() for p in teacher.parameters())
    rows = []
    def record(name,model,target=0,kind="baseline"):
        rows.append(measure(name,model,test,output,total_params,target,kind))
        pd.DataFrame(rows).to_csv(output/"comparison.csv",index=False)
    record("baseline",teacher)
    for kind in ["global","layer"]:
        for target in config["sparsities"]:
            if kind=="layer" and target==0:
                continue
            name = "unpruned_finetuned" if target==0 else f"{kind}_{target:g}"
            model,masks = prune_weights(teacher,target,kind)
            model = train(name,model,data,config,output,config["finetune_epochs"],masks=masks,reuse=args.reuse)
            record(name,model,target,kind)
    for target in config["structured_fractions"]:
        name = f"structured_{target:g}"
        model = train(name,compact_filters(teacher,target),data,config,output,
                      config["finetune_epochs"],reuse=args.reuse)
        record(name,model,target,"structured")
    torch.manual_seed(config["seed"])
    student = CNN((8,16),128)
    for name,teacher_model in [("student_plain",None),("student_distilled",teacher.eval())]:
        model = train(name,copy.deepcopy(student),data,config,output,config["student_epochs"],
                      teacher=teacher_model,reuse=args.reuse)
        record(name,model,kind="student")
    report(output,config)


if __name__ == "__main__":
    main()
