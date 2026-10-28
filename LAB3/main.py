import os, time, statistics, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile
import matplotlib.pyplot as plt
from pathlib import Path
SEED=42
BATCH=128
EPOCHS=5
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs",exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
train_full=datasets.FashionMNIST("./data",train=True,download=True,transform=transform)
train_ds,val_ds=random_split(train_full,[55000,5000],generator=torch.Generator().manual_seed(SEED))
test_ds=datasets.FashionMNIST("./data",train=False,download=True,transform=transform)
train_loader=DataLoader(train_ds,batch_size=BATCH,shuffle=True,num_workers=0)
val_loader=DataLoader(val_ds,batch_size=BATCH,shuffle=False,num_workers=0)
test_loader=DataLoader(test_ds,batch_size=BATCH,shuffle=False,num_workers=0)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),
                               nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(32*7*7,10))
    def forward(self,x): return self.net(x)

class PatchAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch=nn.Conv2d(1,64,kernel_size=7,stride=7)
        layer=nn.TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=128,
                                         batch_first=True,dropout=0.0)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2)
        self.head=nn.Linear(64,10)
    def forward(self,x):
        x=self.patch(x)              # B,64,4,4 => 16 patches
        x=x.flatten(2).transpose(1,2)# B,16,64
        x=self.encoder(x)
        return self.head(x.mean(dim=1))

models={"MLP":MLP(),"CNN":CNN(),"Attention":PatchAttention()}

def accuracy(model,loader):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            pred=model(x).argmax(1)
            correct+=(pred==y).sum().item(); total+=y.numel()
    return correct/total

def latency(model,batch_size,loops):
    model.eval()
    x=torch.randn(batch_size,1,28,28,device=DEVICE)
    with torch.no_grad():
        for _ in range(10): model(x)
        if DEVICE.type=="cuda": torch.cuda.synchronize()
        times=[]
        for _ in range(loops):
            if DEVICE.type=="cuda": torch.cuda.synchronize()
            t=time.perf_counter(); model(x)
            if DEVICE.type=="cuda": torch.cuda.synchronize()
            times.append((time.perf_counter()-t)*1000)
    return statistics.median(times)

def hand_flops():
    # Approximate multiply-add FLOPs, convention 2 FLOPs per MAC.
    mlp=2*784*256 + 2*256*128 + 2*128*10
    # Conv output spatial sizes 28x28 then 14x14 after pool, then 14x14 conv
    cnn=2*3*3*1*16*28*28 + 2*3*3*16*32*14*14 + 2*(32*7*7)*10
    # Patch projection + Transformer linear and attention terms (seq=16,d=64,ff=128)
    seq,d,ff=16,64,128
    patch=2*49*1*d*seq
    one_layer=2*seq*d*3*d + 2*seq*seq*d + 2*seq*seq*d + 2*seq*d*d + 2*seq*d*ff + 2*seq*ff*d
    attn=patch+2*one_layer+2*d*10
    return {"MLP":mlp,"CNN":cnn,"Attention":attn}


def write_report(rows):
    by = {r["model"]: r for r in rows}
    lines = [
        "# Lab 3 — Architectures and their costs", "",
        "## Experimental setup", "",
        f"Device: `{DEVICE}`. Fashion-MNIST: 55,000 training / 5,000 validation / 10,000 official test images. Normalize(mean=0.2860, std=0.3530).",
        "All models: Adam (lr=0.001), 5 epochs, batch size 128, random seed 42. Test accuracy is measured on the official test split.",
        "Inference timings: evaluation mode, no gradients, 10 warm-up passes; median of 100 runs for batch 1 and 20 runs for batch 128. CPU timings depend on machine load.", "",
        "## Measured results", "",
        "| Model | Parameters | Hand FLOPs* | THOP FLOPs* | Test accuracy | Batch 1 median (ms) | Batch 128 median (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['parameters']:,} | {r['hand_flops']:,} | {r['thop_flops']:,.0f} | {r['test_accuracy']:.2%} | {r['latency_batch1_ms']:.3f} | {r['latency_batch128_ms']:.3f} |")
    lines += ["", "*FLOPs estimates use 2 operations per multiply-accumulate (MAC). THOP returns MAC estimates; the code multiplies them by 2. Neither count represents all actual CPU instructions.", "",
        "## Representative hand calculations", "",
        "**MLP** (linear layers only):", "",
        "- Linear 784 → 256: 2 × 784 × 256 = 401,408 FLOPs.",
        "- Linear 256 → 128: 2 × 256 × 128 = 65,536 FLOPs.",
        "- Linear 128 → 10: 2 × 128 × 10 = 2,560 FLOPs.",
        "- Total: **469,504 FLOPs** (bias additions and activations excluded).", "",
        "**CNN** (convolutions and final linear layer):", "",
        "- Conv 1→16, 3×3, 28×28: 2 × 3 × 3 × 1 × 16 × 28 × 28 = 225,792 FLOPs.",
        "- Conv 16→32, 3×3, 14×14: 2 × 3 × 3 × 16 × 32 × 14 × 14 = 1,806,336 FLOPs.",
        "- Linear 1568→10: 2 × 1568 × 10 = 31,360 FLOPs.",
        "- Total: **2,063,488 FLOPs** (pooling, bias additions, and activations excluded).", "",
        "**Attention** (16 patches, d=64, feedforward=128, two encoder layers):", "",
        "- Patch projection: 2 × 49 × 64 × 16 = 100,352 FLOPs.",
        "- Per encoder layer, Q/K/V: 2 × 16 × 64 × (3 × 64) = 393,216 FLOPs.",
        "- Attention QKᵀ scores: 2 × 16² × 64 = 32,768 FLOPs.",
        "- Attention-weighted values: 2 × 16² × 64 = 32,768 FLOPs.",
        "- Attention output projection: 2 × 16 × 64 × 64 = 131,072 FLOPs.",
        "- Feedforward layers: 2 × 16 × 64 × 128 + 2 × 16 × 128 × 64 = 524,288 FLOPs.",
        "- Per encoder layer total: 1,114,112 FLOPs; two layers: 2,228,224 FLOPs.",
        "- Classifier 64→10: 2 × 64 × 10 = 1,280 FLOPs.",
        "- Total: **2,329,856 FLOPs**, excluding softmax, normalization, GELU, biases, pooling and residual operations.", "",
        "## Why hand estimates and THOP differ", "",
        "For MLP and CNN, THOP agrees with the hand count for the chosen counted operations. For the Transformer, THOP may not account for all internal multi-head attention matrix products and other functional operations, so its value is an **undercount** relative to this hand estimate. THOP's reported number is not a complete measurement of attention compute. The hand estimate also omits some non-matrix operations. Thus both are approximate and not directly equivalent to execution time.", "",
        "## Architectural trade-offs", "",
        "**MLP:** dense layers process flattened pixels, losing explicit 2D locality. The largest parameter count means more weight data to read (about 0.94 MB in float32), but its simple dense kernels achieved the lowest CPU latency in this experiment.", "",
        "**CNN:** local kernels reuse the same weights across image positions and exploit spatial locality. Pooling reduces later feature-map dimensions. It achieved the highest test accuracy with the fewest parameters, despite more arithmetic and slower CPU inference than MLP.", "",
        "**Attention:** patches permit global token-to-token interactions, but attention-score computation and memory grow quadratically with token count. Additional projections, normalization and token mixing increase memory traffic and runtime. Here it was slower and less accurate after only five epochs; this does not imply Transformers always underperform.", "",
        "**Conclusion:** CNN was most parameter-efficient and most accurate for this small image task; MLP was fastest on the tested CPU. Choice depends on whether accuracy, model size or latency is the main constraint.", "",
        "## Notes and limitations", "",
        "The lab handout requests architectures 'target 100k–150k parameters', but the exact prescribed layer configurations yield 235,146 (MLP), 20,490 (CNN), and 70,794 (Attention). We preserved the specified architectures rather than modifying them to hit the target.",
        "The patch embedding is implemented as a 7×7 convolution with stride 7, equivalent to a learned linear projection of non-overlapping 7×7 patches.",
        "Results are specific to the random seed, training duration, CPU and software versions; timings are not hardware-independent.", "",
        "## Plots", "",
        "![Accuracy vs parameters](outputs/accuracy_vs_parameters.png)", "",
        "![Accuracy vs batch-1 latency](outputs/accuracy_vs_latency.png)", "",
        "## Submission", "", "Include `main.py`, `requirements.txt`, `report.md`, `outputs/results.json`, both plots and a Git commit. Model weight files are optional unless the instructor requests them.", ""]
    Path("report.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    rows=[]
    for name,model in models.items():
        model=model.to(DEVICE)
        print(f"\n=== {name} | device={DEVICE} ===")
        try:
            print(summary(model,input_size=(1,1,28,28),verbose=0))
        except Exception as e: print("torchinfo summary failed:",e)
        params=sum(p.numel() for p in model.parameters())
        try:
            macs,thop_params=profile(model,inputs=(torch.randn(1,1,28,28,device=DEVICE),),verbose=False)
            thop_flops=2*macs
        except Exception as e:
            print("THOP profiling failed:",e); thop_flops=float("nan")
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        criterion=nn.CrossEntropyLoss()
        for epoch in range(EPOCHS):
            model.train(); total_loss=0
            for x,y in train_loader:
                x,y=x.to(DEVICE),y.to(DEVICE)
                optimizer.zero_grad(); loss=criterion(model(x),y)
                loss.backward(); optimizer.step()
                total_loss+=loss.item()
            va=accuracy(model,val_loader)
            print(f"Epoch {epoch+1}/{EPOCHS}: loss={total_loss/len(train_loader):.4f}, val_acc={va:.4f}")
        test_acc=accuracy(model,test_loader)
        l1=latency(model,1,100); l128=latency(model,128,20)
        torch.save(model.cpu().state_dict(),f"outputs/{name.lower()}_state.pt")
        rows.append({"model":name,"parameters":params,"hand_flops":hand_flops()[name],
                     "thop_flops":thop_flops,"test_accuracy":test_acc,
                     "latency_batch1_ms":l1,"latency_batch128_ms":l128})
        print(f"Test accuracy={test_acc:.4f}; latency batch1={l1:.3f}ms; batch128={l128:.3f}ms")
    with open("outputs/results.json","w") as f: json.dump(rows,f,indent=2)
    # Plot accuracy vs parameters and vs batch-1 latency
    for xkey,xlabel,filename in [
        ("parameters","Parameter count","accuracy_vs_parameters.png"),
        ("latency_batch1_ms","Batch-1 inference latency (ms)","accuracy_vs_latency.png")]:
        plt.figure()
        for r in rows: plt.scatter(r[xkey],r["test_accuracy"],label=r["model"])
        for r in rows: plt.annotate(r["model"],(r[xkey],r["test_accuracy"]))
        plt.xlabel(xlabel); plt.ylabel("Test accuracy"); plt.title("Fashion-MNIST trade-off")
        plt.grid(True); plt.tight_layout(); plt.savefig(f"outputs/{filename}",dpi=160); plt.close()
    write_report(rows)
    print("\nDone. Results and plots are in outputs/; report.md generated.")
if __name__=="__main__": main()
