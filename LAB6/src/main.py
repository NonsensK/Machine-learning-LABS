"""Run reproducible Fashion-MNIST training and the Lab 6 studies."""
import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
from PIL import Image
import platform
import random
import ctypes
import resource
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
import yaml


class CNN(nn.Module):
    """CNN architecture from Lab 3."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, inputs):
        return self.net(inputs)


def load_data(config):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,)),
    ])
    dataset = datasets.FashionMNIST(
        config["data"], train=True, download=True, transform=transform,
    )
    train, validation = random_split(
        dataset, [len(dataset) - config["validation_size"], config["validation_size"]],
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    if config["limit"]:
        train = Subset(train, range(min(config["limit"], len(train))))
        validation = Subset(validation, range(min(config["limit"], len(validation))))
    return train, validation


class NumpyWorkerDataset(Dataset):
    """Fashion-MNIST subset that can cross a spawn boundary without Torch storage IPC."""
    def __init__(self, dataset):
        indices = np.arange(len(dataset))
        while isinstance(dataset, Subset):
            indices = np.asarray(dataset.indices)[indices]
            dataset = dataset.dataset
        self.indices = indices
        self.image_path = Path(dataset.raw_folder) / "train-images-idx3-ubyte"
        self.image_shape = tuple(dataset.data.shape)
        self.images = None if self.image_path.is_file() else dataset.data.numpy()
        self.labels = dataset.targets.numpy()
        self.transform = dataset.transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        if self.images is None:
            self.images = np.memmap(
                self.image_path, mode="r", dtype=np.uint8, offset=16, shape=self.image_shape,
            )
        index = self.indices[position]
        image = Image.fromarray(self.images[index])
        return self.transform(image).numpy(), self.labels[index]


def numpy_collate(samples):
    images, labels = zip(*samples)
    return np.stack(images), np.asarray(labels)


def make_scheduler(config, optimizer):
    name = config["scheduler"]
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config["step_size"], gamma=config["gamma"],
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config["gamma"], patience=config["plateau_patience"],
        )
    raise ValueError(f"Unknown scheduler: {name}")


def train_epoch(model, loader, optimizer, device, accumulation, profiling):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    samples = len(loader.dataset)
    stages = defaultdict(float)

    def stage(name):
        return record_function(name) if profiling else nullcontext()

    started = perf_counter()
    with stage("data_loading"):
        batches = iter(loader)
    for index in range(len(loader)):
        start = perf_counter()
        with stage("data_loading"):
            images, labels = next(batches)
            images = torch.as_tensor(images, device=device)
            labels = torch.as_tensor(labels, device=device)
        stages["data_loading"] += perf_counter() - start
        if index % accumulation == 0:
            remaining = samples - index * loader.batch_size
            group_samples = min(remaining, loader.batch_size * accumulation)
        with stage("forward"):
            losses = nn.functional.cross_entropy(model(images), labels, reduction="sum")
        with stage("backward"):
            (losses / group_samples).backward()
        total_loss += losses.detach()
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            with stage("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    if loader.collate_fn is numpy_collate:
        # Prefetching loaders already consume the sampler through StopIteration.
        next(batches, None)
    loss = total_loss.item() / samples
    seconds = perf_counter() - started
    return loss, seconds, stages["data_loading"]


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss += nn.functional.cross_entropy(logits, labels, reduction="sum")
        correct += (logits.argmax(1) == labels).sum()
    return loss.item() / len(loader.dataset), correct.item() / len(loader.dataset)


def current_rss():
    """Return resident bytes for the training process."""
    if platform.system() == "Darwin":
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        task = ctypes.c_uint.in_dll(libc, "mach_task_self_")
        info = (ctypes.c_uint64 * 6)()
        count = ctypes.c_uint(ctypes.sizeof(info) // 4)
        status = libc.task_info(task, 20, ctypes.byref(info), ctypes.byref(count))
        if status == 0:
            return info[1]
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with open("/proc/self/statm") as stats:
        return int(stats.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def memory_sampler(stop, peak):
    while True:
        peak[0] = max(peak[0], current_rss())
        if stop.wait(0.05):
            return


class DeferredTracking:
    """Save real run events for import into MLflow when it is installed."""
    def __init__(self, output):
        self.output = output

    @contextmanager
    def start_run(self, run_id=None, run_name=None):
        run_id = run_id or uuid4().hex
        self.journal = self.output / "tracking" / f"{run_id}.jsonl"
        self.journal.parent.mkdir(exist_ok=True)
        if not self.journal.exists():
            self.write("name", run_name)
        yield SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def write(self, kind, value):
        with self.journal.open("a") as stream:
            stream.write(json.dumps([kind, value]) + "\n")

    def log_params(self, values):
        self.write("params", values)

    def log_metrics(self, values, step):
        self.write("metrics", [values, step])

    def log_artifact(self, path):
        self.write("artifact", str(Path(path).relative_to(self.output)))


def profile_stages(recorded):
    """Read labeled CPU spans without building Python objects for all Torch ops."""
    stages = {"data_loading", "forward", "backward", "optimizer"}
    spans = []
    with TemporaryDirectory(prefix="lab6-trace-") as temp:
        trace = Path(temp) / "trace.json"
        recorded.export_chrome_trace(str(trace))
        current = None
        with trace.open() as stream:
            for line in stream:
                field, _, value = line.strip().partition(":")
                if field == '"name"':
                    name = value.strip().strip(',"')
                    current = name if name in stages else None
                elif current and field == '"dur"':
                    spans.append({"stage": current, "ms": float(value.strip().rstrip(",")) / 1000})
                    current = None
    if {span["stage"] for span in spans} != stages:
        raise RuntimeError("Profile trace is missing one or more stage annotations.")
    return pd.DataFrame(spans)


def run(name, config, data, output, *, profiling=False, stop_after=None, resume=False):
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "checkpoint.pt"
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    model = CNN().to(device)
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_lr"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config["sgd_lr"], momentum=config["momentum"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    scheduler = make_scheduler(config, optimizer)
    train, validation = data
    collate = None
    if config.get("worker_transport") == "numpy":
        train = NumpyWorkerDataset(train)
        collate = numpy_collate
    order = torch.Generator().manual_seed(config["seed"])
    worker_seed = torch.Generator().manual_seed(config["seed"] + 1)
    sampler = torch.utils.data.RandomSampler(train, generator=order)
    train_loader = DataLoader(
        train, batch_size=config["batch"], sampler=sampler,
        num_workers=config["workers"], generator=worker_seed,
        persistent_workers=config["workers"] > 0, collate_fn=collate,
    )
    val_loader = DataLoader(validation, batch_size=512, shuffle=False)
    history = []
    existing_run = None
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if config != state["config"]:
            raise ValueError("Resume requires the same configuration.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        order.set_state(state["order_rng"])
        worker_seed.set_state(state["worker_rng"])
        history = state["history"]
        existing_run = state["run_id"]
    config_path = folder / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with mlflow.start_run(run_id=existing_run, run_name=name) as active:
        if not resume:
            mlflow.log_params({key: str(value) for key, value in config.items()})
            mlflow.log_params({"train_size": len(train), "validation_samples": len(validation),
                               "platform": platform.platform(), "torch": str(torch.__version__)})
            mlflow.log_artifact(str(config_path))
        for epoch in range(len(history) + 1, (stop_after or config["epochs"]) + 1):
            peak = [0]
            stop = Event()
            sampler_thread = Thread(target=memory_sampler, args=(stop, peak), daemon=True)
            sampler_thread.start()
            profiler = profile(activities=[ProfilerActivity.CPU]) if profiling else nullcontext()
            lr = optimizer.param_groups[0]["lr"]
            try:
                with profiler as recorded:
                    loss, seconds, loading = train_epoch(
                        model, train_loader, optimizer, device, config["accumulation"], profiling,
                    )
            finally:
                stop.set()
                sampler_thread.join()
            val_loss, accuracy = evaluate(model, val_loader, device)
            if config["scheduler"] == "plateau":
                scheduler.step(val_loss)
            elif scheduler:
                scheduler.step()
            row = dict(epoch=epoch, train_loss=loss, val_loss=val_loss, val_accuracy=accuracy,
                       lr=lr, train_seconds=seconds, data_loading_seconds=loading,
                       samples_per_second=len(train) / seconds, peak_rss_mib=peak[0] / 2**20)
            history.append(row)
            mlflow.log_metrics({key: value for key, value in row.items() if key != "epoch"}, step=epoch)
            temp = checkpoint.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict() if scheduler else None,
                            history=history, config=config, run_id=active.info.run_id,
                            torch_rng=torch.get_rng_state(), order_rng=order.get_state(),
                            worker_rng=worker_seed.get_state()), temp)
            os.replace(temp, checkpoint)
            print(f"{name} epoch={epoch} accuracy={accuracy:.4f} time={seconds:.2f}s", flush=True)
            if profiling:
                spans = profile_stages(recorded)
                times = spans.groupby("stage", sort=False)["ms"].sum().to_dict()
                breakdown = [{"stage": key, "ms": value, "epoch_percent": 100 * value / (seconds * 1000)}
                             for key, value in times.items()]
                other = seconds * 1000 - sum(times.values())
                breakdown.append(dict(stage="other", ms=other, epoch_percent=100 * other / (seconds * 1000)))
                pd.DataFrame(breakdown).to_csv(output / "profile.csv", index=False)
                profile_table = folder / "profiler_spans.csv"
                spans.to_csv(profile_table, index=False)
                mlflow.log_artifact(str(output / "profile.csv"))
                mlflow.log_artifact(str(profile_table))
        history_path = folder / "history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(checkpoint))
    return history


def plot_curves(output, name, histories, metric, ylabel):
    fig, axis = plt.subplots(figsize=(7, 4))
    for label, history in histories.items():
        axis.plot([row["epoch"] for row in history], [row[metric] for row in history],
                  marker="o", label=label)
    axis.set(xlabel="Epoch", ylabel=ylabel)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=160)
    plt.close(fig)


def worker_study(config, output, execute):
    config = dict(config, worker_transport="numpy")
    workers = []
    for count in config["worker_values"]:
        try:
            history = execute(f"workers_{count}", dict(config, workers=count, epochs=2))
        except (PermissionError, RuntimeError) as error:
            if not count or not any(message in str(error) for message in ("Operation not permitted", "Permission denied")):
                raise
            reason = "Host denied multiprocessing shared memory"
            workers.append(dict(num_workers=count, status=reason))
            print(f"workers_{count}: {reason}", flush=True)
        else:
            workers.append(dict(num_workers=count, status="measured", **history[-1]))
    table = pd.DataFrame(workers)
    table.to_csv(output / "workers.csv", index=False)
    axis = table.plot(x="num_workers", y="samples_per_second", marker="o", legend=False)
    axis.set(xlabel="DataLoader workers", ylabel="Steady-state samples/s", xticks=config["worker_values"])
    axis.figure.tight_layout()
    axis.figure.savefig(output / "workers.png", dpi=160)
    plt.close(axis.figure)


def study(config, data, output, reuse=False):
    def completed_run(name, settings, **kwargs):
        history_path = output / name / "history.csv"
        config_path = output / name / "config.yaml"
        if reuse and history_path.exists() and config_path.exists():
            history = pd.read_csv(history_path, float_precision="round_trip").to_dict("records")
            if len(history) == settings["epochs"] and yaml.safe_load(config_path.read_text()) == settings:
                return history
        return run(name, settings, data, output, **kwargs)

    baseline = completed_run("baseline", config)

    worker_study(config, output, completed_run)

    batches = []
    batch_runs = {}
    for size in config["batch_values"]:
        history = baseline if size == config["batch"] else completed_run(
            f"batch_{size}", dict(config, batch=size),
        )
        batch_runs[size] = history
        batches.append(dict(batch_size=size, samples_per_second=len(data[0]) * len(history) /
                            sum(row["train_seconds"] for row in history),
                            peak_rss_mib=max(row["peak_rss_mib"] for row in history),
                            final_accuracy=history[-1]["val_accuracy"]))
    pd.DataFrame(batches).to_csv(output / "batches.csv", index=False)

    schedules = {}
    for optimizer in config["optimizers"]:
        for scheduler in config["schedulers"]:
            name = f"{optimizer}_{scheduler}"
            schedules[name] = completed_run(name, dict(config, optimizer=optimizer, scheduler=scheduler,
                                                        epochs=config["schedule_epochs"]))
    pd.DataFrame([dict(run=name, final_accuracy=history[-1]["val_accuracy"],
                       best_accuracy=max(row["val_accuracy"] for row in history),
                       final_val_loss=history[-1]["val_loss"])
                  for name, history in schedules.items()]).to_csv(output / "schedules.csv", index=False)
    plot_curves(output, "schedule_accuracy", schedules, "val_accuracy", "Validation accuracy")
    plot_curves(output, "schedule_lr", schedules, "lr", "Learning rate")

    accumulated = completed_run("accumulated", dict(config, batch=64, accumulation=4))
    direct = batch_runs[256]
    accumulation = pd.DataFrame([
        dict(epoch=a["epoch"], direct_loss=d["train_loss"], accumulated_loss=a["train_loss"],
             direct_accuracy=d["val_accuracy"], accumulated_accuracy=a["val_accuracy"])
        for d, a in zip(direct, accumulated)
    ])
    accumulation.to_csv(output / "accumulation.csv", index=False)
    plot_curves(output, "accumulation", {"256": direct, "64 × 4": accumulated},
                "train_loss", "Training loss")

    interrupted = run("recovery", config, data, output, stop_after=config["interrupt_epoch"])
    resumed = run("recovery", config, data, output, resume=True)
    keys = ("train_loss", "val_loss", "val_accuracy", "lr")
    differences = {key: max(abs(a[key] - b[key]) for a, b in zip(baseline, resumed))
                   for key in keys}
    recovery = dict(interrupted_at=interrupted[-1]["epoch"],
                    resumed_epochs=[row["epoch"] for row in resumed[len(interrupted):]],
                    max_absolute_difference=differences)
    (output / "recovery.json").write_text(json.dumps(recovery, indent=2) + "\n")
    plot_curves(output, "recovery", {"uninterrupted": baseline, "resumed": resumed},
                "val_loss", "Validation loss")
    completed_run("profile", dict(config, epochs=1), profiling=True)
    with mlflow.start_run(run_name="study_summary"):
        for artifact in sorted(output.iterdir()):
            if artifact.is_file() and artifact.suffix in {".csv", ".png", ".json"}:
                mlflow.log_artifact(str(artifact))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--mode", choices=["train", "study", "workers", "profile"], default="train")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="Reuse complete study runs with matching config.")
    parser.add_argument("--limit", type=int, default=0, help="Subset for smoke tests only.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config["data"] = str(Path(args.data or config["data"]).resolve())
    config["limit"] = args.limit
    if config["device"] != "cpu":
        parser.error("These study timings and RSS measurements require device: cpu.")
    if args.stop_after is not None and not 1 <= args.stop_after <= config["epochs"]:
        parser.error("--stop-after must be between 1 and the configured epoch count.")
    torch.set_num_threads(config["threads"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    global mlflow
    try:
        import mlflow
    except ModuleNotFoundError as error:
        if error.name != "mlflow":
            raise
        mlflow = DeferredTracking(output)
        print("MLflow unavailable: recording metrics for later import.", flush=True)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    data = load_data(config)
    os.chdir(output)
    if not isinstance(mlflow, DeferredTracking):
        mlflow.set_tracking_uri("file:mlruns")
        mlflow.set_experiment("Lab 6 Fashion-MNIST")
    if args.mode == "study":
        study(config, data, output, reuse=args.reuse)
    elif args.mode == "workers":
        worker_study(config, output, lambda name, settings: run(name, settings, data, output))
    elif args.mode == "profile":
        run("profile", dict(config, epochs=1), data, output, profiling=True)
    else:
        run("training", config, data, output, stop_after=args.stop_after, resume=args.resume)


if __name__ == "__main__":
    main()
