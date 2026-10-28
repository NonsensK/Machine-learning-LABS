import importlib.metadata
import platform
from pathlib import Path

PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm"
]


def main():
    lines = [f"Python=={platform.python_version()}"]
    for package in PACKAGES:
        try:
            lines.append(f"{package}=={importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{package}==NOT INSTALLED")

    output = "\n".join(lines)
    print(output)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "versions.txt").write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
