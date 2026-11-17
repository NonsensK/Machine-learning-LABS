"""Import the real experiment records produced while MLflow was unavailable."""
import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    results = args.results.resolve()
    journals = sorted((results / "tracking").glob("*.jsonl"))
    if not journals:
        parser.error(f"No deferred runs in {results / 'tracking'}")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.chdir(results)
    mlflow.set_tracking_uri("file:mlruns")
    experiment = mlflow.set_experiment("Lab 6 Fashion-MNIST")
    client = MlflowClient()
    for journal in journals:
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        existing = client.search_runs(
            [experiment.experiment_id], f"tags.deferred_run_id = '{journal.stem}'"
        )
        if any(run.info.status == "FINISHED" for run in existing):
            continue
        name = events[0][1]
        artifacts = {payload for kind, payload in events if kind == "artifact"}
        if not artifacts or (name != "study_summary" and not any(
            path.endswith("/history.csv") for path in artifacts
        )):
            print(f"Skipping incomplete run {name}")
            continue
        with mlflow.start_run(run_name=name, tags={"deferred_run_id": journal.stem}):
            for kind, payload in events[1:]:
                if kind == "params":
                    mlflow.log_params(payload)
                elif kind == "metrics":
                    values, epoch = payload
                    mlflow.log_metrics(values, step=epoch)
            for relative_path in sorted(artifacts):
                mlflow.log_artifact(str(results / relative_path))
        print(f"Imported {name}")


if __name__ == "__main__":
    main()
