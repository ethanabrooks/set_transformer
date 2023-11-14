from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from wandb.apis.public import Artifact, Run

import wandb


def download_artifact(artifact: Artifact, artifact_root: Path):
    while True:
        try:
            artifact.download(root=str(artifact_root))
            break
        except requests.exceptions.ChunkedEncodingError as e:
            print(e)
            print("\nRetrying download...\n")


def get_artifact_root(run: Run, type: str) -> Path:
    artifact = get_artifact(run, type=type)
    if run is None:
        now = datetime.now().strftime("-%d-%m-%H:%M:%S")
        return Path(f"/tmp/{now}", artifact.name)
    else:
        return Path(run.dir, artifact.name)


def get_artifact(run: Run, type: str) -> Optional[Artifact]:
    artifact: Artifact
    for artifact in run.logged_artifacts():
        if artifact.type == type:
            return artifact


def get_run(load_path: str) -> Run:
    api = wandb.Api()
    return api.run(load_path)


def download_and_return_root(load_path: str, type: str) -> Path:
    run = get_run(load_path)
    artifact = get_artifact(run=run, type=type)
    artifact_root = get_artifact_root(run=run, type=type)
    if artifact is None:
        raise ValueError(f"Could not find {type} artifact in {load_path}")
    download_artifact(artifact=artifact, artifact_root=artifact_root)
    return artifact_root
