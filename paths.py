"""Resolve experiment outputs locally or under a configured HPC storage root."""

import os
from pathlib import Path


def has_saved_unet(path):
    """Ignore interrupted saves that contain only config and temporary files."""
    path = Path(path)
    return (path / "config.json").is_file() and any(
        (path / filename).is_file()
        for filename in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")
    )


def experiment_output_dir(config, repo_root):
    output_dir = Path(config["experiment"]["output_dir"]).expanduser()
    if output_dir.is_absolute():
        return output_dir

    storage_root = os.environ.get("MDST_OUTPUT_ROOT")
    if storage_root:
        storage_root = Path(storage_root).expanduser()
        if not storage_root.is_absolute():
            raise ValueError("MDST_OUTPUT_ROOT must be an absolute path")
        return storage_root / output_dir
    return Path(repo_root) / output_dir
