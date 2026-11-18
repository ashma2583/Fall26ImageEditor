"""Train an InstructPix2Pix model (baseline or continuous) on MagicBrush."""

import argparse
import random
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from time import perf_counter

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.baseline import InstructPix2PixModel
from models.continuous import ContinuousInstructPix2PixModel
from data import load_instructpix2pix, load_magicbrush, load_magicbrush_validation, load_ultraedit
from evaluation import evaluate_baseline


ROOT = Path(__file__).resolve().parent


MODEL_TYPES = ("baseline", "continuous")


def resolve_resume_checkpoint(resume_from, output_dir):
    """Return a saved UNet and its completed epoch number."""
    if resume_from is None:
        return 0, None

    if str(resume_from) == "latest":
        candidates = []
        for path in output_dir.glob("epoch-*/unet"):
            try:
                epoch = int(path.parent.name.removeprefix("epoch-"))
            except ValueError:
                continue
            if (path / "config.json").is_file():
                candidates.append((epoch, path))
        if not candidates:
            raise FileNotFoundError(f"No epoch-N/unet checkpoint found in {output_dir}")
        return max(candidates)

    path = Path(resume_from)
    if not path.is_absolute():
        path = ROOT / path
    if path.name != "unet" and (path / "unet").is_dir():
        path = path / "unet"
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"No UNet checkpoint found at {path}")

    epoch_name = path.parent.name
    if epoch_name.startswith("epoch-"):
        try:
            return int(epoch_name.removeprefix("epoch-")), path
        except ValueError:
            pass
    if epoch_name == "final":
        completed = []
        for candidate in path.parent.parent.glob("epoch-*/unet/config.json"):
            try:
                completed.append(int(candidate.parent.parent.name.removeprefix("epoch-")))
            except ValueError:
                continue
        if completed:
            return max(completed), path
    raise ValueError(
        f"Cannot infer the completed epoch from {path}; use an epoch-N/unet checkpoint"
    )


def collate_batch(rows):
    batch = {
        key: torch.stack([torch.as_tensor(row[key]) for row in rows])
        for key in ("source", "target", "original") if key in rows[0]
    }
    batch["instruction"] = [row["instruction"] for row in rows]
    return batch


def build_model(model_type, model_settings, settings, device, resume_checkpoint=None):
    model_path = str(resume_checkpoint) if resume_checkpoint else None
    if model_type == "continuous":
        return ContinuousInstructPix2PixModel(
            model_id=model_settings["pretrained"],
            device=device,
            previous_drop_rate=float(settings.get("previous_drop_rate", 0.0)),
            original_drop_rate=float(settings.get("original_drop_rate", 0.0)),
            instruction_drop_rate=float(settings.get("instruction_drop_rate", 0.0)),
            previous_degrade_rate=float(settings.get("previous_degrade_rate", 0.0)),
            model_path=model_path,
        )
    return InstructPix2PixModel(
        model_id=model_settings["pretrained"],
        device=device,
        source_drop_rate=float(settings.get("source_drop_rate", 0.0)),
        instruction_drop_rate=float(settings.get("instruction_drop_rate", 0.0)),
        model_path=model_path,
        conditioning_dropout_prob=(
            float(settings["conditioning_dropout_prob"])
            if "conditioning_dropout_prob" in settings else None
        ),
    )


def train(config, resume_from=None, epochs_override=None):
    settings = config["training"]
    model_settings = config["model"]
    data_settings = config["dataset"]
    model_type = model_settings.get("type", "baseline")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model.type must be one of {MODEL_TYPES}, got {model_type!r}")
    continuous = model_type == "continuous"
    output_dir = Path(config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    start_epoch, resume_checkpoint = resolve_resume_checkpoint(resume_from, output_dir)
    seed = int(settings["seed"]) + start_epoch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(model_settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The config requests CUDA, but CUDA is unavailable.")

    precision = settings.get("mixed_precision", "no")
    # YAML 1.1 treats an unquoted `no` as False.
    if precision is False:
        precision = "no"
    if precision not in ("no", "fp16", "bf16"):
        raise ValueError("training.mixed_precision must be no, fp16, or bf16")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; use mixed_precision: no on CPU")
    if precision == "bf16" and device.type not in ("cuda", "cpu"):
        raise ValueError("bf16 training requires CUDA or CPU")

    accumulation = int(settings["gradient_accumulation_steps"])
    batch_size = int(settings["batch_size"])
    epochs = int(epochs_override if epochs_override is not None else settings["epochs"])
    if min(accumulation, batch_size, epochs) < 1:
        raise ValueError("gradient_accumulation_steps, batch_size, and epochs must be positive")

    dataset_type = data_settings.get("type", "magicbrush")
    if dataset_type == "magicbrush":
        train_dataset, _ = load_magicbrush(
            name=data_settings.get("name", "osunlp/MagicBrush"),
            resolution=int(data_settings["resolution"]),
            streaming=bool(data_settings.get("streaming", False)),
            include_original=continuous,
        )
    elif dataset_type == "instructpix2pix" and not continuous:
        train_dataset = load_instructpix2pix(
            name=data_settings.get("name", "timbrooks/instructpix2pix-clip-filtered"),
            resolution=int(data_settings["resolution"]),
            streaming=bool(data_settings.get("streaming", True)),
        )
    elif dataset_type == "ultraedit" and not continuous:
        train_dataset = load_ultraedit(
            name=data_settings.get("name", "BleachNick/UltraEdit_500k"),
            split=data_settings.get("split", "FreeForm"),
            resolution=int(data_settings["resolution"]),
            streaming=bool(data_settings.get("streaming", True)),
            edit_category=data_settings.get("edit_category"),
        )
    else:
        raise ValueError(f"Unsupported dataset.type {dataset_type!r} for model.type {model_type!r}")
    if data_settings.get("streaming", False):
        train_dataset = train_dataset.shuffle(
            buffer_size=int(data_settings.get("shuffle_buffer_size", 1000)), seed=seed
        )
    else:
        train_dataset = train_dataset.shuffle(seed=seed)
    max_samples = data_settings.get("max_train_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples < 1:
            raise ValueError("dataset.max_train_samples must be positive")
    evaluation_settings = config.get("evaluation", {})
    evaluate_during_training = not continuous and bool(evaluation_settings.get("enabled", False))
    evaluation_source = evaluation_settings.get("source", "magicbrush_dev")
    holdout_rows = None
    if evaluate_during_training and evaluation_source == "heldout":
        holdout_count = int(evaluation_settings.get("num_examples", 24))
        if holdout_count < 1:
            raise ValueError("evaluation.num_examples must be positive")
        if not data_settings.get("streaming", False) or not data_settings.get("materialize", False) or max_samples is None:
            raise ValueError("evaluation.source=heldout requires streaming, materialize, and max_train_samples")
        materialize_start = perf_counter()
        selected = list(islice(train_dataset, max_samples + holdout_count))
        if len(selected) < max_samples + holdout_count:
            raise RuntimeError("Too few filtered examples for the requested train and held-out sets")
        train_dataset = selected[:max_samples]
        holdout_rows = selected[max_samples:]
        print(f"Materialized {len(train_dataset)} training and {len(holdout_rows)} held-out "
              f"examples in {perf_counter() - materialize_start:.1f}s", flush=True)
    elif max_samples is not None:
        train_dataset = (train_dataset.take(max_samples) if data_settings.get("streaming", False)
                         else train_dataset.select(range(min(max_samples, len(train_dataset)))))
    if data_settings.get("materialize", False) and holdout_rows is None:
        materialize_start = perf_counter()
        train_dataset = list(train_dataset)
        if not train_dataset:
            raise RuntimeError("The selected training subset is empty")
        print(f"Materialized {len(train_dataset)} fixed training examples in "
              f"{perf_counter() - materialize_start:.1f}s", flush=True)
    loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch)

    model = build_model(model_type, model_settings, settings, device, resume_checkpoint)
    if resume_checkpoint:
        print(f"Continuing from epoch {start_epoch}: {resume_checkpoint}")
        print("AdamW and gradient-scaler state start fresh; only UNet weights were saved.")
    optimizer = torch.optim.AdamW(
        model.unet.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")
    output_dir.mkdir(parents=True, exist_ok=True)

    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    model.train()
    model.vae.eval()
    model.text_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    final_epoch = start_epoch + epochs
    max_grad_norm = float(settings.get("max_grad_norm", 1.0))
    max_train_steps = settings.get("max_train_steps")
    if max_train_steps is not None:
        max_train_steps = int(max_train_steps)
        if max_train_steps < 1:
            raise ValueError("training.max_train_steps must be positive")
    save_every_epochs = int(settings.get("save_every_epochs", 1))
    if save_every_epochs < 0:
        raise ValueError("training.save_every_epochs must be nonnegative")
    evaluation_every_steps = int(evaluation_settings.get("every_steps", 2000))
    if evaluate_during_training and evaluation_every_steps < 1:
        raise ValueError("evaluation.every_steps must be positive")
    if evaluate_during_training and evaluation_source in ("train_subset", "heldout"):
        selected_rows = train_dataset if evaluation_source == "train_subset" else holdout_rows
        if not isinstance(selected_rows, list):
            raise ValueError("evaluation.source=train_subset requires dataset.materialize: true")
        evaluation_examples = [
            {
                "source": torch.as_tensor(row["source"]),
                "target": torch.as_tensor(row["target"]),
                "instruction": row["instruction"],
                "img_id": f"{evaluation_source}-{index}",
            }
            for index, row in enumerate(selected_rows[:int(evaluation_settings.get("num_examples", 24))])
        ]
    elif evaluate_during_training and evaluation_source == "magicbrush_dev":
        evaluation_examples = load_magicbrush_validation(
            resolution=int(data_settings["resolution"]),
            count=int(evaluation_settings.get("num_examples", 24)),
            seed=int(settings["seed"]),
            name=evaluation_settings.get("name", "osunlp/MagicBrush"),
        )
    elif evaluate_during_training:
        raise ValueError(f"Unsupported evaluation.source {evaluation_source!r}")
    else:
        evaluation_examples = None
    global_step = 0
    last_evaluation_step = None
    if evaluate_during_training:
        evaluate_baseline(model, evaluation_examples, output_dir, config,
                          start_epoch, global_step, precision)
        last_evaluation_step = global_step
    for epoch in range(start_epoch + 1, final_epoch + 1):
        epoch_start = perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch - 1)
        pending = 0
        total_loss = 0.0
        batches = 0
        optimizer_steps = 0
        skipped_steps = 0
        reached_max_steps = False
        evaluation_seconds = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{final_epoch}"):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            autocast = (torch.autocast(device_type=device.type, dtype=autocast_dtype)
                        if precision != "no" else nullcontext())
            with autocast:
                if continuous:
                    original = batch["original"].to(device)
                    loss = model(original, source, target, batch["instruction"])
                else:
                    loss = model(source, target, batch["instruction"])
            scaler.scale(loss / accumulation).backward()
            total_loss += loss.detach().item()
            batches += 1
            pending += 1
            if pending == accumulation:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_grad_norm)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.is_enabled() and scaler.get_scale() < old_scale:
                    skipped_steps += 1
                else:
                    optimizer_steps += 1
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                if evaluate_during_training and global_step and global_step % evaluation_every_steps == 0 \
                        and last_evaluation_step != global_step:
                    evaluation_start = perf_counter()
                    evaluate_baseline(model, evaluation_examples, output_dir, config,
                                      epoch, global_step, precision)
                    evaluation_seconds += perf_counter() - evaluation_start
                    last_evaluation_step = global_step
                if max_train_steps is not None and global_step >= max_train_steps:
                    reached_max_steps = True
                    break

        if batches == 0:
            raise RuntimeError("The training dataset yielded no batches.")
        if pending:
            # Correct the scale of a short final accumulation group.
            for parameter in model.unet.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(accumulation / pending)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_grad_norm)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < old_scale:
                skipped_steps += 1
            else:
                optimizer_steps += 1
                global_step += 1
            optimizer.zero_grad(set_to_none=True)
        epoch_seconds = perf_counter() - epoch_start
        train_seconds = max(epoch_seconds - evaluation_seconds, 1e-9)
        summary = (f"Epoch {epoch}: mean loss {total_loss / batches:.4f}; "
                   f"optimizer steps {optimizer_steps}, skipped {skipped_steps}; "
                   f"train {train_seconds:.1f}s, evaluation {evaluation_seconds:.1f}s, "
                   f"steps/s {optimizer_steps / train_seconds:.3f}")
        if device.type == "cuda":
            summary += f"; peak CUDA allocated {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB"
        if save_every_epochs and epoch % save_every_epochs == 0:
            checkpoint = output_dir / f"epoch-{epoch}" / "unet"
            model.save_model(checkpoint)
            summary += f"; saved {checkpoint}"
        print(summary, flush=True)
        if evaluate_during_training and last_evaluation_step != global_step and \
                (evaluation_settings.get("at_epoch_end", True) or reached_max_steps):
            evaluate_baseline(model, evaluation_examples, output_dir, config,
                              epoch, global_step, precision)
            last_evaluation_step = global_step
        if reached_max_steps:
            break

    final_checkpoint = output_dir / "final" / "unet"
    model.save_model(final_checkpoint)
    print(f"Saved final UNet to {final_checkpoint}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument(
        "--resume-from", type=Path,
        help="epoch-N/unet checkpoint, final/unet, or 'latest' in the configured output directory",
    )
    parser.add_argument("--epochs", type=int, help="Epochs to run in this job (overrides training.epochs)")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train(config, resume_from=args.resume_from, epochs_override=args.epochs)


if __name__ == "__main__":
    main()

