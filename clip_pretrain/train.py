import argparse
import csv
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from open_clip.loss import CoCaLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_pretrain.config import PROJECT_ROOT, load_config, resolve_path
from clip_pretrain.data import OmiClipTarDataset, make_collate
from clip_pretrain.model import build_model, configure_trainable, export_visual_checkpoint, parameter_summary
from clip_pretrain.prepare_index import GROUPS, build_index


DEFAULT_CONFIG = "clip_pretrain/configs/adenocarcinoma.yaml"
DEFAULT_RESUME = None


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_loader(config, split, tokenizer, db_path):
    section = config["data"]
    max_samples = section.get(f"{split}_max_samples")
    dataset = OmiClipTarDataset(
        db_path=db_path,
        split=split,
        val_ratio=section["val_ratio"],
        seed=config["seed"],
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=section.get("num_workers", 0),
        pin_memory=True,
        drop_last=split == "train",
        collate_fn=make_collate(tokenizer),
    )


def save_training_checkpoint(path, model, optimizer, scaler, epoch, global_step, best_valid, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_valid": best_valid,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
        },
        path,
    )


def load_training_checkpoint(path, model, optimizer, scaler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    return checkpoint["epoch"] + 1, checkpoint["global_step"], checkpoint.get("best_valid", float("inf"))


def compute_losses(loss_fn, outputs):
    losses = loss_fn(
        outputs["image_features"],
        outputs["text_features"],
        outputs["logits"],
        outputs["labels"],
        outputs["logit_scale"],
        output_dict=True,
    )
    losses["loss"] = losses["contrastive_loss"] + losses["caption_loss"]
    return losses


def _optimizer_step(model, optimizer, scaler, config):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        config["training"]["grad_clip"],
    )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def run_epoch(model, loader, loss_fn, optimizer, scaler, config, device, epoch, global_step, total_updates, train):
    model.train(train)
    totals = {"loss": 0.0, "contrastive_loss": 0.0, "caption_loss": 0.0}
    count = 0
    accumulation = int(config["training"]["gradient_accumulation"])
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"epoch {epoch + 1} {'train' if train else 'valid'}")
    batch_index = -1
    for batch_index, batch in enumerate(progress):
        if train and global_step >= total_updates:
            break
        images = batch["image"].to(device, non_blocking=True)
        texts = batch["text"].to(device, non_blocking=True)
        autocast = torch.autocast(device_type=device.type, dtype=torch.float16) if device.type == "cuda" else nullcontext()
        grad_context = torch.enable_grad() if train else torch.no_grad()
        with grad_context, autocast:
            outputs = model(images, texts)
            losses = compute_losses(loss_fn, outputs)
            scaled_loss = losses["loss"] / accumulation
        if train:
            scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % accumulation == 0
            if should_step:
                lr = cosine_lr(
                    global_step,
                    total_updates,
                    config["training"]["warmup_steps"],
                    config["training"]["learning_rate"],
                )
                for group_config in optimizer.param_groups:
                    group_config["lr"] = lr
                _optimizer_step(model, optimizer, scaler, config)
                global_step += 1
        batch_size = images.size(0)
        count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size
        progress.set_postfix(loss=totals["loss"] / count, con=totals["contrastive_loss"] / count, cap=totals["caption_loss"] / count)
    if train and batch_index >= 0 and (batch_index + 1) % accumulation:
        lr = cosine_lr(
            global_step,
            total_updates,
            config["training"]["warmup_steps"],
            config["training"]["learning_rate"],
        )
        for group_config in optimizer.param_groups:
            group_config["lr"] = lr
        _optimizer_step(model, optimizer, scaler, config)
        global_step += 1
    return {key: value / max(1, count) for key, value in totals.items()}, global_step


def append_metrics(path, row):
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--resume", default=DEFAULT_RESUME)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = resolve_path(config["data"]["root"])
    index_dir = resolve_path(config["data"]["index_dir"])
    group = config["data"]["group"]
    db_path = build_index(data_root, index_dir, group, overwrite=False)
    if args.smoke_test:
        config["data"]["train_max_samples"] = 4
        config["data"]["valid_max_samples"] = 2
        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 2
        config["training"]["gradient_accumulation"] = 1
        config["training"]["max_updates"] = 1

    resume = resolve_path(args.resume or config.get("resume"))
    output_root = resolve_path(config["output_dir"])
    run_name = config.get("run_name", group)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_dir = resume.parent if resume else output_root / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    init_checkpoint = resolve_path(config["model"].get("init_checkpoint"))
    model = build_model(init_checkpoint, device=device, strict=True)
    configure_trainable(model, **config["model"]["trainable"])
    if config["model"].get("grad_checkpointing", True):
        model.set_grad_checkpointing(True)
    summary = parameter_summary(model)
    print(f"Model parameters: total={summary['total']:,}, trainable={summary['trainable']:,}")

    tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")
    train_loader = build_loader(config, "train", tokenizer, db_path)
    valid_loader = build_loader(config, "valid", tokenizer, db_path)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        eps=config["training"]["eps"],
        weight_decay=config["training"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = CoCaLoss(
        caption_loss_weight=config["loss"]["caption_weight"],
        clip_loss_weight=config["loss"]["contrastive_weight"],
        pad_id=0,
        cache_labels=True,
    )

    start_epoch = 0
    global_step = 0
    best_valid = float("inf")
    if resume:
        start_epoch, global_step, best_valid = load_training_checkpoint(resume, model, optimizer, scaler)
        print(f"Resumed from {resume}: epoch={start_epoch}, global_step={global_step}")

    metrics_path = run_dir / "metrics.csv"
    total_updates = max(1, math.ceil(len(train_loader) / config["training"]["gradient_accumulation"]) * config["training"]["epochs"])
    configured_max_updates = config["training"].get("max_updates")
    if configured_max_updates:
        total_updates = int(configured_max_updates)
    for epoch in range(start_epoch, config["training"]["epochs"]):
        train_metrics, global_step = run_epoch(
            model, train_loader, loss_fn, optimizer, scaler, config, device, epoch, global_step, total_updates, True
        )
        valid_metrics, _ = run_epoch(
            model, valid_loader, loss_fn, optimizer, scaler, config, device, epoch, global_step, total_updates, False
        )
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch + 1, "global_step": global_step, "lr": lr}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"valid_{key}": value for key, value in valid_metrics.items()})
        append_metrics(metrics_path, row)
        if valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            if not args.smoke_test:
                export_visual_checkpoint(model, run_dir / "checkpoint_visual_best.pt", {"group": group, "epoch": epoch + 1, "valid_loss": best_valid})
        if not args.smoke_test:
            save_training_checkpoint(run_dir / "last.pt", model, optimizer, scaler, epoch, global_step, best_valid, config)
            export_visual_checkpoint(model, run_dir / "checkpoint_visual_last.pt", {"group": group, "epoch": epoch + 1})
        print(row)
        if global_step >= total_updates:
            print(f"Reached max_updates={total_updates}")
            break


if __name__ == "__main__":
    main()
