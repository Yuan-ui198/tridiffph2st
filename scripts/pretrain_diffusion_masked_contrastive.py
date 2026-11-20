import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_OUT_DIR = PROJECT_ROOT / "checkpoints" / "diffusion" / "hespotex5_masked_contrastive"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from diffusion import DiffusionPrecond, MultiScaleDiffusionFusion
from diffusion.dataset import MultiGenePatchDataset, collate_open_gene_patch
from diffusion.losses import MaskedContrastiveEDM2Loss


class EMA:
    def __init__(self, modules, decay=0.999):
        self.decay = decay
        self.shadow = self._snapshot(modules)

    @staticmethod
    def _snapshot(modules):
        shadow = {}
        for prefix, module in modules.items():
            for name, value in module.state_dict().items():
                shadow[f"{prefix}.{name}"] = value.detach().cpu().clone()
        return shadow

    @torch.no_grad()
    def update(self, modules):
        current = self._snapshot(modules)
        for name, value in current.items():
            if not torch.is_floating_point(value):
                self.shadow[name] = value.clone()
            else:
                self.shadow[name].mul_(self.decay).add_(value, alpha=1.0 - self.decay)


def parse_list(values):
    result = []
    for value in values or []:
        result.extend([part for part in str(value).split(",") if part])
    return result


def parse_optional_int_list(values):
    parsed = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if not part or part.lower() in {"none", "null"}:
                parsed.append(None)
            else:
                parsed.append(int(part))
    return parsed


def resolve_project_path(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def infer_vocab_size(gene_files):
    import h5py

    max_gene_id = -1
    for gene_file in gene_files:
        with h5py.File(gene_file, "r") as f:
            max_gene_id = max(max_gene_id, int(f["gene_ids"][:].max()))
    return max_gene_id + 1


def require_files(paths, label):
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing {label} files:\n  {joined}")


def describe_h5_pairs(patch_files, gene_files, label, max_samples_per_dataset=None):
    import h5py

    print(f"\n{label} datasets:")
    max_samples_per_dataset = max_samples_per_dataset or [None] * len(patch_files)
    for patch_file, gene_file, max_samples in zip(patch_files, gene_files, max_samples_per_dataset):
        with h5py.File(patch_file, "r") as p, h5py.File(gene_file, "r") as g:
            n_img = int(p["img"].shape[0])
            n_gene = int(g["genes"].shape[0])
            n_panel = int(g["gene_ids"].shape[0])
        effective = min(n_img, int(max_samples)) if max_samples is not None and int(max_samples) > 0 else n_img
        suffix = f", effective_samples={effective}" if effective != n_img else ""
        print(f"  {Path(patch_file).parent}: samples={n_img}{suffix}, genes={n_panel}")
        if n_img != n_gene:
            raise ValueError(f"Patch/gene sample mismatch: {patch_file} has {n_img}, {gene_file} has {n_gene}")


def split_ema_state(ema_shadow):
    diffusion = {}
    image_head = {}
    for name, value in ema_shadow.items():
        if name.startswith("diffusion."):
            diffusion[name.replace("diffusion.", "", 1)] = value
        elif name.startswith("image_head."):
            image_head[name.replace("image_head.", "", 1)] = value
    return diffusion, image_head


def apply_modules_device(modules, device):
    for module in modules.values():
        module.to(device)


def run_epoch(modules, loader, loss_fn, optimizer, ema, device, max_steps=None, train=True, epoch=0):
    diffusion = modules["diffusion"]
    image_head = modules["image_head"]
    diffusion.train(train)
    image_head.train(train)
    totals = {"loss": 0.0, "denoising_loss": 0.0, "contrastive_loss": 0.0}
    total_steps = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        total = len(loader) if max_steps is None else min(len(loader), max_steps)
        desc = f"epoch {epoch} train" if train else f"epoch {epoch} valid"
        progress = tqdm(loader, total=total, desc=desc, dynamic_ncols=True, leave=True)
        for patches, genes, gene_ids, gene_mask in progress:
            patches = patches.to(device) / 127.5 - 1
            genes = genes.to(device)
            gene_ids = gene_ids.to(device)
            gene_mask = gene_mask.to(device) if gene_mask is not None else None
            if train:
                optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(diffusion, image_head, patches, gene_values=genes, gene_ids=gene_ids, gene_mask=gene_mask)
            loss = losses["loss"]
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(diffusion.parameters()) + list(image_head.parameters()),
                    1.0,
                )
                optimizer.step()
                if ema is not None:
                    ema.update(modules)
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            total_steps += 1
            progress.set_postfix(
                loss=f"{totals['loss'] / total_steps:.4f}",
                den=f"{totals['denoising_loss'] / total_steps:.4f}",
                con=f"{totals['contrastive_loss'] / total_steps:.4f}",
            )
            if max_steps is not None and total_steps >= max_steps:
                break
    return {key: value / max(total_steps, 1) for key, value in totals.items()}, total_steps


def save_checkpoint(path, modules, optimizer, ema, args, epoch, global_step, valid_metrics=None):
    diffusion = modules["diffusion"]
    image_head = modules["image_head"]
    payload = {
        "model": diffusion.cpu().state_dict(),
        "image_head": image_head.cpu().state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "global_step": global_step,
        "valid_loss": None if valid_metrics is None else valid_metrics["loss"],
        "valid_metrics": valid_metrics,
    }
    if ema is not None:
        ema_diffusion, ema_image_head = split_ema_state(ema.shadow)
        payload["ema"] = ema_diffusion
        payload["image_head_ema"] = ema_image_head
    else:
        payload["ema"] = diffusion.cpu().state_dict()
        payload["image_head_ema"] = image_head.cpu().state_dict()
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path, modules, optimizer, ema, device):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    modules["diffusion"].load_state_dict(payload.get("model", payload), strict=True)
    if "image_head" in payload:
        modules["image_head"].load_state_dict(payload["image_head"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and "ema" in payload:
        shadow = {}
        for name, value in payload["ema"].items():
            shadow[f"diffusion.{name}"] = value.detach().cpu().clone()
        for name, value in payload.get("image_head_ema", payload.get("image_head", {})).items():
            shadow[f"image_head.{name}"] = value.detach().cpu().clone()
        ema.shadow = shadow
    apply_modules_device(modules, device)
    return int(payload.get("epoch", 0)), int(payload.get("global_step", 0)), payload.get("valid_loss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "pretrain" / "diffusion_masked_contrastive_hespotex5.yaml"))
    parser.add_argument("--patch_path", action="append", default=[])
    parser.add_argument("--gene_path", action="append", default=[])
    parser.add_argument("--valid_patch_path", action="append", default=[])
    parser.add_argument("--valid_gene_path", action="append", default=[])
    parser.add_argument("--max_samples_per_dataset", action="append", default=[], help="Optional train sample caps per dataset. Use comma list, e.g. none,none,none,10000,none.")
    parser.add_argument("--valid_max_samples_per_dataset", action="append", default=[], help="Optional validation sample caps per dataset.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--model_channels", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_gene_blocks", type=int, default=2)
    parser.add_argument("--gene_mask_ratio", type=float, default=0.15)
    parser.add_argument("--contrastive_weight", type=float, default=0.05)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--balance_datasets", action="store_true", default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--valid_max_steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = {}
    if args.config and Path(args.config).is_file():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    optim_cfg = cfg.get("optimization", {})
    contrast_cfg = cfg.get("contrastive", {})

    train_patch_files = [resolve_project_path(path) for path in (parse_list(args.patch_path) or train_cfg.get("patch_paths", []))]
    train_gene_files = [resolve_project_path(path) for path in (parse_list(args.gene_path) or train_cfg.get("gene_paths", []))]
    valid_patch_files = [resolve_project_path(path) for path in (parse_list(args.valid_patch_path) or train_cfg.get("valid_patch_paths", []))]
    valid_gene_files = [resolve_project_path(path) for path in (parse_list(args.valid_gene_path) or train_cfg.get("valid_gene_paths", []))]
    train_max_samples = parse_optional_int_list(args.max_samples_per_dataset) or train_cfg.get("max_samples_per_dataset")
    valid_max_samples = parse_optional_int_list(args.valid_max_samples_per_dataset) or train_cfg.get("valid_max_samples_per_dataset")
    if args.out_dir == str(DEFAULT_OUT_DIR):
        args.out_dir = resolve_project_path(cfg.get("out_dir", args.out_dir))
    else:
        args.out_dir = resolve_project_path(args.out_dir)
    defaults = parser.parse_args([])
    for name in ("epochs", "batch_size", "valid_batch_size", "lr", "ema_decay", "balance_datasets", "device", "max_steps", "valid_max_steps"):
        if name in optim_cfg and optim_cfg[name] is not None and getattr(args, name) == getattr(defaults, name):
            setattr(args, name, optim_cfg[name])
    for name in ("vocab_size", "model_channels", "embed_dim", "num_gene_blocks"):
        if name in model_cfg and model_cfg[name] is not None and getattr(args, name) == getattr(defaults, name):
            setattr(args, name, model_cfg[name])
    for name in ("gene_mask_ratio", "contrastive_weight", "contrastive_temperature"):
        if name in contrast_cfg and getattr(args, name) == getattr(defaults, name):
            setattr(args, name, contrast_cfg[name])
    args.no_ema = bool(optim_cfg.get("no_ema", args.no_ema))

    if len(train_patch_files) != len(train_gene_files):
        raise ValueError("--patch_path and --gene_path counts must match")
    if len(valid_patch_files) != len(valid_gene_files):
        raise ValueError("--valid_patch_path and --valid_gene_path counts must match")
    if train_max_samples is None:
        train_max_samples = [None] * len(train_patch_files)
    if valid_max_samples is None:
        valid_max_samples = [None] * len(valid_patch_files)
    if len(train_max_samples) != len(train_patch_files):
        raise ValueError("train.max_samples_per_dataset must match patch_paths length")
    if len(valid_max_samples) != len(valid_patch_files):
        raise ValueError("train.valid_max_samples_per_dataset must match valid_patch_paths length")
    if not train_patch_files:
        raise ValueError("No training datasets provided")
    require_files(train_patch_files, "training patch")
    require_files(train_gene_files, "training gene")
    require_files(valid_patch_files, "validation patch")
    require_files(valid_gene_files, "validation gene")
    describe_h5_pairs(train_patch_files, train_gene_files, "Train", train_max_samples)
    if valid_patch_files:
        describe_h5_pairs(valid_patch_files, valid_gene_files, "Valid", valid_max_samples)

    vocab_size = args.vocab_size or infer_vocab_size(train_gene_files + valid_gene_files)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pretrain_config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "vocab_size": vocab_size}, f, indent=2)

    train_dataset = MultiGenePatchDataset(
        train_patch_files,
        train_gene_files,
        balance_datasets=args.balance_datasets,
        max_samples_per_dataset=train_max_samples,
        seed=0,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_open_gene_patch)
    valid_loader = None
    if valid_patch_files:
        valid_dataset = MultiGenePatchDataset(
            valid_patch_files,
            valid_gene_files,
            balance_datasets=False,
            max_samples_per_dataset=valid_max_samples,
            seed=10_000,
        )
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=0, collate_fn=collate_open_gene_patch)

    diffusion = DiffusionPrecond(
        img_resolution=224,
        img_channels=3,
        vocab_size=vocab_size,
        model_channels=args.model_channels,
        embed_dim=args.embed_dim,
        num_gene_blocks=args.num_gene_blocks,
        use_fp16=True,
    ).to(args.device)
    image_head = MultiScaleDiffusionFusion(out_dim=args.embed_dim, model_channels=args.model_channels, dropout=0.1).to(args.device)
    modules = {"diffusion": diffusion, "image_head": image_head}
    optimizer = torch.optim.AdamW(list(diffusion.parameters()) + list(image_head.parameters()), lr=args.lr)
    ema = None if args.no_ema else EMA(modules, decay=args.ema_decay)
    loss_fn = MaskedContrastiveEDM2Loss(
        gene_mask_ratio=args.gene_mask_ratio,
        contrastive_weight=args.contrastive_weight,
        temperature=args.contrastive_temperature,
    )

    metrics_path = out_dir / "metrics.csv"
    start_epoch = 0
    global_step = 0
    best_valid = float("inf")
    if args.resume:
        start_epoch, global_step, resume_valid = load_checkpoint(Path(resolve_project_path(args.resume)), modules, optimizer, ema, args.device)
        best_valid = float(resume_valid) if resume_valid is not None else best_valid
        print(f"Resumed from {args.resume}: epoch={start_epoch}, global_step={global_step}, valid_loss={resume_valid}")

    with open(metrics_path, "a" if args.resume else "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "global_step", "train_loss", "train_denoising_loss", "train_contrastive_loss", "valid_loss", "valid_denoising_loss", "valid_contrastive_loss"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not args.resume:
            writer.writeheader()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics, steps = run_epoch(modules, train_loader, loss_fn, optimizer, ema, args.device, max_steps=args.max_steps, train=True, epoch=epoch)
        global_step += steps
        valid_metrics = None
        if valid_loader is not None:
            valid_metrics, _ = run_epoch(modules, valid_loader, loss_fn, optimizer, ema=None, device=args.device, max_steps=args.valid_max_steps, train=False, epoch=epoch)

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "global_step", "train_loss", "train_denoising_loss", "train_contrastive_loss", "valid_loss", "valid_denoising_loss", "valid_contrastive_loss"])
            writer.writerow(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_metrics["loss"],
                    "train_denoising_loss": train_metrics["denoising_loss"],
                    "train_contrastive_loss": train_metrics["contrastive_loss"],
                    "valid_loss": "" if valid_metrics is None else valid_metrics["loss"],
                    "valid_denoising_loss": "" if valid_metrics is None else valid_metrics["denoising_loss"],
                    "valid_contrastive_loss": "" if valid_metrics is None else valid_metrics["contrastive_loss"],
                }
            )

        print(f"epoch={epoch} global_step={global_step} train={train_metrics} valid={valid_metrics}")
        apply_modules_device(modules, "cpu")
        save_checkpoint(out_dir / "diffusion_last.pkl", modules, optimizer, ema, args, epoch, global_step, valid_metrics)
        if valid_metrics is not None and valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            save_checkpoint(out_dir / "diffusion_best.pkl", modules, optimizer, ema, args, epoch, global_step, valid_metrics)
        apply_modules_device(modules, args.device)

    print(f"Saved masked-contrastive diffusion checkpoints and metrics to {out_dir}")


if __name__ == "__main__":
    main()
