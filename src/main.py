
import os
import argparse
from datetime import datetime
from glob import glob 
from pathlib import Path
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import loggers as pl_loggers

from model.model_interface import ModelInterface, CustomWriter
from data_interface import DataInterface
from utils import ( load_callbacks, 
                    load_config, 
                    load_loggers, 
                    fix_seed )

torch.set_float32_matmul_precision('high')
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def register_trusted_checkpoint_globals():
    """Allow trusted local Lightning checkpoints to load under PyTorch 2.6+."""
    # Lightning checkpoints created by this project contain the addict.Dict
    # configuration saved by save_hyperparameters(). PyTorch 2.6+ defaults to
    # weights_only=True when loading; allowlist only this known local type.
    try:
        from addict import Dict as AddictDict
        torch.serialization.add_safe_globals([AddictDict])
        # Lightning calls torch.load without an explicit weights_only value.
        # This trusted local checkpoint contains optimizer/config state, so
        # force the legacy full-checkpoint loader for this resume process.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    except (ImportError, AttributeError):
        pass


def resolve_project_path(path_value):
    if path_value is None or path_value == "":
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)

    # AutoDL final datasets live under dataset/. Keep old data/ configs
    # readable while preferring the synchronized dataset/ layout when present.
    candidates = [PROJECT_ROOT / path]
    if path.parts and path.parts[0].lower() == "data":
        candidates.insert(0, PROJECT_ROOT / "dataset" / Path(*path.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])

def get_parse():
    parser = argparse.ArgumentParser()
    
    # Main configuration
    parser.add_argument('--config_name', type=str, default='config/autodl/final/TriDiffPH2ST_HER2ST_final', help='Path to the configuration file for the experiment.')
    parser.add_argument('--mode', type=str, default='cv', help='Mode of operation: "cv" for cross-validation, "eval" for evaluation, "inference" for inference')
    # Acceleration 
    parser.add_argument('--gpu', type=int, default=1, help='Number of gpus to use')
    # Experiments
    parser.add_argument('--exp_id', type=int, default=0, help='Experiment ID for tracking different runs')
    # Others
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation')
    parser.add_argument('--all_folds', action='store_true', default=False, help='Run all folds instead of only --fold.')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override max epochs for quick PyCharm runs.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Optional checkpoint path for evaluation/inference.')
    parser.add_argument('--diffusion_ckpt_path', type=str, default=None, help='Optional frozen diffusion checkpoint path for DiffPH2ST.')
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Resume cv training from a Lightning checkpoint, e.g. logs/.../last.ckpt.')
    parser.add_argument('--run_tag', type=str, default=None, help='Optional unique suffix for a new timestamped run directory.')
    parser.add_argument(
        '--ablation',
        choices=['full', 'no_diffusion', 'no_omiclip', 'no_h3'],
        default='full',
        help='Controlled large-module ablation variant.',
    )

    args = parser.parse_args()
    
    return args

def main(cfg):    
    warnings.filterwarnings("ignore", message="Some cells have zero counts")

    if getattr(cfg.MODEL, "resume_ckpt_path", None):
        register_trusted_checkpoint_globals()
    
    fix_seed(cfg.GENERAL.seed)
    
    # Configurations
    mode = cfg.DATA.mode
    gpus = cfg.GENERAL.gpu
    
    # Define Data 
    DataInterface_dict = {'dataset_name': cfg.DATA.dataset_name,
                        'data_config': cfg.DATA,
                        'config': cfg}
    dm = DataInterface(**DataInterface_dict)
    
    # Define model
    ModelInterface_dict = {'model_name': cfg.MODEL.model_name,
                            'config': cfg}
    
    # Train or test model
    if mode == 'cv':
        # Load loggers and callbacks for Trainer
        loggers = load_loggers(cfg)
        callbacks = load_callbacks(cfg)

        
        model = ModelInterface(**ModelInterface_dict)
        
        # Instancialize Trainer 
        strategy = DDPStrategy(find_unused_parameters=True) if gpus > 1 else "auto"
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy = strategy,
            devices = gpus,
            max_epochs = cfg.TRAINING.num_epochs,
            logger = loggers,
            check_val_every_n_epoch = 1,
            callbacks = callbacks,
            precision = '16-mixed',
            enable_progress_bar=True,
            log_every_n_steps=1
        )
        
        trainer.fit(model, datamodule = dm, ckpt_path=getattr(cfg.MODEL, "resume_ckpt_path", None))
        
    elif mode == 'eval':
        log_path = Path(cfg.GENERAL.log_path)
        config_path = Path(cfg.config)
        if config_path.suffix:
            config_path = config_path.with_suffix("")
        model_name = config_path.name
        log_name = config_path.parent
        if log_name.parts and log_name.parts[0].lower() == "config":
            log_name = Path(*log_name.parts[1:]) if len(log_name.parts) > 1 else Path("default")

        ckpt_candidates = sorted((log_path / log_name / model_name).glob(f"*/fold{cfg.DATA.fold}/*.ckpt"))
        if not ckpt_candidates:
            raise FileNotFoundError(f"No checkpoint found under {log_path / log_name / model_name} for fold{cfg.DATA.fold}")
        ckpt_path = str(ckpt_candidates[-1])
        ckpt_dir = str(Path(ckpt_path).parents[1])
        current_time = Path(ckpt_dir).name
        
        output_path = f"{cfg.DATA.output_dir}/{model_name}"
        os.makedirs(f"{output_path}/fold{cfg.DATA.fold}", exist_ok=True)
        cfg.DATA.output_path = output_path
    
        csv_logger = pl_loggers.CSVLogger(str(log_path / log_name),
                                    name = f"{model_name}/{current_time}", version = f'fold{cfg.DATA.fold}/eval', )
        
        trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            devices=gpus,
                            precision = '16-mixed',
                            logger=[csv_logger])
        
        # checkpoint = cfg.GENERAL.ckpt_path
        model = ModelInterface.load_from_checkpoint(ckpt_path, **ModelInterface_dict)
        
        trainer.test(model, datamodule = dm)
        
    elif mode == 'inference':
        model_name = Path(cfg.config).name

        pred_path = f"{cfg.DATA.output_dir}/{model_name}/fold{cfg.DATA.fold}"
        os.makedirs(pred_path, exist_ok=True)
        
        pred_writer = CustomWriter(pred_dir=pred_path, write_interval="epoch")
        trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            devices=gpus, 
                            callbacks=[pred_writer],
                            precision = '16-mixed',
                            logger=False)

        model = ModelInterface.load_from_checkpoint(cfg.MODEL.ckpt_path, **ModelInterface_dict)
        
        trainer.predict(model, datamodule = dm, return_predictions=False)

    else:
        raise Exception("Invalid mode")
    
    return model

if __name__ == '__main__':
    args = get_parse()   
    cfg = load_config(args.config_name)

    if args.ablation == 'no_diffusion':
        cfg.MODEL.use_diffusion_branch = False
    elif args.ablation == 'no_omiclip':
        cfg.MODEL.use_omiclip_branch = False
    elif args.ablation == 'no_h3':
        cfg.MODEL.use_h3_branch = False
    if args.ablation != 'full':
        cfg.GENERAL.log_experiment_name = f'TriDiffPH2ST_ABLATION_HESTBowel_{args.ablation}'

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_tag = str(args.run_tag or getattr(cfg.GENERAL, "run_tag", "") or "").strip()
    if args.resume_ckpt_path:
        # A resumed run keeps writing to its original timestamped directory.
        resume_path = Path(args.resume_ckpt_path)
        if resume_path.name == "last.ckpt" and resume_path.parent.name.startswith("fold"):
            timestamp = resume_path.parent.parent.name
    elif run_tag:
        timestamp = f"{timestamp}-{run_tag}"
    cfg.GENERAL.timestamp = timestamp
    cfg.config = args.config_name
    cfg.GENERAL.exp_id = args.exp_id
    cfg.GENERAL.gpu = args.gpu
    if args.max_epochs is not None:
        cfg.TRAINING.num_epochs = args.max_epochs
    cfg.MODEL.ckpt_path = args.ckpt_path
    if args.diffusion_ckpt_path is not None:
        cfg.MODEL.diffusion_ckpt_path = args.diffusion_ckpt_path
    cfg.MODEL.resume_ckpt_path = args.resume_ckpt_path
    cfg.DATA.mode = args.mode
    cfg.DATA.data_dir = resolve_project_path(getattr(cfg.DATA, "data_dir", None) or "data/her2st")
    cfg.DATA.output_dir = resolve_project_path(getattr(cfg.DATA, "output_dir", None) or "output/pred/ST/andrew")
    cfg.GENERAL.log_path = resolve_project_path(getattr(cfg.GENERAL, "log_path", None) or "logs")
    cfg.DATA.dataset_name = getattr(cfg.DATA, "dataset_name", None) or "TriDataset"
    cfg.DATA.cpm = getattr(cfg.DATA, "cpm", False)
    cfg.DATA.smooth = getattr(cfg.DATA, "smooth", False)
    
    if args.mode == 'cv':
        folds = range(cfg.TRAINING.num_k) if args.all_folds else [args.fold]
        for fold in folds:
            cfg.DATA.fold = fold
            main(cfg)
    elif args.mode == 'eval':
        cfg.DATA.fold = args.fold
        main(cfg)
    else:
        cfg.DATA.fold = args.fold
        main(cfg)
