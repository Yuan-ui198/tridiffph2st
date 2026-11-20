# TriDiffPH2ST

TriDiffPH2ST is a modular tri-branch framework for predicting spot-level spatial gene expression from routine H&E histology. The model combines molecularly informed diffusion features, domain-adapted pathology semantics, and higher-order spatial context before gated fusion and gene-expression decoding.

## Architecture

- **Diffusion branch:** masked gene-conditioned diffusion pretraining and image-gene contrastive alignment provide morphology features associated with transcriptional variation. At inference, the frozen diffusion U-Net aggregates features across multiple noise levels and decoder scales.
- **Semantic branch (DA):** a pathology image-text encoder is adapted with increased representation of digestive-system and adenocarcinoma-related samples, then frozen for semantic feature extraction and zero-shot tissue annotation.
- **Spatial branch (H3):** coordinate-defined neighbourhoods and frozen visual embeddings are used to model higher-order local tissue microenvironment interactions.
- **Fusion and prediction:** the three representations are combined by the TriDiffPH2ST prediction head to reconstruct the target gene panel.

## Included code

This repository intentionally contains only the seven core Python entry points requested for the code release:

```text
autodl/run_downstream.py
src/main.py
src/model/TriDiffPH2ST/TriDiffPH2ST.py
src/model/model_interface.py
src/data_interface.py
scripts/pretrain_diffusion_masked_contrastive.py
clip_pretrain/train.py
```

`autodl/run_downstream.py` launches downstream cross-validation through `src/main.py`. `src/data_interface.py` constructs the Lightning data module, `src/model/model_interface.py` wraps optimization and evaluation, and `src/model/TriDiffPH2ST/TriDiffPH2ST.py` defines the main architecture. The two training scripts implement diffusion pretraining and semantic encoder pretraining, respectively.

## Requirements

The code was developed with Python and PyTorch. A complete run also requires the project's training stack, including PyTorch Lightning, TorchMetrics, NumPy, PyYAML, tqdm, OpenCLIP, and the packages used by the local diffusion, dataset, and pathology-encoder modules. Install versions compatible with the checkpoints and configuration files used in your experiment.

## Running from the full project

The seven-file snapshot is a core-code release rather than a standalone package. The commands below illustrate the entry points when the missing project assets are available:

```bash
# Downstream cross-validation
python autodl/run_downstream.py config/autodl/final/TriDiffPH2ST_HER2ST_final --all-folds

# Masked-contrastive diffusion pretraining
python scripts/pretrain_diffusion_masked_contrastive.py \
  --config config/pretrain/diffusion_masked_contrastive_hespotex5.yaml

# Domain-adapted semantic pretraining
python clip_pretrain/train.py \
  --config clip_pretrain/configs/adenocarcinoma.yaml
```

Configuration files, datasets, spot-aligned H&E crops, UNI embeddings, pretrained weights, diffusion modules, dataset implementations, and auxiliary utilities are deliberately excluded from this minimal repository. To execute the code, place the snapshot back into the corresponding full project layout and provide the required assets through the configuration files. No datasets or model checkpoints are redistributed here.

## Reproducibility

Downstream evaluation uses the fold and configuration supplied to `src/main.py`. Paths to data, checkpoints, output directories, and pretrained encoders should be set in the local configuration rather than hard-coded into this repository.

## Citation

If you use this code, please cite the TriDiffPH2ST paper:

```text
TriDiffPH2ST: A modular tri-branch framework for histology-guided spatial transcriptomic prediction.
```
