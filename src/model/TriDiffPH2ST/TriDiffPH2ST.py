import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import (
    AgreementAwareThreeBranchGeneHead,
    DiffusionPrecond,
    MultiScaleDiffusionFusion,
    MultiTimestepTokenFusion,
    ThreeBranchGatedResidualGeneHead,
    extract_diffusion_features,
)
from model.DiffPH2ST.spatial import (
    HierarchicalManyBodyMicroenvironmentBranch,
    ManyBodyMicroenvironmentBranch,
    S2SpotMicroenvironmentBranch,
    SpatialHypergraphBranch,
)
from .plip import FrozenPLIPVisionEncoder


class TriDiffPH2ST(nn.Module):
    def __init__(
        self,
        num_outputs=171,
        emb_dim=1024,
        fusion_dim=512,
        diffusion_ckpt_path="",
        diffusion_vocab_size=1000,
        diffusion_model_channels=64,
        diffusion_embed_dim=512,
        diffusion_num_gene_blocks=2,
        diffusion_noise_label=0.01,
        diffusion_noise_labels=None,
        diffusion_chunk_size=0,
        diffusion_pool_ckpt_path="",
        diffusion_pool_ckpt_key="image_head_ema",
        plip_ckpt_path="",
        plip_encoder_name="plip",
        plip_chunk_size=0,
        depth1=1,
        depth2=5,
        depth3=4,
        num_heads1=4,
        num_heads2=8,
        num_heads3=4,
        mlp_ratio1=4.0,
        mlp_ratio2=4.0,
        mlp_ratio3=4.0,
        dropout1=0.4,
        dropout2=0.3,
        dropout3=0.3,
        kernel_size=3,
        res_neighbor=(5, 5),
        fusion_head="gated_residual",
        agreement_dim=128,
        predictor_hidden_dim=512,
        agreement_bias_init=0.1,
        spatial_branch_type="hypergraph",
        micro_hidden_dim=512,
        micro_num_heads=4,
        micro_many_body_topk=12,
        micro_dropout=0.3,
        micro_use_grid_position=False,
        s2_hidden_dim=512,
        s2_num_heads=4,
        s2_knn_k=8,
        s2_graph_layers=2,
        s2_graph_input_type="image",
        s2_use_official_gcn=True,
        s2_dropout=0.3,
        s2_use_grid_position=False,
        h3_hidden_dim=512,
        h3_num_heads=4,
        h3_knn_k=8,
        h3_dropout=0.3,
        h3_use_four_body=True,
        use_diffusion_branch=True,
        use_omiclip_branch=True,
        use_h3_branch=True,
    ):
        super().__init__()
        self.fusion_dim = int(fusion_dim)
        self.use_diffusion_branch = bool(use_diffusion_branch)
        self.use_omiclip_branch = bool(use_omiclip_branch)
        self.use_h3_branch = bool(use_h3_branch)
        self.diffusion_noise_label = diffusion_noise_label
        self.diffusion_noise_labels = self._parse_noise_labels(diffusion_noise_labels, diffusion_noise_label)
        self.diffusion_chunk_size = int(diffusion_chunk_size or 0)
        if self.use_diffusion_branch:
            self.diffusion = self._load_diffusion(
                diffusion_ckpt_path,
                vocab_size=diffusion_vocab_size,
                model_channels=diffusion_model_channels,
                embed_dim=diffusion_embed_dim,
                num_gene_blocks=diffusion_num_gene_blocks,
            )
            self.diffusion.requires_grad_(False)
            self.diffusion.eval()
            self.diffusion_pool = MultiScaleDiffusionFusion(
                out_dim=fusion_dim,
                model_channels=diffusion_model_channels,
                dropout=dropout1,
            )
            self._load_diffusion_pool(diffusion_pool_ckpt_path, diffusion_pool_ckpt_key)
            self.diffusion_timestep_fusion = (
                MultiTimestepTokenFusion(
                    feature_dim=fusion_dim,
                    num_timesteps=len(self.diffusion_noise_labels),
                    dropout=dropout1,
                )
                if len(self.diffusion_noise_labels) > 1
                else None
            )
        else:
            self.diffusion = None
            self.diffusion_pool = None
            self.diffusion_timestep_fusion = None

        if self.use_omiclip_branch:
            self.foundation_encoder = FrozenPLIPVisionEncoder(
                ckpt_path=plip_ckpt_path,
                out_dim=fusion_dim,
                chunk_size=plip_chunk_size,
                encoder_name=plip_encoder_name,
            )
        else:
            self.foundation_encoder = None

        spatial_branch_type = str(spatial_branch_type).lower()
        if not self.use_h3_branch:
            self.spatial_branch = None
        elif spatial_branch_type in {"s2potae", "s2spot", "s2_spot", "multiscale", "multi_scale"}:
            self.spatial_branch = S2SpotMicroenvironmentBranch(
                emb_dim=emb_dim,
                out_dim=fusion_dim,
                hidden_dim=s2_hidden_dim,
                num_heads=s2_num_heads,
                knn_k=s2_knn_k,
                graph_layers=s2_graph_layers,
                graph_input_type=s2_graph_input_type,
                st_dim=num_outputs,
                use_official_gcn=s2_use_official_gcn,
                dropout=s2_dropout,
                use_grid_position=s2_use_grid_position,
                res_neighbor=res_neighbor,
            )
        elif spatial_branch_type in {"many_body", "manybody", "microenvironment"}:
            self.spatial_branch = ManyBodyMicroenvironmentBranch(
                emb_dim=emb_dim,
                out_dim=fusion_dim,
                hidden_dim=micro_hidden_dim,
                num_heads=micro_num_heads,
                many_body_topk=micro_many_body_topk,
                dropout=micro_dropout,
                use_grid_position=micro_use_grid_position,
                res_neighbor=res_neighbor,
            )
        elif spatial_branch_type in {"h3_many_body", "h3", "hierarchical_many_body", "h3_microenvironment"}:
            self.spatial_branch = HierarchicalManyBodyMicroenvironmentBranch(
                emb_dim=emb_dim,
                out_dim=fusion_dim,
                hidden_dim=h3_hidden_dim,
                num_heads=h3_num_heads,
                knn_k=h3_knn_k,
                dropout=h3_dropout,
                use_four_body=h3_use_four_body,
            )
        else:
            self.spatial_branch = SpatialHypergraphBranch(
                emb_dim=emb_dim,
                out_dim=fusion_dim,
                global_depth=depth2,
                neighbor_depth=depth3,
                fusion_depth=depth1,
                global_heads=num_heads2,
                neighbor_heads=num_heads3,
                fusion_heads=num_heads1,
                mlp_ratio=mlp_ratio1,
                dropout=dropout2,
                kernel_size=kernel_size,
                res_neighbor=res_neighbor,
            )
        print(
            "TriDiffPH2ST branches: "
            f"diffusion={'on' if self.use_diffusion_branch else 'off'}, "
            f"omiclip={'on' if self.use_omiclip_branch else 'off'}, "
            f"h3={'on' if self.use_h3_branch else 'off'}"
        )
        if str(fusion_head).lower() in {"agreement", "agreement_aware", "agreement_aware_residual"}:
            self.head = AgreementAwareThreeBranchGeneHead(
                feature_dim=fusion_dim,
                out_dim=num_outputs,
                agreement_dim=agreement_dim,
                predictor_hidden_dim=predictor_hidden_dim,
                dropout=dropout1,
                agreement_bias_init=agreement_bias_init,
            )
        else:
            self.head = ThreeBranchGatedResidualGeneHead(
                feature_dim=fusion_dim,
                out_dim=num_outputs,
                dropout=dropout1,
            )

    def _parse_noise_labels(self, noise_labels, fallback):
        if noise_labels is None or noise_labels == "":
            return [float(fallback)]
        if isinstance(noise_labels, str):
            return [float(item.strip()) for item in noise_labels.split(",") if item.strip()]
        if isinstance(noise_labels, (list, tuple)):
            return [float(item) for item in noise_labels]
        return [float(noise_labels)]

    def _load_diffusion(self, ckpt_path, vocab_size, model_channels, embed_dim, num_gene_blocks):
        model = DiffusionPrecond(
            img_resolution=224,
            img_channels=3,
            vocab_size=vocab_size,
            model_channels=model_channels,
            embed_dim=embed_dim,
            num_gene_blocks=num_gene_blocks,
            use_fp16=True,
        )
        path = self._resolve_checkpoint_path(ckpt_path, "diffusion")
        if path is not None:
            with open(path, "rb") as f:
                ckpt = pickle.load(f)
            state = ckpt.get("ema", ckpt.get("model", ckpt))
            if isinstance(state, nn.Module):
                return state
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected:
                print(f"Unexpected diffusion checkpoint keys: {unexpected[:5]}")
            if missing:
                print(f"Missing diffusion checkpoint keys: {missing[:5]}")
        return model

    @staticmethod
    def _resolve_checkpoint_path(ckpt_path, label):
        if not ckpt_path:
            return None
        path = Path(ckpt_path)
        if not path.is_file():
            project_root = Path(__file__).resolve().parents[3]
            path = project_root / path
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {ckpt_path}")
        return path

    def _load_diffusion_pool(self, ckpt_path, ckpt_key):
        if not ckpt_path:
            return
        path = self._resolve_checkpoint_path(ckpt_path, "Diffusion pool")
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        state = ckpt.get(ckpt_key)
        if state is None and ckpt_key == "image_head_ema":
            state = ckpt.get("image_head")
        if state is None:
            raise KeyError(f"Checkpoint does not contain '{ckpt_key}' or fallback 'image_head': {ckpt_path}")
        missing, unexpected = self.diffusion_pool.load_state_dict(state, strict=False)
        if unexpected:
            print(f"Unexpected diffusion_pool checkpoint keys: {unexpected[:5]}")
        if missing:
            print(f"Missing diffusion_pool checkpoint keys: {missing[:5]}")
        print(f"Initialized diffusion_pool from {path} [{ckpt_key}]")

    def forward(self, img, mask=None, neighbor_emb=None, label=None, position=None, global_emb=None, pid=None, sid=None, dataset=None, **kwargs):
        if label is None and "label" in kwargs:
            label = kwargs["label"]
        if pid is not None and pid.dim() > 1:
            pid = pid.view(-1)
        if sid is not None and sid.dim() > 1:
            sid = sid.view(-1)
        if label is not None and label.dim() == 3:
            label = label.squeeze(0)
        if mask is not None and mask.dim() == 3:
            mask = mask.squeeze(0)
        if neighbor_emb is not None and neighbor_emb.dim() == 4:
            neighbor_emb = neighbor_emb.squeeze(0)
        if position is not None and position.dim() == 3:
            position = position.squeeze(0)
        if global_emb is not None and global_emb.dim() == 3:
            global_emb = global_emb.squeeze(0)

        foundation_feature = (
            self.foundation_encoder(img)
            if self.use_omiclip_branch
            else self._zero_feature(img)
        )
        diffusion_feature = self._encode_diffusion(img)
        if self.use_h3_branch:
            spatial_feature = self.spatial_branch(
                neighbor_emb=neighbor_emb,
                mask=mask,
                global_emb=global_emb,
                position=position,
                pid=pid,
                sid=sid,
                dataset=dataset or kwargs.get("dataset"),
                label=label,
            )
        else:
            spatial_feature = self._zero_feature(img)
        logits = self.head(diffusion_feature, foundation_feature, spatial_feature)
        out = {"logits": logits}
        if label is not None:
            out["loss"] = F.mse_loss(logits, label)
        return out

    def _encode_diffusion(self, img):
        if not self.use_diffusion_branch:
            return self._zero_feature(img)
        if img.dim() == 5:
            img = img.squeeze(0)
        if img.shape[-1] == 3:
            img = img.permute(0, 3, 1, 2)
        # Keep the large validation image tensor unmodified and normalize only
        # one spot chunk at a time before calling the frozen diffusion encoder.
        chunk_size = self.diffusion_chunk_size if self.diffusion_chunk_size > 0 else img.size(0)
        diffusion = self.diffusion.to(img.device)
        noise_labels = self.diffusion_noise_labels
        chunks = []
        for img_part in img.split(chunk_size, dim=0):
            x = img_part.float()
            if x.min() < 0 and x.max() <= 5:
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                x = (x * std + mean).clamp(0, 1)
                x = x * 2 - 1
            elif x.min() >= 0 and x.max() <= 1.5:
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                x = (x * std + mean).clamp(0, 1)
                x = x * 2 - 1
            elif x.max() > 2:
                x = x / 127.5 - 1
            chunks.append(self._encode_diffusion_chunk(diffusion, x, noise_labels))
        return torch.cat(chunks, dim=0)

    def _zero_feature(self, img):
        batch_size = img.shape[0]
        return torch.zeros(
            (batch_size, self.fusion_dim),
            device=img.device,
            dtype=torch.float32,
        )

    def _encode_diffusion_chunk(self, diffusion, x, noise_labels):
        if len(noise_labels) == 1:
            with torch.no_grad():
                features = extract_diffusion_features(diffusion, x, noise_label=noise_labels[0])
            return self.diffusion_pool(features)

        tokens = []
        for noise_label in noise_labels:
            with torch.no_grad():
                features = extract_diffusion_features(diffusion, x, noise_label=noise_label)
            tokens.append(self.diffusion_pool(features))
        tokens = torch.stack(tokens, dim=1)
        return self.diffusion_timestep_fusion(tokens)
