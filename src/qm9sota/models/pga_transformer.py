from __future__ import annotations

import torch
import torch.nn as nn

from qm9sota.geometry.info_volume import local_edge_volume_features
from qm9sota.geometry.multivector import Multivector
from qm9sota.models.multivector.attention import MultivectorAttention, MVAttentionConfig
from qm9sota.models.multivector.edge_attention import (
    MultivectorEdgeAttention,
    MVEdgeAttentionConfig,
)
from qm9sota.models.tiny_radial_mpnn import mean_pool


class PGAMultivectorTransformer(nn.Module):
    """
    PGA/multivector transformer prototype.

    Modes:
      dense: dense scalar multivector attention from B0
      edge:  edge-restricted scalar multivector attention from M1

    This is still a scalar-first multivector scaffold. Vector/bivector value
    transport comes in later stages.
    """

    def __init__(
        self,
        node_in_dim: int = 11,
        hidden_dim: int = 128,
        out_dim: int = 19,
        num_layers: int = 4,
        dropout: float = 0.0,
        attention_mode: str = "edge",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.attention_mode = attention_mode

        self.node_in = nn.Linear(node_in_dim, hidden_dim)

        if attention_mode == "dense":
            self.layers = nn.ModuleList(
                [
                    MultivectorAttention(
                        MVAttentionConfig(
                            scalar_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            dropout=dropout,
                        )
                    )
                    for _ in range(num_layers)
                ]
            )
        elif attention_mode == "edge":
            self.layers = nn.ModuleList(
                [
                    MultivectorEdgeAttention(
                        MVEdgeAttentionConfig(
                            scalar_dim=hidden_dim,
                            edge_dim=3,
                            hidden_dim=hidden_dim,
                            dropout=dropout,
                        )
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode_nodes(self, data):
        s = self.node_in(data.x.float())
        mv = Multivector(s=s)

        if self.attention_mode == "dense":
            for layer in self.layers:
                mv = layer(mv)
            return mv.s

        if self.attention_mode == "edge":
            edge_features = local_edge_volume_features(
                data.pos.float(),
                data.edge_index,
            )
            for layer in self.layers:
                mv = layer(
                    mv,
                    edge_index=data.edge_index,
                    edge_features=edge_features,
                )
            return mv.s

        raise ValueError(f"Unknown attention_mode: {self.attention_mode}")

    def forward(self, data, return_embeddings: bool = False):
        h = self.encode_nodes(data)
        graph_h = mean_pool(h, data.batch)
        pred_norm = self.head(graph_h)

        if return_embeddings:
            return {
                "pred": pred_norm,
                "node_embeddings": h,
                "graph_embedding": graph_h,
            }

        return pred_norm


def build_pga_transformer(cfg: dict) -> PGAMultivectorTransformer:
    model_cfg = cfg.get("model", {})
    return PGAMultivectorTransformer(
        node_in_dim=int(model_cfg.get("node_in_dim", 11)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        out_dim=int(model_cfg.get("out_dim", 19)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        attention_mode=str(model_cfg.get("attention_mode", "edge")),
    )
