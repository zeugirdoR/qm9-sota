from __future__ import annotations

import torch
import torch.nn as nn

from qm9sota.geometry.multivector import Multivector
from qm9sota.models.multivector.attention import MultivectorAttention, MVAttentionConfig
from qm9sota.models.tiny_radial_mpnn import mean_pool


class PGAMultivectorTransformer(nn.Module):
    """
    First PGA/multivector transformer prototype.

    This version starts conservatively:
      - atom features -> scalar channels
      - position unit directions are prepared as vector channels
      - attention currently updates scalar channels
      - vector/bivector value transport will be added in later stages
    """

    def __init__(
        self,
        node_in_dim: int = 11,
        hidden_dim: int = 128,
        out_dim: int = 19,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.node_in = nn.Linear(node_in_dim, hidden_dim)

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

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode_nodes(self, data):
        s = self.node_in(data.x.float())

        mv = Multivector(s=s)

        for layer in self.layers:
            mv = layer(mv)

        return mv.s

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
    )
