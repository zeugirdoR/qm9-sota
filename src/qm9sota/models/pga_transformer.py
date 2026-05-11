from __future__ import annotations

import math

import torch
import torch.nn as nn

from qm9sota.geometry.info_volume import (
    local_edge_volume_features,
    radial_edge_features,
    cauchy_binet_edge_features,
)
from qm9sota.geometry.multivector import Multivector
from qm9sota.models.multivector.attention import (
    MultivectorAttention,
    MVAttentionConfig,
)
from qm9sota.models.multivector.edge_attention import (
    MultivectorEdgeAttention,
    MVEdgeAttentionConfig,
)
from qm9sota.models.tiny_radial_mpnn import mean_pool


ELECTRONIC_IDX = [0, 1, 2, 3, 4]
GEOM_THERMAL_IDX = [5, 6, 11, 16, 17, 18]
ENERGY_IDX = [7, 8, 9, 10, 12, 13, 14, 15]


def make_head(hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class FamilyHeads(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = 19):
        super().__init__()

        if out_dim != 19:
            raise ValueError("FamilyHeads currently assumes QM9 out_dim=19.")

        self.electronic_head = make_head(hidden_dim, len(ELECTRONIC_IDX))
        self.geom_thermal_head = make_head(hidden_dim, len(GEOM_THERMAL_IDX))
        self.energy_head = make_head(hidden_dim, len(ENERGY_IDX))

        self.register_buffer("electronic_idx", torch.tensor(ELECTRONIC_IDX, dtype=torch.long), persistent=False)
        self.register_buffer("geom_thermal_idx", torch.tensor(GEOM_THERMAL_IDX, dtype=torch.long), persistent=False)
        self.register_buffer("energy_idx", torch.tensor(ENERGY_IDX, dtype=torch.long), persistent=False)

    def forward(self, graph_h: torch.Tensor) -> torch.Tensor:
        out = graph_h.new_zeros(graph_h.shape[0], 19)

        out[:, self.electronic_idx] = self.electronic_head(graph_h)
        out[:, self.geom_thermal_idx] = self.geom_thermal_head(graph_h)
        out[:, self.energy_idx] = self.energy_head(graph_h)

        return out


def scheduled_lambda(
    epoch_float: float,
    *,
    warmup_epochs: float,
    ramp_epochs: float,
    lambda_end: float,
) -> float:
    if epoch_float < warmup_epochs:
        return 0.0

    if ramp_epochs <= 0:
        return lambda_end

    frac = max(0.0, min(1.0, (epoch_float - warmup_epochs) / ramp_epochs))
    return lambda_end * 0.5 * (1.0 - math.cos(math.pi * frac))


class PGAMultivectorTransformer(nn.Module):
    """
    PGA/multivector transformer prototype.

    Main validated M4 path:
      radial edge features
      scalar + vector edge attention
      invariant vector magnitude feedback
      single prediction head

    Optional M6b path:
      scheduled Cauchy-Binet local volume attention bias
      score += lambda_cb(epoch) * cb_bias(cb_features)

    Cauchy-Binet is not concatenated into the main edge message.
    """

    def __init__(
        self,
        node_in_dim: int = 11,
        hidden_dim: int = 128,
        out_dim: int = 19,
        num_layers: int = 4,
        dropout: float = 0.0,
        attention_mode: str = "edge",
        edge_feature_mode: str = "radial",
        num_rbf: int = 32,
        cutoff: float = 8.0,
        vector_channels: int = 8,
        head_mode: str = "single",
        use_cb_bias: bool = False,
        cb_lambda_end: float = 0.10,
        cb_warmup_epochs: float = 3.0,
        cb_ramp_epochs: float = 5.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.attention_mode = attention_mode
        self.edge_feature_mode = edge_feature_mode
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.vector_channels = vector_channels
        self.head_mode = head_mode

        self.use_cb_bias = use_cb_bias
        self.cb_lambda_end = cb_lambda_end
        self.cb_warmup_epochs = cb_warmup_epochs
        self.cb_ramp_epochs = cb_ramp_epochs

        self.current_epoch_float = 0.0

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
            if edge_feature_mode == "simple":
                edge_dim = 3
            elif edge_feature_mode in {"radial", "radial_cb_bias"}:
                edge_dim = 3 + num_rbf
            else:
                raise ValueError(f"Unknown edge_feature_mode: {edge_feature_mode}")

            self.layers = nn.ModuleList(
                [
                    MultivectorEdgeAttention(
                        MVEdgeAttentionConfig(
                            scalar_dim=hidden_dim,
                            edge_dim=edge_dim,
                            hidden_dim=hidden_dim,
                            vector_channels=vector_channels,
                            dropout=dropout,
                            cb_dim=32,
                            use_cb_bias=use_cb_bias,
                        )
                    )
                    for _ in range(num_layers)
                ]
            )

        else:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")

        if head_mode == "single":
            self.head = make_head(hidden_dim, out_dim)
        elif head_mode == "family":
            self.head = FamilyHeads(hidden_dim=hidden_dim, out_dim=out_dim)
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

    def set_epoch_float(self, epoch_float: float) -> None:
        self.current_epoch_float = float(epoch_float)

    def cb_lambda(self) -> float:
        if not self.use_cb_bias:
            return 0.0

        return scheduled_lambda(
            self.current_epoch_float,
            warmup_epochs=self.cb_warmup_epochs,
            ramp_epochs=self.cb_ramp_epochs,
            lambda_end=self.cb_lambda_end,
        )

    def _edge_features(self, data) -> torch.Tensor:
        pos = data.pos.float()
        edge_index = data.edge_index

        if self.edge_feature_mode == "simple":
            return local_edge_volume_features(pos, edge_index)

        if self.edge_feature_mode in {"radial", "radial_cb_bias"}:
            return radial_edge_features(
                pos,
                edge_index,
                num_basis=self.num_rbf,
                cutoff=self.cutoff,
            )

        raise ValueError(f"Unknown edge_feature_mode: {self.edge_feature_mode}")

    def _cb_features(self, data) -> torch.Tensor | None:
        if not self.use_cb_bias:
            return None

        return cauchy_binet_edge_features(
            data.pos.float(),
            data.edge_index,
        )

    def encode_nodes(self, data):
        s = self.node_in(data.x.float())
        mv = Multivector(s=s)

        if self.attention_mode == "dense":
            for layer in self.layers:
                mv = layer(mv)
            return mv.s

        if self.attention_mode == "edge":
            edge_features = self._edge_features(data)
            cb_features = self._cb_features(data)
            lambda_cb = self.cb_lambda()

            for layer in self.layers:
                mv = layer(
                    mv,
                    edge_index=data.edge_index,
                    edge_features=edge_features,
                    pos=data.pos.float(),
                    cb_features=cb_features,
                    lambda_cb=lambda_cb,
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
        edge_feature_mode=str(model_cfg.get("edge_feature_mode", "radial")),
        num_rbf=int(model_cfg.get("num_rbf", 32)),
        cutoff=float(model_cfg.get("cutoff", 8.0)),
        vector_channels=int(model_cfg.get("vector_channels", 8)),
        head_mode=str(model_cfg.get("head_mode", "single")),
        use_cb_bias=bool(model_cfg.get("use_cb_bias", False)),
        cb_lambda_end=float(model_cfg.get("cb_lambda_end", 0.10)),
        cb_warmup_epochs=float(model_cfg.get("cb_warmup_epochs", 3.0)),
        cb_ramp_epochs=float(model_cfg.get("cb_ramp_epochs", 5.0)),
    )
