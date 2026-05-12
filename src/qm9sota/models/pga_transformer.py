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


class AttentiveGraphToken(nn.Module):
    """
    Learned graph-level attention pooling.

    Produces an invariant graph context vector from node embeddings.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

        score = self.score(h).squeeze(-1)

        max_per_graph = h.new_full((num_graphs,), -float("inf"))
        max_per_graph.scatter_reduce_(0, batch, score, reduce="amax", include_self=True)

        exp_score = torch.exp(score - max_per_graph[batch])
        denom = h.new_zeros(num_graphs)
        denom.index_add_(0, batch, exp_score)

        attn = exp_score / denom[batch].clamp_min(1e-12)

        v = self.value(h)
        weighted = v * attn.unsqueeze(-1)

        out = h.new_zeros(num_graphs, h.shape[-1])
        out.index_add_(0, batch, weighted)
        return out


class GlobalFeedbackBlock(nn.Module):
    """
    E2 read-write global electronic token.

    Step 1:
      global token reads all node embeddings.

    Step 2:
      each node receives node-specific feedback from the global token.

    This is still permutation-invariant/equivariant at the graph level:
      - global token is pooled over nodes
      - feedback to node i depends on h_i and graph context
      - scalar prediction remains invariant after graph pooling
    """

    def __init__(self, hidden_dim: int, feedback_scale: float = 1.0):
        super().__init__()
        self.token = AttentiveGraphToken(hidden_dim)
        self.feedback_scale = float(feedback_scale)

        self.gate = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.update = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_h = self.token(h, batch)
        global_per_node = global_h[batch]

        joint = torch.cat([h, global_per_node], dim=-1)
        gate = self.gate(joint)
        update = self.update(joint)

        h_new = self.norm(h + self.feedback_scale * gate * update)
        return h_new, global_h


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

    M4:
      radial edge features
      scalar + vector edge attention
      invariant vector magnitude feedback
      single prediction head

    E1:
      read-only global electronic token

    E2:
      read-write global token:
        token reads nodes
        token writes node-specific feedback back into nodes
        graph embedding uses updated nodes + token

    Diagnostic options:
      family heads
      scheduled Cauchy-Binet bias
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
        use_global_token: bool = False,
        global_feedback: bool = False,
        global_feedback_layers: int = 1,
        global_feedback_scale: float = 1.0,
        use_motor: bool = False,
        motor_lambda_end: float = 0.10,
        motor_warmup_epochs: float = 15.0,
        motor_ramp_epochs: float = 25.0,
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
        self.use_global_token = use_global_token
        self.global_feedback = global_feedback

        self.use_motor = use_motor
        self.motor_lambda_end = motor_lambda_end
        self.motor_warmup_epochs = motor_warmup_epochs
        self.motor_ramp_epochs = motor_ramp_epochs

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
                            use_motor=use_motor,
                        )
                    )
                    for _ in range(num_layers)
                ]
            )

        else:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")

        if global_feedback:
            self.feedback_blocks = nn.ModuleList(
                [
                    GlobalFeedbackBlock(
                        hidden_dim=hidden_dim,
                        feedback_scale=global_feedback_scale,
                    )
                    for _ in range(int(global_feedback_layers))
                ]
            )
            self.global_token = None
            graph_context_dim = 2 * hidden_dim
            self.graph_fuse = nn.Sequential(
                nn.LayerNorm(graph_context_dim),
                nn.Linear(graph_context_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        elif use_global_token:
            self.global_token = AttentiveGraphToken(hidden_dim)
            self.feedback_blocks = None
            graph_context_dim = 2 * hidden_dim
            self.graph_fuse = nn.Sequential(
                nn.LayerNorm(graph_context_dim),
                nn.Linear(graph_context_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        else:
            self.global_token = None
            self.feedback_blocks = None
            self.graph_fuse = None

        head_dim = hidden_dim

        if head_mode == "single":
            self.head = make_head(head_dim, out_dim)
        elif head_mode == "family":
            self.head = FamilyHeads(hidden_dim=head_dim, out_dim=out_dim)
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

    def motor_lambda(self) -> float:
        if not self.use_motor:
            return 0.0

        return scheduled_lambda(
            self.current_epoch_float,
            warmup_epochs=self.motor_warmup_epochs,
            ramp_epochs=self.motor_ramp_epochs,
            lambda_end=self.motor_lambda_end,
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
            lambda_motor = self.motor_lambda()

            for layer in self.layers:
                mv = layer(
                    mv,
                    edge_index=data.edge_index,
                    edge_features=edge_features,
                    pos=data.pos.float(),
                    cb_features=cb_features,
                    lambda_cb=lambda_cb,
                    lambda_motor=lambda_motor,
                )

            return mv.s

        raise ValueError(f"Unknown attention_mode: {self.attention_mode}")

    def encode_graph(self, data):
        h = self.encode_nodes(data)

        if self.global_feedback:
            last_global = None
            for block in self.feedback_blocks:
                h, last_global = block(h, data.batch)

            mean_h = mean_pool(h, data.batch)
            graph_h = self.graph_fuse(torch.cat([mean_h, last_global], dim=-1))
            return h, graph_h

        mean_h = mean_pool(h, data.batch)

        if self.use_global_token:
            global_h = self.global_token(h, data.batch)
            graph_h = self.graph_fuse(torch.cat([mean_h, global_h], dim=-1))
        else:
            graph_h = mean_h

        return h, graph_h

    def forward(self, data, return_embeddings: bool = False):
        h, graph_h = self.encode_graph(data)
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
        use_global_token=bool(model_cfg.get("use_global_token", False)),
        global_feedback=bool(model_cfg.get("global_feedback", False)),
        global_feedback_layers=int(model_cfg.get("global_feedback_layers", 1)),
        global_feedback_scale=float(model_cfg.get("global_feedback_scale", 1.0)),
        use_motor=bool(model_cfg.get("use_motor", False)),
        motor_lambda_end=float(model_cfg.get("motor_lambda_end", 0.10)),
        motor_warmup_epochs=float(model_cfg.get("motor_warmup_epochs", 15.0)),
        motor_ramp_epochs=float(model_cfg.get("motor_ramp_epochs", 25.0)),
        use_cb_bias=bool(model_cfg.get("use_cb_bias", False)),
        cb_lambda_end=float(model_cfg.get("cb_lambda_end", 0.10)),
        cb_warmup_epochs=float(model_cfg.get("cb_warmup_epochs", 3.0)),
        cb_ramp_epochs=float(model_cfg.get("cb_ramp_epochs", 5.0)),
    )
