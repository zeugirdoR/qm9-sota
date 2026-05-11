from __future__ import annotations

import torch
import torch.nn as nn


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1
    out = x.new_zeros(num_graphs, x.size(-1))
    out.index_add_(0, batch, x)

    counts = x.new_zeros(num_graphs, 1)
    ones = torch.ones(x.size(0), 1, dtype=x.dtype, device=x.device)
    counts.index_add_(0, batch, ones)
    return out / counts.clamp_min(1.0)


class TinyRadialMPNN(nn.Module):
    """Small 3D graph baseline.

    This is a plumbing/debug model, not a SOTA candidate.
    It uses atom features, bond graph connectivity, and squared 3D distances.
    """

    def __init__(self, node_in_dim: int = 11, hidden_dim: int = 128, out_dim: int = 19, num_layers: int = 3):
        super().__init__()
        self.node_in = nn.Linear(node_in_dim, hidden_dim)

        self.msg_mlps = nn.ModuleList()
        self.upd_mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.msg_mlps.append(make_mlp(hidden_dim * 2 + 1, hidden_dim, hidden_dim))
            self.upd_mlps.append(make_mlp(hidden_dim * 2, hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )


    def encode_nodes(self, data):
        x = data.x.float()
        pos = data.pos.float()
        edge_index = data.edge_index

        h = self.node_in(x)

        src, dst = edge_index

        for msg_mlp, upd_mlp, norm in zip(
            self.msg_mlps,
            self.upd_mlps,
            self.norms,
        ):
            rel = pos[src] - pos[dst]
            dist2 = rel.pow(2).sum(dim=-1, keepdim=True)

            edge_feat = torch.cat([h[src], h[dst], dist2], dim=-1)
            msg = msg_mlp(edge_feat)

            agg = h.new_zeros(h.shape)
            agg.index_add_(0, dst, msg)

            dh = upd_mlp(torch.cat([h, agg], dim=-1))
            h = norm(h + dh)

        return h

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


def build_model(cfg: dict):
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "tiny_radial_mpnn")

    if name == "tiny_radial_mpnn":
        return TinyRadialMPNN(
            node_in_dim=int(model_cfg.get("node_in_dim", 11)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            out_dim=int(model_cfg.get("out_dim", 19)),
            num_layers=int(model_cfg.get("num_layers", 3)),
        )

    if name == "pga_multivector_transformer":
        from qm9sota.models.pga_transformer import build_pga_transformer
        return build_pga_transformer(cfg)

    raise ValueError(f"Unknown model: {name}")
