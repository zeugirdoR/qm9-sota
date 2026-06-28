"""PaiNN — equivariant message-passing backbone (Schütt, Unke, Gastegger 2021), pure PyTorch + PyG.

A real upgrade over TinyRadialMPNN: scalar features s_i (invariant) AND vector features v_i (equivariant,
[N,3,F]) updated by rotation-equivariant message/update blocks over the molecular bond graph, with a
learned atomwise readout. Scalar (energy-like) predictions are rotation- and translation-INVARIANT by
construction (self-test in __main__). No torch_cluster / radius_graph dependency (uses data.edge_index),
so no new aarch64 build risk. Exposes a graph embedding (pooled scalar features) for the OOD certificate.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GaussianRBF(nn.Module):
    def __init__(self, num_rbf: int = 20, cutoff: float = 5.0):
        super().__init__()
        offsets = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("offsets", offsets)
        self.coeff = -0.5 / float((offsets[1] - offsets[0]) ** 2)
        self.cutoff = cutoff

    def forward(self, d):  # d: [E]
        x = d.unsqueeze(-1) - self.offsets
        rbf = torch.exp(self.coeff * x.pow(2))
        env = 0.5 * (torch.cos(d * torch.pi / self.cutoff) + 1.0) * (d < self.cutoff).float()
        return rbf * env.unsqueeze(-1)


class PaiNNMessage(nn.Module):
    def __init__(self, F_dim: int, num_rbf: int):
        super().__init__()
        self.F = F_dim
        self.phi = nn.Sequential(nn.Linear(F_dim, F_dim), nn.SiLU(), nn.Linear(F_dim, 3 * F_dim))
        self.filt = nn.Linear(num_rbf, 3 * F_dim)

    def forward(self, s, v, edge_index, rbf, direction):  # s [N,F], v [N,3,F], direction [E,3]
        src, dst = edge_index
        x = self.phi(s)[src] * self.filt(rbf)                 # [E, 3F]
        ds, dvv, dvs = torch.split(x, self.F, dim=-1)         # each [E,F]
        dv = dvv.unsqueeze(1) * v[src] + dvs.unsqueeze(1) * direction.unsqueeze(-1)  # [E,3,F]
        s_agg = torch.zeros_like(s).index_add_(0, dst, ds)
        v_agg = torch.zeros_like(v).index_add_(0, dst, dv)
        return s + s_agg, v + v_agg


class PaiNNUpdate(nn.Module):
    def __init__(self, F_dim: int):
        super().__init__()
        self.F = F_dim
        self.U = nn.Linear(F_dim, F_dim, bias=False)          # bias-free over F preserves equivariance
        self.V = nn.Linear(F_dim, F_dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * F_dim, F_dim), nn.SiLU(), nn.Linear(F_dim, 3 * F_dim))

    def forward(self, s, v):  # s [N,F], v [N,3,F]
        Uv, Vv = self.U(v), self.V(v)                          # linear over last dim -> [N,3,F]
        a = self.mlp(torch.cat([s, torch.linalg.norm(Vv, dim=1)], dim=-1))   # invariant input -> [N,3F]
        avv, asv, ass = torch.split(a, self.F, dim=-1)
        dv = avv.unsqueeze(1) * Uv                             # [N,3,F]
        ds = ass + asv * (Uv * Vv).sum(dim=1)                  # scalar from invariant inner product
        return s + ds, v + dv


class PaiNN(nn.Module):
    def __init__(self, F_dim: int = 128, num_layers: int = 3, num_rbf: int = 20,
                 cutoff: float = 5.0, out_dim: int = 19, max_z: int = 100):
        super().__init__()
        self.F = F_dim
        self.embed = nn.Embedding(max_z, F_dim)
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.messages = nn.ModuleList([PaiNNMessage(F_dim, num_rbf) for _ in range(num_layers)])
        self.updates = nn.ModuleList([PaiNNUpdate(F_dim) for _ in range(num_layers)])
        self.readout = nn.Sequential(nn.Linear(F_dim, F_dim), nn.SiLU(), nn.Linear(F_dim, out_dim))

    def encode_nodes(self, data):
        z = data.z.long(); pos = data.pos.float()
        src, dst = data.edge_index
        rij = pos[dst] - pos[src]
        d = torch.linalg.norm(rij, dim=-1)
        direction = rij / d.clamp_min(1e-8).unsqueeze(-1)
        rbf = self.rbf(d)
        s = self.embed(z)
        v = pos.new_zeros(s.size(0), 3, self.F)
        for msg, upd in zip(self.messages, self.updates):
            s, v = msg(s, v, data.edge_index, rbf, direction)
            s, v = upd(s, v)
        return s

    @staticmethod
    def _pool(x, batch, num_graphs, mean=False):
        out = x.new_zeros(num_graphs, x.size(-1)).index_add_(0, batch, x)
        if mean:
            cnt = x.new_zeros(num_graphs, 1).index_add_(0, batch, torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype))
            out = out / cnt.clamp_min(1.0)
        return out

    def forward(self, data, return_embeddings: bool = False):
        s = self.encode_nodes(data)
        num_graphs = int(data.batch.max().item()) + 1
        pred = self._pool(self.readout(s), data.batch, num_graphs)        # sum of atomic contributions
        if return_embeddings:
            graph_emb = self._pool(s, data.batch, num_graphs, mean=True)  # mean scalar features
            return {"pred": pred, "node_embeddings": s, "graph_embedding": graph_emb}
        return pred


if __name__ == "__main__":  # rotation + translation invariance self-test (scalar predictions)
    from torch_geometric.data import Data, Batch
    torch.manual_seed(0)
    mols = []
    for _ in range(4):
        n = int(torch.randint(4, 10, (1,)))
        ei = torch.randint(0, n, (2, n * 3))
        mols.append(Data(z=torch.randint(1, 10, (n,)), pos=torch.randn(n, 3), edge_index=ei))
    batch = Batch.from_data_list(mols)
    m = PaiNN(F_dim=32, num_layers=3, out_dim=19).eval()
    with torch.no_grad():
        p1 = m(batch)
        Q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        b2 = batch.clone(); b2.pos = batch.pos @ Q.T + torch.randn(3)     # rotate + translate
        p2 = m(b2)
    err = (p1 - p2).abs().max().item()
    print(f"PaiNN rotation+translation invariance: max|Δpred|={err:.2e} ->", "PASS" if err < 1e-4 else "FAIL")
