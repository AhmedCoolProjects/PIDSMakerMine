import torch
import torch.nn as nn

from pidsmaker.encoders.custom_mlp import CustomMLPAsbtract


class CustomEdgeMLP(CustomMLPAsbtract):
    def __init__(self, in_dim, out_dim, architecture, dropout, src_dst_projection_coef, edge_vector_dim=0):
        self.edge_vector_dim = edge_vector_dim
        proj_dim = in_dim * 2 * src_dst_projection_coef
        if edge_vector_dim > 0:
            proj_dim += edge_vector_dim
        super().__init__(proj_dim, out_dim, architecture, dropout)

        self.lin_src = nn.Linear(in_dim, in_dim * src_dst_projection_coef)
        self.lin_dst = nn.Linear(in_dim, in_dim * src_dst_projection_coef)
        if edge_vector_dim > 0:
            self.lin_edge = nn.Linear(edge_vector_dim, edge_vector_dim)

    def forward(self, h_src, h_dst, edge_vector=None):
        parts = [self.lin_src(h_src), self.lin_dst(h_dst)]
        if edge_vector is not None and self.edge_vector_dim > 0:
            parts.append(self.lin_edge(edge_vector))
        h = torch.cat(parts, dim=-1)
        h = self.mlp(h)
        return h
