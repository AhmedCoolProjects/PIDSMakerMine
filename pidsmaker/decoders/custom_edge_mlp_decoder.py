import torch
import torch.nn as nn

from pidsmaker.encoders.custom_mlp import CustomMLPAsbtract


class CustomEdgeMLP(CustomMLPAsbtract):
    def __init__(self, in_dim, out_dim, architecture, dropout, src_dst_projection_coef, edge_vector_input_dim=0, edge_vector_proj_dim=0):
        self.edge_vector_proj_dim = edge_vector_proj_dim
        proj_dim = in_dim * 2 * src_dst_projection_coef
        if edge_vector_proj_dim > 0:
            proj_dim += edge_vector_proj_dim
        super().__init__(proj_dim, out_dim, architecture, dropout)

        self.lin_src = nn.Linear(in_dim, in_dim * src_dst_projection_coef)
        self.lin_dst = nn.Linear(in_dim, in_dim * src_dst_projection_coef)
        if edge_vector_proj_dim > 0:
            self.lin_edge = nn.Linear(edge_vector_input_dim, edge_vector_proj_dim)

    def forward(self, h_src, h_dst, edge_vector=None):
        parts = [self.lin_src(h_src), self.lin_dst(h_dst)]
        if self.edge_vector_proj_dim > 0:
            if edge_vector is not None:
                parts.append(self.lin_edge(edge_vector))
            else:
                parts.append(torch.zeros(h_src.size(0), self.edge_vector_proj_dim, device=h_src.device))
        h = torch.cat(parts, dim=-1)
        h = self.mlp(h)
        return h
