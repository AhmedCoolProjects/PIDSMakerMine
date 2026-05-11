import torch
import torch.nn as nn

from pidsmaker.encoders.custom_mlp import CustomMLPAsbtract, build_mlp_from_string


class CustomEdgeMLP(CustomMLPAsbtract):
    def __init__(self, in_dim, out_dim, architecture, dropout, src_dst_projection_coef, edge_vector_input_dim=0, edge_vector_proj_dim=0):
        self.edge_vector_proj_dim = edge_vector_proj_dim
        node_proj_dim = in_dim * 2 * src_dst_projection_coef
        edge_proj_dim = node_proj_dim
        if edge_vector_proj_dim > 0:
            edge_proj_dim += edge_vector_proj_dim
        super().__init__(edge_proj_dim, out_dim, architecture, dropout)

        self.lin_src = nn.Linear(in_dim, in_dim * src_dst_projection_coef)
        self.lin_dst = nn.Linear(in_dim, in_dim * src_dst_projection_coef)
        if edge_vector_proj_dim > 0:
            self.lin_edge = nn.Linear(edge_vector_input_dim, edge_vector_proj_dim)
            self.mlp_node = build_mlp_from_string(architecture, node_proj_dim, out_dim, dropout)
        else:
            self.mlp_node = None

    def forward(self, h_src, h_dst, edge_vector=None):
        h_src_proj = self.lin_src(h_src)
        h_dst_proj = self.lin_dst(h_dst)
        parts = [h_src_proj, h_dst_proj]
        node_h = torch.cat([h_src_proj, h_dst_proj], dim=-1)
        if self.edge_vector_proj_dim > 0:
            if edge_vector is not None:
                parts.append(self.lin_edge(edge_vector))
            else:
                parts.append(torch.zeros(h_src.size(0), self.edge_vector_proj_dim, device=h_src.device))
        h = torch.cat(parts, dim=-1)
        h_edge = self.mlp(h)
        if self.mlp_node is not None:
            h_node = self.mlp_node(node_h)
            return h_node, h_edge
        return h_edge, h_edge
