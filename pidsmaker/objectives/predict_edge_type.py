import torch.nn as nn

from pidsmaker.utils.utils import compute_class_weights


class EdgeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, edge_type_dim, edge_loss_lambda=0.3):
        super(EdgeTypePrediction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.edge_type_dim = edge_type_dim
        self.edge_loss_lambda = edge_loss_lambda

    def forward(self, h_src, h_dst, edge_type, inference, edge_vector=None, **kwargs):
        logits_node, logits_edge = self.decoder(h_src=h_src, h_dst=h_dst, edge_vector=edge_vector)

        class_weights = (
            compute_class_weights(edge_type, num_classes=self.edge_type_dim)
            if self.balanced_loss
            else None
        )

        edge_type_classes = edge_type.argmax(dim=1)
        loss_node = self.loss_fn(logits_node, edge_type_classes, inference=inference, class_weights=class_weights)
        loss_edge = self.loss_fn(logits_edge, edge_type_classes, inference=inference, class_weights=class_weights)
        loss = loss_node + self.edge_loss_lambda * loss_edge
        return {"loss": loss}
