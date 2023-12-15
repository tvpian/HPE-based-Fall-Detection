import torch
import torch.nn as nn
from torch_geometric.data import Batch

from models.transformer import Transformer
from models.gcn import PoseGCN


class ActionRecognizer(nn.Module):
    def __init__(
        self,
        gcn_num_features: int,
        gcn_hidden_dim1: int,
        gcn_hidden_dim2: int,
        gcn_output_dim: int,
        transformer_d_model: int,
        transformer_nhead: int,
        transformer_num_layers: int,
        transformer_num_features: int,
        transformer_dropout: float = 0.1,
        transformer_dim_feedforward: int = 2048,
        transformer_num_classes: int = 2,
        dataset: str = "ntu",
    ) -> None:
        """
        Parameters
        ----------
        gcn_num_features : int
            Number of features in the input sequence
        gcn_hidden_dim1 : int
            Dimension of the first hidden layer of the GCN
        gcn_hidden_dim2 : int
            Dimension of the second hidden layer of the GCN
        gcn_output_dim : int
            Dimension of the output layer of the GCN
        transformer_d_model : int
            Dimension of the input embedding
        transformer_nhead : int
            Number of attention heads
        transformer_num_layers : int
            Number of transformer encoder layers
        transformer_num_features : int
            Number of features in the input sequence
        transformer_dropout : float, optional
            Dropout rate, by default 0.1
        transformer_dim_feedforward : int, optional
            Dimension of the feedforward network, by default 2048
        """
        super(ActionRecognizer, self).__init__()

        self.gcn = PoseGCN(
            gcn_num_features, gcn_hidden_dim1, gcn_hidden_dim2, gcn_output_dim
        )
        self.transformer = Transformer(
            transformer_d_model,
            transformer_nhead,
            transformer_num_layers,
            transformer_num_features,
            transformer_dropout,
            transformer_dim_feedforward,
            num_classes=transformer_num_classes,
        )
        self.num_classes = transformer_num_classes
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        kps : torch.Tensor
            Input sequence of keypoints

        Returns
        -------
        torch.Tensor
            Classification of the input sequence of keypoints
        """
        outputs = []

        for item in batch:
            view_embedding = self.gcn(item)

            output = self.transformer(view_embedding.unsqueeze(0).to(self.device))
            outputs.append(output)

        return torch.stack(outputs).squeeze(1)
