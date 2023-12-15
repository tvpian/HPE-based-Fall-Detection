import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool


class PoseGCN(torch.nn.Module):
    def __init__(
        self, num_features: int, hidden_dim1: int, hidden_dim2: int, output_dim: int
    ) -> None:
        """
        Parameters
        ----------
        num_features : int
            Number of features in the input sequence
        hidden_dim1 : int
            Dimension of the first hidden layer of the GCN
        hidden_dim2 : int
            Dimension of the second hidden layer of the GCN
        output_dim : int
            Dimension of the output layer of the GCN
        """
        super(PoseGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, output_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Parameters
        ----------
        data : Data
            Pose Graph

        Returns
        -------
        torch.Tensor
            Output of the GCN of shape (batch_size, output_dim)
        """
        x, edge_index, batch = (
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.batch.to(self.device),
        )

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)
        return x
