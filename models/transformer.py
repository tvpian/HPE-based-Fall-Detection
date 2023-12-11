import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from typing import Tuple, List


def get_positional_encoding(seq_length: int, d_model: int) -> torch.Tensor:
    """
    Generates the positional encoding for the transformer input

    Parameters
    ----------
    seq_length : int
        Number of frames in the input sequence
    d_model : int
        Dimension of the input embedding

    Returns
    -------
    torch.Tensor
        Positional encoding of the input sequence
    """
    pos = torch.arange(seq_length).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model).float()
    angle_rads = pos * angle_rates

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.empty((seq_length, d_model))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines

    return pos_encoding.unsqueeze(0)


class Transformer(nn.Module):
    """
    Transformer-based binary classifier
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_features: int,
        dropout: float = 0.1,
        dim_ff: int = 2048,
        num_classes: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        d_model : int
            Dimension of the input embedding
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer encoder layers
        num_features : int
            Number of features in the input sequence
        dropout : float, optional
            Dropout rate, by default 0.1
        dim_feedforward : int, optional
            Dimension of the feedforward network, by default 2048
        """
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_classes = num_classes

        self.pos_encoding = get_positional_encoding(
            1000, d_model
        )  ## TODO: this is a hack and needs to be fixed

        self.encoder = nn.Linear(num_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=dim_ff, dropout=dropout
            ),
            num_layers,
        )
        self.decoder = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_length, num_features)

        Returns
        -------
        torch.Tensor
            Output of the transformer-based binary classifier of shape (batch_size, num_classes)
        """
        x = x.permute(1, 0, 2)
        x = self.encoder(x) * math.sqrt(self.d_model)

        x += self.pos_encoding[:, : x.size(1), :].type_as(x)
        x = self.dropout(x)

        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.permute(1, 0, 2)

        x = self.decoder(x[:, -1, :])

        return x
