import argparse
from torch.utils.data import Dataset

from models.transformer import Transformer
from data_mgmt.dataloader import DataLoader

from typing import Dict, Tuple

def get_model(
    config: Dict,
    args: argparse.Namespace,
    dataset: Tuple[Dataset, Dataset, Dataset],
) -> Tuple[
    Transformer, Tuple[DataLoader, DataLoader, DataLoader]
]:
    """
    Returns the model and the dataloader

    Parameters
    ----------
    config : Dict
        Configuration for the model
    args : argparse.Namespace
        Arguments passed to the program
    dataset : Tuple[Dataset, Dataset, Dataset]
        Dataset to use for training, validation and testing

    Returns
    -------
    Tuple[Transformer, Tuple[DataLoader, DataLoader, DataLoader]]
        Model and the dataloaders
    """
    train_dataset, val_dataset, test_dataset = dataset
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    return Transformer(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        num_features=config["num_features"],
        dropout=config["dropout"],
        dim_ff=config["dim_feedforward"],
    ), (train_loader, val_loader, test_loader)
