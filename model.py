import argparse
from torch.utils.data import Dataset

from models.transformer import Transformer
from models.action_recognizer import ActionRecognizer
from data_mgmt.dataloaders.transformer import DataLoader as TransformerDataLoader
from data_mgmt.dataloaders.gcn_transformer import DataLoader as GCNTransformerDataLoader

from typing import Dict, Tuple

def get_transformer_model(
    config: Dict,
    args: argparse.Namespace,
    dataset: Tuple[Dataset, Dataset, Dataset],
) -> Tuple[
    Transformer, Tuple[TransformerDataLoader, TransformerDataLoader, TransformerDataLoader]
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
    train_loader = TransformerDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = TransformerDataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = TransformerDataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    return Transformer(
        d_model=config["transformer_d_model"],
        nhead=config["transformer_nhead"],
        num_layers=config["transformer_num_layers"],
        num_features=config["transformer_num_features"],
        dropout=config["transformer_dropout"],
        dim_ff=config["transformer_dim_feedforward"],
        num_classes=config["transformer_num_classes"],
        dataset=args.dataset_type,
    ), (train_loader, val_loader, test_loader)

def get_gcn_transformer_model(
    config: Dict,
    args: argparse.Namespace,
    dataset: Tuple[Dataset, Dataset, Dataset],
) -> Tuple[
    ActionRecognizer, Tuple[GCNTransformerDataLoader, GCNTransformerDataLoader, GCNTransformerDataLoader]
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
    Tuple[ActionRecognizer, Tuple[DataLoader, DataLoader, DataLoader]]
        Model and the dataloaders
    """
    train_dataset, val_dataset, test_dataset = dataset
    train_loader = GCNTransformerDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = GCNTransformerDataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = GCNTransformerDataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    return ActionRecognizer(
        gcn_num_features=config["gcn_num_features"],
        gcn_hidden_dim1=config["gcn_hidden_dim1"],
        gcn_hidden_dim2=config["gcn_hidden_dim2"],
        gcn_output_dim=config["gcn_output_dim"],
        transformer_d_model=config["transformer_d_model"],
        transformer_nhead=config["transformer_nhead"],
        transformer_num_layers=config["transformer_num_layers"],
        transformer_num_features=config["transformer_num_features"],
        transformer_dropout=config["transformer_dropout"],
        transformer_dim_feedforward=config["transformer_dim_feedforward"],
        transformer_num_classes=config["transformer_num_classes"],
        dataset=args.dataset_type,
    ), (train_loader, val_loader, test_loader)