import torch
import numpy as np
import argparse
from collections import Counter

from trainer import Trainer
from model import get_transformer_model, get_gcn_transformer_model
from utils.logger import Logger
from utils.model_config import ModelConfig
from data_mgmt.datasets.ur_dataset import URDataset
from data_mgmt.datasets.ntu_dataset import NTUDataset

from typing import Tuple

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data",
        help="Path to the dataset folder",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        help="Model to use for training, transformer or gcn_transformer",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="ur",
        help="Type of dataset to use, ntu or ur",    
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=11,
        help="Number of frames to skip",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Shuffle the dataset"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output/",
        help="Path to the output folder",
    )
    parser.add_argument(
        "--logger_config",
        type=str,
        default="./config/logger.ini",
        help="Path to the logger config file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./config/model.json",
        help="Path to the model config file",
    )
    parser.add_argument(
        "--occlude",
        action="store_true",
        help="Whether to occlude the input or not",
    )
    args = parser.parse_args()

    if args.dataset_type not in ["ntu", "ur"]:
        raise ValueError("Dataset type should be either ntu or ur")
    
    if args.model not in ["transformer", "gcn_transformer"]:
        raise ValueError("Model should be either transformer or gcn_transformer")
    
    if args.dataset_type == "ur":
        if args.skip % 2 == 0:
            raise ValueError("Skip frames should be odd")
        if args.skip > 11:
            raise ValueError("Skip frames should be less than 11")

    return args


def load_dataset(args : argparse.Namespace, logger : Logger) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    np.random.seed(42)
    if args.dataset_type == "ntu":
        dataset = NTUDataset(args.dataset, occlude=args.occlude)
    elif args.dataset_type == "ur":
        dataset = URDataset(args.dataset, skip=args.skip)

    if len(dataset) > 0:
        logger.info("Dataset loaded successfully.")
        logger.info(f"Dataset size: {len(dataset)}")
    else:
        logger.error("Dataset loading failed.")
        logger.info("Check if the dataset folder is correct.")
        exit()

    train_size = int(0.60 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    test_size = int(0.3 * len(val_dataset))
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_dataset, [len(val_dataset) - test_size, test_size], generator=generator
    )

    label_counts = Counter(dataset.labels)
    unique_labels = len(list(set(dataset.labels)))
    logger.info(f"Number of unique labels: {unique_labels}")

    for label, count in label_counts.items():
        logger.info(f"Label: {label}, Count: {count}")

    return train_dataset, val_dataset, test_dataset


def main():
    args = parse_args()
    logger = Logger(args.logger_config).get_logger()

    logger.info("\n")
    logger.info("Loading the dataset...")
    train_dataset, val_dataset, test_dataset = load_dataset(
        args, logger
    )

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Testing dataset size: {len(test_dataset)}")

    model_config = ModelConfig(args.model_config).get_config()
    if args.model == "transformer":
        model, (train_dataloader, val_dataloader, test_dataloader) = get_transformer_model(
            model_config, args, (train_dataset, val_dataset, test_dataset)
        )
    elif args.model == "gcn_transformer":
        model, (train_dataloader, val_dataloader, test_dataloader) = get_gcn_transformer_model(
            model_config, args, (train_dataset, val_dataset, test_dataset)
        )
    
    trainer = Trainer(model, lr=args.lr, logger=logger, model_type=args.model)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")

    logger.info("Training the model. Please wait...")
    trainer.train(
        train_dataloader,
        val_dataloader,
        epochs=args.epochs,
        output_path=args.output_folder,
        save_model=True,
    )

    logger.info("")
    logger.info("Testing model on the test dataset...")
    trainer.test(
        test_dataloader, output_path=args.output_folder
    )


if __name__ == "__main__":
    main()
