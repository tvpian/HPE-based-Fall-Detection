import torch
import numpy as np
import argparse
from collections import Counter

from trainer import Trainer
from model import get_model
from utils.logger import Logger
from utils.model_config import ModelConfig
from data_mgmt.dataset import KeypointsDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data",
        help="Path to the dataset folder",
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
        "--single_view",
        action="store_true",
        help="Use single view",
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
    args = parser.parse_args()

    return args


def load_dataset(dataset_folder, logger):
    np.random.seed(42)
    dataset = KeypointsDataset(dataset_folder, skip=3)

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
        args.dataset, logger
    )

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Testing dataset size: {len(test_dataset)}")

    model_config = ModelConfig(args.model_config).get_config()
    model, (train_dataloader, val_dataloader, test_dataloader) = get_model(
        model_config, args, (train_dataset, val_dataset, test_dataset)
    )

    trainer = Trainer(model, lr=args.lr, logger=logger)
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
