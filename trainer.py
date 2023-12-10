import os
import time
from itertools import cycle
from logging import Logger

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from typing import Tuple

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-5,
        logger: Logger = None,
    ) -> None:
        """
        Parameters
        ----------
        model : nn.Module
            Model to be trained
        lr : float, optional
            Learning rate, by default 5e-5
        logger : Logger, optional
            Logger object, by default None
        """

        self.logger = logger
        self.model = model
        self.lr = lr
        self.writer = SummaryWriter()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=1, gamma=0.75
        # )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.75, patience=6, threshold=0.001
        )

        self.logger.info(f"Selected device: {self.device}")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )

        self.train_loss = []
        self.val_loss = []
        self.best_model = None

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 20,
        output_path: str = "../output",
        save_model: bool = False,
    ) -> None:
        """
        Trains the model

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset
        epochs : int, optional
            Number of epochs to train, by default 20

        Returns
        -------
        None
        """
        torch.manual_seed(0)
        best_val_loss = float("inf")

        for epoch in range(epochs):
            epoch_loss, epoch_correct, epoch_count = self.train_one_epoch(train_loader)
            self.train_loss.append(epoch_loss)
            self.logger.info(
                f"epoch: {epoch} | epoch train loss: {epoch_loss:.4f}   | epoch train accuracy: {epoch_correct / epoch_count:.4f}   | lr: {self.optimizer.param_groups[0]['lr']:.8f}"
            )

            epoch_val_loss, epoch_val_correct, epoch_val_count = self.evaluate(
                val_loader
            )
            self.val_loss.append(epoch_val_loss)
            self.logger.info(
                f"epoch: {epoch} | epoch val   loss: {epoch_val_loss:.4f}   | epoch val accuracy: {epoch_val_correct / epoch_val_count:.4f}   | lr: {self.optimizer.param_groups[0]['lr']:.8f}"
            )
            self.scheduler.step(epoch_val_loss / len(val_loader))

            if epoch_val_loss < best_val_loss and save_model:
                best_val_loss = epoch_val_loss
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                file_name = (
                    output_path
                    + time.strftime("%Y%m%d-%H%M%S")
                    + "_"
                    + self.model.__class__.__name__
                    + "_best_model.pt"
                )
                torch.save(self.model.state_dict(), file_name)
                self.best_model = self.model

            self.writer.add_scalar(
                "Training Loss per Epoch", (epoch_loss / len(train_loader)), epoch
            )
            self.writer.add_scalar(
                "Training Accuracy per Epoch", (epoch_correct / epoch_count), epoch
            )

            self.writer.add_scalar(
                "Validation Loss per Epoch", (epoch_val_loss / len(train_loader)), epoch
            )
            self.writer.add_scalar(
                "Validation Accuracy per Epoch",
                (epoch_val_correct / epoch_val_count),
                epoch,
            )

        self.writer.close()
        self._plot_losses(output_path)

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, int, int]:
        """
        Trains the model for one epoch

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset

        Returns
        -------
        Tuple[float, int, int]
            Tuple containing the epoch loss, epoch correct and epoch count
        """
        self.model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_count = 0
        for idx, batch in enumerate(iter(train_loader)):
            predictions = self.model(batch[0])
            labels = batch[1].to(self.device)

            loss = self.criterion(predictions, labels)
            self.writer.add_scalar("Training loss per batch", loss, idx)

            correct = predictions.argmax(axis=1) == labels
            epoch_correct += correct.sum().item()
            epoch_count += correct.size(0)

            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.optimizer.step()

        return epoch_loss, epoch_correct, epoch_count

    def evaluate(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, int, int]:
        """
        Evaluates the model

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset

        Returns
        -------
        Tuple[float, int, int]
            Tuple containing the validation epoch loss, validation epoch correct
            and validation epoch count
        """
        self.model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_correct = 0
            val_epoch_count = 0

            for idx, batch in enumerate(iter(val_loader)):
                predictions = self.model(batch[0])
                labels = batch[1].to(self.device)

                val_loss = self.criterion(predictions, labels)
                self.writer.add_scalar("Validation loss per batch", val_loss, idx)

                correct = predictions.argmax(axis=1) == labels

                val_epoch_correct += correct.sum().item()
                val_epoch_count += correct.size(0)
                val_epoch_loss += val_loss.item()

        return val_epoch_loss, val_epoch_correct, val_epoch_count

    def test(
        self,
        test_loader: torch.utils.data.DataLoader,
        output_path: str,
    ) -> None:
        """
        Evaluates the model

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            DataLoader for the test dataset

        Returns
        -------
        Tuple[float, int, int]
            Tuple containing the test epoch loss, test epoch correct
            and test epoch count
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        torch.cuda.empty_cache()
        self.best_model.to(self.device)
        self.best_model.eval()
        with torch.no_grad():
            predictions = []
            labels = []
            for idx, batch in enumerate(iter(test_loader)):
                predictions.extend(self.best_model(batch[0]).argmax(axis=1).tolist())
                labels.extend(batch[1].tolist())

            self.logger.info(f"Predictions: {predictions}")
            self.logger.info(f"Labels: {labels}")

            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1_score, _ = precision_recall_fscore_support(
                labels, predictions, average="weighted"
            )

            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"Precision: {precision}")
            self.logger.info(f"Recall: {recall}")
            self.logger.info(f"F1 Score: {f1_score}")

            plt.figure(figsize=(8, 6))
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(3), colors):
                fpr, tpr, _ = roc_curve(labels, predictions, pos_label=i)
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    color=color,
                    lw=2,
                    label=f"ROC curve of class {i} (area = {roc_auc:.2f})",
                )

            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic for Multi-class")
            plt.legend(loc="lower right")
            file_name = (
                time.strftime("%Y%m%d-%H%M%S")
                + "_"
                + self.model.__class__.__name__
                + "_"
                + "roc_curve"
                + ".png"
            )
            plt.savefig(os.path.join(output_path, file_name))
            plt.show()

            cm = confusion_matrix(labels, predictions)
            ax = sns.heatmap(
                cm,
                annot=True,
                fmt="g",
                cmap="Blues",
                xticklabels=list(range(self.model.num_classes)),
                yticklabels=list(range(self.model.num_classes)),
            )
            ax.set_xlabel("Predicted labels")
            ax.set_ylabel("True labels")
            ax.set_title("Confusion Matrix")
            file_name = (
                time.strftime("%Y%m%d-%H%M%S")
                + "_"
                + self.model.__class__.__name__
                + "_"
                + "confusion_matrix"
                + ".png"
            )
            plt.savefig(os.path.join(output_path, file_name))
            plt.show()

    def _plot_losses(self, output_path: str) -> None:
        """
        Plots the training and validation losses

        Returns
        -------
        None
        """
        plt.plot(self.train_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        plt.plot(self.val_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend(["Training Loss", "Validation Loss"])
        file_name = (
            output_path
            + time.strftime("%Y%m%d-%H%M%S")
            + "_"
            + self.model.__class__.__name__
            + "_"
            + "loss"
            + ".png"
        )
        plt.savefig(file_name)
        plt.show()
