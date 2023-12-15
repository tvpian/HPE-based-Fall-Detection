import torch
from torch_geometric.data import Batch
from torch.utils.data.dataloader import default_collate
from typing import Any, List, Mapping, Sequence, Tuple


class Collater:
    """
    Collates the batch of data

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to collate
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch) -> Any:
        """
        Collates the batch of data

        Parameters
        ----------
        batch : List[Any]
            Batch of data

        Returns
        -------
        Any
            Collated batch of data
        """
        elem = batch[0]

        if isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch: List[Any]) -> Any:
        """
        Collates the batch of data

        Parameters
        ----------
        batch : List[Any]
            Batch of data

        Returns
        -------
        Any
            Collated batch of data
        """
        batched_graphs = [item["poses"] for item in batch]
        labels = [item["label"] for item in batch]

        for i in range(len(batched_graphs)):
            batched_graphs[i] = Batch.from_data_list(batched_graphs[i])

        labels = torch.tensor(labels, dtype=torch.long)

        return batched_graphs, labels


class DataLoader(torch.utils.data.DataLoader):
    """
    Dataloader for the single view case

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to load
    batch_size : int
        Batch size, by default 1
    shuffle : bool, optional
        Whether to shuffle the dataset, by default False
    """

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False, **kwargs):
        self.collator = Collater(dataset)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collator.collate_fn,
            **kwargs,
        )
