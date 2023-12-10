import torch
from torch.utils.data import Dataset 
import numpy as np
import os

from typing import Tuple

def get_label(file_name: str) -> int:
    if "adl" in file_name:
        return 0
    return 1


def is_valid_file(file_name: str, skip: int = 11) -> bool:
    """
    Checks if the file is a valid file

    Parameters
    ----------
    file_name : str
        Name of the file
    skip : int, optional
        Number of frames to skip, by default 11

    Returns
    -------
    bool
        True if the file is valid, False otherwise
    """
    npy_file = file_name.endswith(".npy")
    cam0 = "cam0" in file_name
    skip_frame_num = file_name.split("/")[-1].split("-")[-2] == str(skip)

    return npy_file and cam0 and skip_frame_num


class KeypointsDataset(Dataset):
    """
    Dataset class for the keypoint dataset
    """

    def __init__(self, dataset_folder: str, skip: int = 11) -> None:
        self.dataset_folder = dataset_folder

        self.poses = []
        self.labels = []
        self.file_names = []

        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if is_valid_file(file, skip):
                    file_path = os.path.join(root, file)

                    kps = np.load(file_path)
                    kps = kps[:, :, :3]
                    kps = self._get_flattened_keypoints(torch.tensor(kps))

                    self.poses.append(kps)
                    self.labels.append(get_label(file_path))
                    self.file_names.append(file_path)

    def _get_flattened_keypoints(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Returns the flattened keypoints

        Parameters
        ----------
        keypoints : torch.Tensor
            Keypoints

        Returns
        -------
        torch.Tensor
            Flattened keypoints
        """
        return keypoints.reshape(keypoints.shape[0], -1)
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns
        -------
        int : len
            Number of samples in the dataset
        """
        return len(self.poses)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns the sample at the given index

        Returns
        -------
        dict : {kps, label, file_name}
            A dictionary containing the keypoint array, label and file name
        """
        poses = self.poses[index]
        label = self.labels[index]
        return poses, label