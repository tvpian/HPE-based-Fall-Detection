import torch
from torch.utils.data import Dataset 
from torch_geometric.data import Data
import numpy as np
import os

from typing import Tuple

def get_label(file_name: str) -> int:
    if "adl" in file_name:
        return 1
    return 0


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
    skip_frame_num = file_name.split("/")[-1].split("-")[-2] == str(skip)

    return npy_file and skip_frame_num

def get_edge_index():
    """
    Returns the edge index of the pose graph
    
    Returns
    -------
    torch.Tensor
        Edge index of the pose graph
    """
    POSE_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),  # Head to left shoulder
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),  # Head to right shoulder
        (9, 10),
        (11, 12),  # Left and right shoulder
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),  # Left arm
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),  # Right arm
        (11, 23),
        (12, 24),
        (23, 24),  # Torso
        (23, 25),
        (25, 27),
        (27, 29),
        (29, 31),  # Left leg
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32),  # Right leg
    ]
    edge_index = torch.tensor(POSE_CONNECTIONS, dtype=torch.long).t().contiguous()

    return edge_index

class URDataset(Dataset):
    """
    Dataset class for the keypoint dataset
    """

    def __init__(self, dataset_folder: str, skip: int = 11) -> None:
        self.dataset_folder = dataset_folder
        self.edge_index = get_edge_index()

        self.keypoints = []
        self.poses = []
        self.labels = []

        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if is_valid_file(file, skip):
                    file_path = os.path.join(root, file)

                    kps = np.load(file_path)
                    kps = kps[:, :, :3]
                    pose_graphs = self._create_pose_graph(torch.tensor(kps))
                    kps = self._get_flattened_keypoints(torch.tensor(kps))

                    self.poses.append(pose_graphs)
                    self.keypoints.append(kps)
                    self.labels.append(get_label(file_path))

    def _create_pose_graph(self, keypoints: torch.Tensor) -> Data:
        """
        Creates a Pose Graph from the given keypoints and edge index

        Parameters
        ----------
        keypoints : torch.Tensor
            Keypoints of the pose
        edge_index : torch.Tensor
            Edge index of the pose

        Returns
        -------
        Data
            Pose Graph
        """
        pose_graphs = []
        for t in range(keypoints.shape[0]):
            pose_graph = Data(
                x=torch.tensor(keypoints[t, :, :], dtype=torch.float),
                edge_index=self.edge_index,
            )
            pose_graphs.append(pose_graph)

        return pose_graphs

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
        return len(self.keypoints)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns the sample at the given index

        Returns
        -------
        dict : {kps, label, file_name}
            A dictionary containing the keypoint array, label and file name
        """
        keypoints = self.keypoints[index]
        label = self.labels[index]
        poses = self.poses[index]

        return {"keypoints": keypoints, "label": label, "poses": poses}