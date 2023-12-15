import torch
import numpy as np
import os
import regex as re

from torch_geometric.data import Data
from torch_geometric.data import Dataset

from typing import Dict

label_action = [
    {"id": 0, "A043": "falling"},
    {"id" : 1, "A008" : "sitting down"},
    {"id": 1, "A026": "hopping (one foot jumping)"},
]

file_name_regex = r"S(\d{3})C001P(\d{3})R(\d{3})A(\d{3})"
file_name_regex = re.compile(file_name_regex)


def get_label(file_name: str) -> int:
    """
    Returns the label of the file

    Parameters
    ----------
    file_name : str
        Name of the file

    Returns
    -------
    int
        Label of the file
    """
    label = file_name[-4:]
    for i in label_action:
        if label in i:
            return i["id"]
    return -1


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
    file_name = file_name.split("/")[-1].split(".")[0]

    if file_name_regex.match(file_name) is None or get_label(file_name) == -1:
        return False

    return npy_file


def get_edge_index():
    POSE_CONNECTIONS = [
        (3, 2),
        (20, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 24),
        (11, 23),
        (20, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 21),
        (7, 22),
        (0, 1),
        (1, 20),
        (0, 16),
        (0, 12),
        (16, 17),
        (17, 18),
        (18, 19),
        (12, 13),
        (13, 14),
        (14, 15),
    ]
    edge_index = torch.tensor(POSE_CONNECTIONS, dtype=torch.long).t().contiguous()

    return edge_index


def get_multiview_files(dataset_folder: str) -> list:
    """
    Returns a list of files that have multiple views

    Parameters
    ----------
    dataset_folder : str
        Path to the dataset folder

    Returns
    -------
    list
        List of files that have multiple views
    """
    multiview_files = []

    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if is_valid_file(file):
                file_name = file.split("/")[-1].split(".")[0]

                file_name = file_name.split("C001")
                other_views = [
                    file_name[0] + "C002" + file_name[1],
                    file_name[0] + "C003" + file_name[1],
                ]

                not_exist = False
                for view in other_views:
                    if not os.path.exists(os.path.join(root, view + ".skeleton.npy")):
                        not_exist = True
                        break
                if not_exist:
                    continue

                other_views.append(file_name[0] + "C001" + file_name[1])
                for i in range(len(other_views)):
                    other_views[i] = os.path.join(
                        root, other_views[i] + ".skeleton.npy"
                    )
                multiview_files.append(other_views)

    return multiview_files


class NTUDataset(Dataset):
    """
    Dataset class for the keypoint dataset
    """

    def __init__(
        self, dataset_folder: str, skip: int = 11, occlude: bool = False
    ) -> None:
        super().__init__(None, None, None)
        self.dataset_folder = dataset_folder
        self.edge_index = get_edge_index()

        self.poses = []
        self.labels = []
        self.keypoints = []

        self.occluded_kps = np.array([23, 24, 10, 11, 9, 8, 4, 5, 6, 7, 21, 22])

        self.multi_view_files = get_multiview_files(dataset_folder)
        for files in self.multi_view_files:
            rand_view = np.random.randint(3)

            for idx, file in enumerate(files):
                file_data = np.load(file, allow_pickle=True).item()
                frames = file_data["skel_body0"]

                if occlude and idx == rand_view:
                    frames = self._occlude_keypoints(frames)
                pose_graphs = self._create_pose_graph(frames)

                if "C001" in file:
                    kps = self._get_flattened_keypoints(torch.tensor(frames))
                    self.keypoints.append(kps)
                    self.poses.append(pose_graphs)

            file_name = files[0].split("/")[-1].split(".")[0]
            self.labels.append(get_label(file_name))

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

    def _occlude_keypoints(
        self, frames: torch.Tensor, mask_prob: float = 0.2
    ) -> torch.Tensor:
        """
        Occludes the keypoints of the pose

        Parameters
        ----------
        frames : torch.Tensor
            Keypoints of the pose
        mask_prob : float, optional
            Probability of masking the frames, by default 0.5

        Returns
        -------
        torch.Tensor
            Occluded frames
        """
        index = np.random.randint(3)
        if index == 0:
            mask_indices = np.arange(0, frames.shape[0] // 2)
        elif index == 1:
            mask_indices = np.arange(frames.shape[0] // 2, frames.shape[0])
        else:
            mask_indices = np.arange(frames.shape[0])

        masked_kps = frames[mask_indices]
        masked_kps[:, self.occluded_kps, :] = -1
        frames[mask_indices] = masked_kps

        return frames

    def len(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns
        -------
        int : len
            Number of samples in the dataset
        """
        return len(self.labels)

    def get(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns the sample at the given index

        Returns
        -------
        Dict[str, torch.Tensor] : sample
            Sample at the given index
        """
        keypoints = self.keypoints[index]
        poses = self.poses[index]
        label = self.labels[index]

        return {"keypoints": keypoints, "poses": poses, "label": label}