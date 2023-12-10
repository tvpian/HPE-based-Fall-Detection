import json
import os

from typing import Dict


class ModelConfig:
    """
    Class to parse the model configuration file

    Parameters
    ----------
    config_file : str
        Path to the model configuration file
    """

    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = self._parse_config()

    def _parse_config(self) -> Dict:
        """
        Parse the model configuration file

        Returns
        -------
        Dict
            Dictionary containing the model configuration
        """
        with open(self.config_file, "r") as f:
            config = json.load(f)
        return config

    def get_config(self) -> Dict:
        """
        Returns the model configuration

        Returns
        -------
        Dict
            Dictionary containing the model configuration
        """
        return self.config
