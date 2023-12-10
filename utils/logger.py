import logging
from logging.config import fileConfig


class Logger:
    """
    Logger class to log the training and testing process
    """

    def __init__(self, log_config_file: str) -> None:
        fileConfig(log_config_file)
        self.logger = logging.getLogger()

    def get_logger(self) -> logging.Logger:
        """
        Returns the logger object

        Returns
        -------
        logging.Logger
            Logger object
        """
        return self.logger
