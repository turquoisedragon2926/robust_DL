import logging
import os
from datetime import datetime

class Logger:
    _instance = None

    @staticmethod
    def initialize(log_dir='results/logs', log_filename='training.log'):
        if not Logger._instance:
            Logger._instance = Logger(log_dir, log_filename)

    @staticmethod
    def get_instance():
        if Logger._instance is None:
            Logger()
        return Logger._instance

    def __init__(self, log_dir='results/logs', log_filename='training.log'):
        if Logger._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Logger._instance = self
            self._setup_logger(log_dir, log_filename)

    def _setup_logger(self, log_dir, log_filename):
        self.log_dir = log_dir
        self.log_filename = log_filename

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set the logging level

        # Create handlers (console and file)
        c_handler = logging.StreamHandler()  # Console handler
        f_handler = logging.FileHandler(os.path.join(log_dir, log_filename))  # File handler
        c_handler.setLevel(logging.INFO)  # Console handler level
        f_handler.setLevel(logging.DEBUG)  # File handler level

        # Create formatters and add them to the handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def log(self, message, level=logging.INFO):
        """
        Logs a message.
        :param message: Message to log.
        :param level: Logging level.
        """
        if level == logging.DEBUG:
            self.logger.debug(message)
        elif level == logging.INFO:
            self.logger.info(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.CRITICAL:
            self.logger.critical(message)
        else:
            self.logger.info(message)
