import logging.config
import yaml


def get_logger(name: str):
    """
    Initialize and configure a logger based on the provided configuration file.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A configured logger instance.

    The function reads a YAML configuration file located at './configs/logger.yaml',
    sets up the logging configuration, and returns a logger instance with the specified name.
    """

    with open("./configs/logger.yaml") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    logger = logging.getLogger(name)

    return logger
