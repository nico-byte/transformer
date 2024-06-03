import logging.config
import yaml


def get_logger(name: str):
    with open('./configs/logger.yaml') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        
    logger = logging.getLogger(name)
    
    return logger
    