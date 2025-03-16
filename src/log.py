# TODO: log settings
import logging

def set_logger(log_file, log_level=logging.INFO):
    
    logger = logging.getLogger()
    logger.setLevel(log_level)

    file_handler = logging.FileHandler('logs/' + log_file, mode = 'w')
    file_handler.setLevel(log_level)
    
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    
    return logger