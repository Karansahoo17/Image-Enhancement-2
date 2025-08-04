"""BasicSR utilities package"""

from .registry import ARCH_REGISTRY

def get_root_logger(logger_name='basicsr', log_level='INFO'):
    """Get root logger for BasicSR"""
    import logging
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, log_level.upper()))
    return logger

__all__ = ['ARCH_REGISTRY', 'get_root_logger']
