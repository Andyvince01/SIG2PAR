# >>> src/utils/__init__.py
# Original author: Andrea Vincenzo Ricciardi

import datetime
import numpy as np
import random
import torch
from contextlib import contextmanager, nullcontext
from pathlib import Path
from src.utils.gpu_monitoring import GPUStats
from src.utils.metrics import Metrics, MultiTaskMetrics
from src.utils.timer import Timer
from src.utils.logger import Logger

#--- Constants ---#
ROOT = Path(__file__).parent.parent.parent.resolve()

LOGGER = Logger(log_name='base', log_level='ALL', on_file=ROOT / 'logs', on_screen=True)
CKPT_DIR = ROOT / 'runs' / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

color_to_int_map = {
    "N/A": -1,
    "black": 1, 
    "blue": 2, 
    "brown": 3,
    "gray": 4, 
    "green": 5, 
    "orange": 6, 
    "pink": 7, 
    "purple": 8, 
    "red": 9, 
    "white": 10, 
    "yellow": 11,
}

gender_to_int_map = {
    'N/A': -1,
    'male': 0,
    'female': 1
}

bag_to_int_map = {
    'N/A': -1,
    'no': 0,
    'yes': 1
}

hat_to_int_map = {
    'N/A': -1,
    'no': 0,
    'yes': 1
}

int_to_color_map = {value: key for key, value in color_to_int_map.items()}
int_to_gender_map = {value: key for key, value in gender_to_int_map.items()}
int_to_bag_map = {value: key for key, value in bag_to_int_map.items()}
int_to_hat_map = {value: key for key, value in hat_to_int_map.items()}

# Set a torch seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

#--- Utility functions ---#
@contextmanager
def context_if(condition: bool, context_factory: callable) -> contextmanager:
    """Return a context manager based on a condition.

    Parameters
    ----------
    condition : bool
        Condition to check.
    context_factory : callable
        Context manager factory to use if the condition is True.

    Yields
    ------
    contextmanager
        Context manager based on the condition.
    """
    #--- If condition is True, use the context factory ---#
    if condition:
        with context_factory() as ctx:
            yield ctx
    #--- If condition is False, use a null context (no-op) ---#
    else:
        with nullcontext() as ctx:
            yield ctx
            
            
#--- __all__ ---#
__all__ = ["GPUStats", "context_if", "ROOT", "LOGGER", "Timer", "Metrics", "MultiTaskMetrics", "CKPT_DIR"]