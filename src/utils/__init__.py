# # >>> src/utils/__init__.py
# # Original author: Andrea Vincenzo Ricciardi

import datetime
import numpy as np
import random
import time
import tqdm
import torch
import os

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

TQDM = tqdm.tqdm

#--- TASKS ---#
ATTRIBUTE_MAPS = {
    "color": {
        "N/A": -1, "black": 1, "blue": 2, "brown": 3, "gray": 4, "green": 5,
        "orange": 6, "pink": 7, "purple": 8, "red": 9, "white": 10, "yellow": 11
    },
    "gender": {"N/A": -1, "male": 0, "female": 1},
    "bag": {"N/A": -1, "no": 0, "yes": 1},
    "hat": {"N/A": -1, "no": 0, "yes": 1}
}
REVERSE_MAPS = {k: {v: key for key, v in map.items()} for k, map in ATTRIBUTE_MAPS.items()}

TASKS = {
    "upper_color": len(ATTRIBUTE_MAPS["color"]) - 1,
    "lower_color": len(ATTRIBUTE_MAPS["color"]) - 1,
    "gender": len(ATTRIBUTE_MAPS["gender"]) - 2,
    "bag": len(ATTRIBUTE_MAPS["bag"]) - 2,
    "hat": len(ATTRIBUTE_MAPS["hat"]) - 2
}

#--- Set a random seed for reproducibility ---#
def set_random_seed(seed: int = 42):
    """Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        The random seed to set. Default is 42.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seed(42)

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
            
def remove_text_model_weights(filepath : str, output_path : str):
    """Remove text model weights from a safetensors file and save the filtered tensors to a new file.
    
    Parameters
    ----------
    filepath : str
        Path to the original safetensors file.
    output_path : str
        Path to save the filtered safetensors file.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    #--- Check if the input file exists ---#
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file {filepath} does not exist.")

    #--- Check if the output directory exists, if not create it ---#
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    #--- Read the original safetensors file and filter out unwanted tensors ---#
    with safe_open(filepath, framework="pt", device="cpu") as f:
        filtered_tensors = {
            key: f.get_tensor(key) for key in f.keys() if 'text_model' not in key and "logit_" not in key
        }

    #--- Save the filtered tensors to a new safetensors file ---#
    save_file(filtered_tensors, output_path)
            
def select_device(device: str = None) -> torch.device:
    """Select the device to use for training.

    Parameters
    ----------
    device : str, optional
        Device to use. Can be 'cpu', 'cuda', 'mps', or a specific GPU index.
        If None, the function will automatically select the best available device.

    Returns
    -------
    torch.device
        The selected device.
    """
    #--- Return the device if it is already a torch.device variable ---#
    if isinstance(device, torch.device):
        return device
    
    #--- Get the value of the device argument ---#
    device = str(device).strip().lower() if device else None
    
    #--- Check if the GPU is available on the system ---#
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch, 'mps') and torch.backends.mps.is_available()
    
    #--- Auto-select the best available device if not specified ---#
    if device is None:
        if has_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            # LOGGER.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda:0")
        elif has_mps:
            # LOGGER.info("🍎 Using MPS (Apple Silicon) device")
            return torch.device("mps")
        else:
            # LOGGER.info("💻 CUDA/MPS not available, using CPU")
            return torch.device("cpu")
    
    #--- Match specific device requests ---#
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # LOGGER.info("💻 Using CPU as requested")
        return torch.device("cpu")
    
    elif device == "cuda":
        if has_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            # LOGGER.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda:0")
        else:
            LOGGER.warning("⚠️ CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
    elif device == "mps":
        if has_mps:
            # LOGGER.info("🍎 Using MPS (Apple Silicon) device")
            return torch.device("mps")
        else:
            LOGGER.warning("⚠️ MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
    elif device.isdigit():
        if has_cuda and int(device) < torch.cuda.device_count():
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            LOGGER.info(f"🚀 Using CUDA device {device}: {torch.cuda.get_device_name(int(device))}")
            return torch.device(f"cuda:{device}")
        else:
            # LOGGER.warning(f"⚠️ CUDA device {device} not available, falling back to CPU")
            return torch.device("cpu")
    else:
        LOGGER.warning(f"⚠️ Unsupported device '{device}', falling back to CPU")
        return torch.device("cpu")

DEVICE = select_device()

class TQDM2(tqdm.tqdm):
    """Classe migliorata per visualizzare una barra di progresso basata su tqdm."""
    
    def __init__(self, 
        iterable=None, 
        desc="Progress", 
        total=None, 
        width=75, 
        show_eta=True, 
        show_metrics=False, 
        **kwargs
    ):
        """Initialize the progress bar.
        Parameters
        ----------
        iterable : iterable, optional
            Iterable object to track progress. If provided, total will be inferred.
        desc : str, optional
            Description to display. Default is "Progress".
        total : int, optional
            Total number of iterations. Required if iterable is not provided.
        width : int, optional
            Width of the progress bar. Default is 75.
        show_eta : bool, optional
            Whether to show the estimated time of arrival (ETA). Default is True.
        show_metrics : bool, optional
            Whether to show additional metrics. Default is False.
        **kwargs : dict
            Additional arguments to pass to tqdm.
        """
        # Aggiungi stile alla descrizione
        styled_desc = f"\033[1m{desc}\033[0m" if desc else ""
        
        # Configura tqdm con barre e separatori personalizzati
        bar_format = ('{desc} \033[38;5;240m | \033[0m {bar} \033[38;5;240m \033[0m {percentage:3.0f}%')
        
        # Aggiungi informazioni sul tempo se richieste
        if show_eta:
            bar_format += ' \033[38;5;240m╾╼\033[0m \033[38;5;45m⏰ {elapsed}\033[0m'
            bar_format += ' \033[38;5;208m⏳ {remaining}\033[0m'
            
        # Preparazione per metriche
        if show_metrics:
            bar_format += '{postfix}'
                    
        # Passa le configurazioni a tqdm
        super().__init__(
            iterable=iterable,
            desc=styled_desc,
            total=total,
            ncols=width + len(styled_desc) + 40,  # Aggiusta la larghezza totale
            bar_format=bar_format,
            ascii=False,
            leave=True,
            **kwargs
        )
        
        # Personalizza i caratteri della barra
        self.ascii = False
        self.bar_format = bar_format
        self.show_metrics = show_metrics
        
    def update(self, metrics=None, n=1):
        """Update the progress bar with improved aesthetics.
        
        Parameters
        ----------
        metrics : dict, optional
            Dictionary of metrics to display. Default is None.
        n : int, optional
            Number of iterations to update the progress bar. Default is 1.
        """
        # Aggiorna la barra di progresso standard
        if n < 0:
            n = 1
            
        # Gestisci le metriche se fornite
        if self.show_metrics and metrics:
            # Formatta le metriche per la visualizzazione
            formatted_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, str):
                    formatted_metrics[k] = v
                elif isinstance(v, (int, float)):
                    formatted_metrics[k] = f"{v:.4f}"
                else:
                    formatted_metrics[k] = str(v)
            
            # Aggiorna il postfix con le metriche formattate
            self.set_postfix(**formatted_metrics)
            
        # Chiama l'update standard di tqdm
        super().update(n)
    
    def finish(self):
        """Complete the progress bar."""
        self.close()
        
    def _format_time(self, seconds):
        """Format seconds into a human-readable string.
        
        Parameters
        ----------
        seconds : float
            Time in seconds to format.
            
        Returns
        -------
        str
            Formatted time string.
        """
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

#--- __all__ ---#
__all__ = ["GPUStats", "context_if", "ROOT", "LOGGER", "Timer", "Metrics", "MultiTaskMetrics", "CKPT_DIR", "TASKS", "TQDM", "set_random_seed", "DEVICE", "remove_text_model_weights"]

if __name__ == "__main__":
    import random

    progress = TQDM(total=100, desc="📚 Training", show_metrics=True)

    for i in range(100):
        loss = random.uniform(0.1, 0.5)
        acc = random.uniform(0.7, 0.99)
        progress.update(metrics={"loss": loss, "accuracy": acc})
        time.sleep(0.01)

    progress.finish()
