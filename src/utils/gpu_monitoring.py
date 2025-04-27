# >>> gpu_monitoring.py
# Original author: Andrea Vincenzo Ricciardi

import gc
import torch

class GPUStats:
    """Class for monitoring GPU memory usage.
    
    Methods
    -------
    __init__()
        Initializes the GPUMonitoring instance.
    get_gpu_memory()
        Returns the current GPU memory usage in MB.
    get_gpu_memory_total()
        Returns the total GPU memory available in MB.
    get_gpu_memory_free()
        Returns the free GPU memory available in MB.
    """
    
    NUM_DEVICES = torch.cuda.device_count()
    
    def __enter__(self) -> "GPUStats":
        """Start the GPU monitoring context.

        Returns
        -------
        self : GPUMonitoring
            The GPUMonitoring instance for context usage.
        """
        #--- Free CUDA Memory ---#
        GPUStats.free()        
        
        #--- Record the starting GPU memory usage ---#
        self.start_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        self.start_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return self

    def __exit__(self, *args) -> None:
        """ Stop the GPU monitoring context and calculate memory usage. 
        
        Parameters
        ----------
        exc_type : type or None
            Exception type, if raised.
        exc_value : BaseException or None
            Exception value, if raised.
        traceback : traceback or None
            Traceback object, if an exception occurred.
        """
        #--- Record the ending GPU memory usage ---#
        self.end_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        self.end_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        #--- Record the peak GPU memory usage ---#
        self.peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self.peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    
    @staticmethod
    def free() -> None:
        """ Static Method to free GPU memory. """
        #--- Free CUDA Memory ---#
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        #--- Free Python Memory by calling garbage collector ---#
        gc.collect()