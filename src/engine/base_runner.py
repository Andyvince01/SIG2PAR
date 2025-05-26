import torch
from abc import ABC, abstractmethod
from src.utils.metrics import MultiTaskMetrics

class BaseRunner(ABC):
    """ Base class for all runners. """

    def __init__(self, 
        model: torch.nn.Module, 
        device: torch.device, 
        tasks: list, 
        losses: dict,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        csv_dir : str = "runs/"
    ) -> None:
        """ Initialize the BaseRunner. It sets the model, device, tasks, losses, optimizer, scheduler, and number of epochs.
        
        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be trained.
        device : torch.device   
            The device on which the training will be performed
        tasks : list
            A list of task names, each corresponding to a specific output head of the model.
        losses : dict
            A dictionary mapping task names to loss functions for each task.
        optimizer : torch.optim.Optimizer | None
            The optimizer used for training the model.
        scheduler : torch.optim.lr_scheduler._LRScheduler | None
            The learning rate scheduler for training.
        """
        self.model = model
        self.device = device
        self.tasks = tasks.keys()
        self.losses = losses
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler(enabled=True)
        self.patience = 0
        self.best_val_loss = float('inf')
        self.metrics = MultiTaskMetrics(device=device, csv_dir=csv_dir, tasks=tasks)

    @abstractmethod
    def run_epoch(self, epoch_idx: int, dataloader: torch.utils.data.DataLoader) -> dict[str, any]:
        """ This method should be implemented in subclasses to define the training and validation loop for each epoch. """
        pass