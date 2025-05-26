import torch

from src.engine.base_runner import BaseRunner
from src.utils import LOGGER, TQDM

class Trainer(BaseRunner):
    """ A class to train the model. It inherits from the BaseRunner class. """
        
    def __init__(self, 
        model: torch.nn.Module, 
        device: torch.device, 
        tasks: list, 
        losses: dict, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        csv_dir : str = "runs/train/"
    ) -> None:
        """Initializes the TrainNet class.
        
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
        optimizer : torch.optim.Optimizer
            The optimizer used for training the model.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler for training.
        """
        super().__init__(model, device, tasks, losses, optimizer, scheduler, csv_dir)
                
    def run_epoch(self, epoch_idx: int, dataloader: torch.utils.data.DataLoader) -> dict[str, any]:
        """ Train the model for one epoch.
        
        Parameters
        ----------
        epoch_idx : int
            The current epoch index
        dataloader : torch.utils.data.DataLoader
            DataLoader for the training or validation dataset.
        
        Returns
        -------
        dict[str, any]
            A dictionary containing the average loss and average metrics for the current epoch.
        """
        #--- Set the model to TRAINING mode ---#
        self.model.train()

        #--- Initialize the progress bar ---#
        progress_bar = TQDM(dataloader, desc=f"🏋️  Training Epoch no. {epoch_idx}")

        for batch_idx, samples in enumerate(progress_bar):
            #--- Get the inputs and labels ---#
            inputs = samples['inputs'].to(self.device)
            labels = {task: samples['attributes'][task].to(self.device) for task in self.tasks}

            #--- Reset the gradients for the optimizer ---#
            self.optimizer.zero_grad()

            #--- Forward pass with autocast for mixed precision ---#
            with torch.autocast(device_type="cuda"):
                outputs = self.model(**inputs)

                #--- Compute the task losses ---#
                task_losses = {
                    task: self.losses[task](outputs[task], labels[task])
                    for task in self.tasks
                }
                
                # Compute the batch loss
                batch_loss = sum(task_losses.values())
                                
                #--- Update the task metrics ---#
                self.metrics.process_batch(outputs, labels, task_losses)
                
            #--- Backward & Optimizer Step ---#
            if torch.isfinite(batch_loss):
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                LOGGER.error(f"❌ Non-finite loss encountered: {batch_loss}")
                raise ValueError(f"❌ Non-finite loss encountered: {batch_loss}")
            
            #--- Scheduler Step ---#
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            #--- Update the progress bar with the current loss and accuracy ---#
            progress_bar.set_postfix({
                "📉 Loss": f"{batch_loss:.4f}",
                "📈 Accuracy": f"{self.metrics.compute_average_accuracy():.2%}"
            })

        #--- Return the average loss and metrics for the epoch ---#
        return self.metrics.compute_metrics(epoch_idx=epoch_idx)