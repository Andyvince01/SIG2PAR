import torch

from src.engine.base_runner import BaseRunner
from src.utils import LOGGER, TQDM

class Validator(BaseRunner):
    """ A class for validating a neural network model for multiple tasks. It inherits from the BaseRunner class. """
    
    def __init__(self, 
        model: torch.nn.Module, 
        device: torch.device, 
        tasks: list, 
        losses: dict,
        csv_dir : str = "runs/valid/"
    ) -> None:
        """Initializes the Validator class.
        
        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be validated.
        device : torch.device   
            The device on which the validation will be performed.
        tasks : list
            A list of task names, each corresponding to a specific output head of the model.
        losses : dict
            A dictionary mapping task names to loss functions for each task.
        """
        super().__init__(model, device, tasks, losses, csv_dir=csv_dir)
        
    @torch.inference_mode()
    def run_epoch(self, epoch_idx: int, dataloader: torch.utils.data.DataLoader) -> dict[str, any]:
        """Evaluate the model for one validation epoch.
        
        Parameters
        ----------
        epoch_idx : int
            The current epoch index.
        dataloader : torch.utils.data.DataLoader
            DataLoader for the validation set.
        
        Returns
        -------
        dict[str, any]
            A dictionary containing the average loss and average metrics for the current epoch.
        """
        #--- Set the model to EVALUATION mode ---#
        self.model.eval()

        #--- Initialize the progress bar ---#
        progress_bar = TQDM(dataloader, desc=f"🚀 Validating Epoch no. {epoch_idx}")

        for batch_idx, samples in enumerate(progress_bar):
            #--- Get the inputs and labels ---#
            inputs = samples['inputs'].to(self.device)
            labels = {task: samples['attributes'][task].to(self.device) for task in self.tasks}

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
    
            #--- Update the progress bar with the current loss and accuracy ---#
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    "📉 Loss": f"{batch_loss:.4f}",
                    "📈 Accuracy": f"{self.metrics.compute_average_accuracy():.2%}"
                })

        #--- Return the average loss and metrics for the epoch ---#
        return self.metrics.compute_metrics(epoch_idx=epoch_idx)