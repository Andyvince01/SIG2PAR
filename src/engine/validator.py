import torch

from tqdm import tqdm
from src.engine.base_runner import BaseRunner
from src.utils import CKPT_DIR, LOGGER

class Validator(BaseRunner):
    """ A class for validating a neural network model for multiple tasks. It inherits from the BaseRunner class. """
    
    def __init__(self, 
        model: torch.nn.Module, 
        device: torch.device, 
        tasks: list, 
        losses: dict,
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
        super().__init__(model, device, tasks, losses)
        
    @torch.inference_mode()
    def run_epoch(self, dataloader: torch.utils.data.DataLoader) -> dict[str, any]:
        """Evaluate the model for one validation epoch.
        
        Parameters
        ----------
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
        progress_bar = tqdm(dataloader, desc="🚀 Validating Epoch...", dynamic_ncols=True, leave=False)

        for batch_idx, samples in enumerate(progress_bar):
            #--- Get the inputs and labels ---#
            inputs, labels = samples['inputs'].to(self.device), samples['attributes'].to(self.device)

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
            progress_bar.set_postfix({
                "📉 Loss": f"{batch_loss:.4f}",
                "📈 Accuracy": f"{self.metrics.compute_average_accuracy():.2%}"
            })

        #--- Return the average loss and metrics for the epoch ---#
        return self.metrics.compute_metrics(save_csv=CKPT_DIR)['average']