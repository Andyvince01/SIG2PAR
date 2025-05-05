import json
import torch

TASKS = {'upper_color': 11, "lower_color": 11, "gender": 1, "bag": 1, "hat": 1}

class Metrics:
    """ Class to compute and store the metrics for a given task. """
    def __init__(self, task : str, nc : int, device : torch.device):
        """ Initialize the metrics class.
        
        Parameters
        ----------
        task : str
            The name of the task.
        nc : int    
            The number of classes for the task.
        device : torch.device
            The device to use for the computations.
        """
        self.task = task
        self.nc = nc        
        self.device = device

        self.loss = 0.0
        self.total_samples = 0
        self.matrix = torch.zeros((nc, nc), dtype=torch.int64, device=self.device) if nc > 1 else torch.zeros((2, 2), dtype=torch.int64, device=self.device)

    @torch.inference_mode()
    def process_batch(self, outputs: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> None:
        """ The function to process a batch of data. It updates the loss and the confusion matrix.
        
        Parameters
        ----------
        outputs : torch.Tensor
            The model outputs.
        labels : torch.Tensor
            The ground truth labels.
        loss : torch.Tensor
            The loss value.
        """
        #--- Update the loss ---#
        self.loss += loss.item() * labels.size(0)       # Multiply by batch size
        self.total_samples += labels.size(0)            # Increment the total samples

        #--- Update the confusion matrix ---#
        if self.nc > 1:
            preds = torch.argmax(outputs, dim=1).to(dtype=torch.int64)
        else:
            preds = (torch.sigmoid(outputs) > 0.5).squeeze(1).to(dtype=torch.int64)

        # Convert labels to int64 if they are not already
        labels = labels.to(dtype=torch.int64)

        # Increment the confusion matrix
        for t, p in zip(labels, preds):
            self.matrix[t, p] += 1
            
    @torch.inference_mode()
    def compute_metrics(self, save_csv : str | None = None) -> dict:
        """ Compute the metrics from the confusion matrix.
        
        Parameters
        ----------
        save_csv : str | None
            Whether to save the metrics to a CSV file or not.
        
        Returns
        -------
        dict
            A dictionary containing all the metrics computed for the current epoch.
        """
        #--- Check if save_csv is a boolean ---#
        if isinstance(save_csv, bool) and save_csv:
            raise ValueError("save_csv must be a string or None, not a boolean.")
        
        #--- Compute TP, TN, FP, FN ---#
        TP = self.matrix.diag().float()
        FP = self.matrix.sum(0).float() - TP           # Column sum - TP
        FN = self.matrix.sum(1).float() - TP           # Row sum - TP
        TN = self.matrix.sum().float() - (TP + FP + FN)
        
        #--- Compute metrics ---#
        accuracy = TP.sum() / self.matrix.sum()
        precision = torch.where((TP + FP) > 0, TP / (TP + FP), torch.zeros_like(TP, device=self.matrix.device))
        recall = torch.where((TP + FN) > 0, TP / (TP + FN), torch.zeros_like(TP, device=self.matrix.device))
        f1 = torch.where((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), torch.zeros_like(precision, device=self.matrix.device))
        
        # Normalize the epoch loss
        self.loss /= self.total_samples

        #--- Save the metrics to a CSV file if required ---#
        metrics = {
            'loss': self.loss,
            'accuracy': accuracy.mean().item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item(),
            'TP': TP.sum().item(),
            'TN': TN.sum().item(),
            'FP': FP.sum().item(),
            'FN': FN.sum().item()
        }
        
        if save_csv:
            with open(f"{save_csv}/metrics_task_{self.task}.json", 'a') as f:
                # Save the metrics to a CSV file
                json.dump(metrics, f, indent=4)
        
        #-- Return the metrics ---#
        return metrics
    
    def reset(self) -> None:
        """ Reset the metrics for the next epoch.
        """
        self.loss = 0.0
        self.total_samples = 0
        self.matrix.zero_(0)
        
class MultiTaskMetrics():
    """ Class to compute and store the metrics for multiple tasks. """
    def __init__(self, device : torch.device):
        """ Initialize the metrics class.
        
        Parameters
        ----------
        device : torch.device
            The device to use for the computations.
        """
        #--- Set the attributes ---#
        self.device = device

        #--- Initialize the metrics for each task ---#
        self.metrics = {
            task: Metrics(task=task, nc=nc, device=device) 
            for task, nc in TASKS.items()
        }
        
    @torch.inference_mode()
    def process_batch(self, outputs: dict, labels: dict, loss: dict) -> None:
        """ The function to process a batch of data. It updates the loss and the confusion matrix.
        
        Parameters
        ----------
        outputs : dict
            The model outputs.
        labels : dict
            The ground truth labels.
        loss : dict
            The loss value.
        """
        #--- Update the metrics for each task ---#
        for task in TASKS.keys():
            self.metrics[task].process_batch(outputs[task], labels[task], loss[task])
            
    @torch.inference_mode()
    def compute_metrics(self, save_csv : bool) -> dict:
        """ Compute the metrics from the confusion matrix.
        
        Parameters
        ----------
        save_csv : bool
            Whether to save the metrics to a CSV file or not.
        
        Returns
        -------
        dict
            A dictionary containing all the metrics computed for the current epoch.
        """        
        #--- Compute the metrics for each task ---#
        metrics = {
            task: self.metrics[task].compute_metrics(save_csv=save_csv)
            for task in TASKS.keys()
        }
        
        #--- Compute the overall metrics ---#
        average_accuracy = sum([metrics[task]['accuracy'] for task in TASKS.keys()]) / len(TASKS)
        average_precision = sum([metrics[task]['precision'] for task in TASKS.keys()]) / len(TASKS)
        average_recall = sum([metrics[task]['recall'] for task in TASKS.keys()]) / len(TASKS)
        average_f1 = sum([metrics[task]['f1'] for task in TASKS.keys()]) / len(TASKS)
        average_loss = sum([metrics[task]['loss'] for task in TASKS.keys()]) / len(TASKS)
        
        #--- Add the overall metrics to the dictionary ---#
        metrics['average'] = {
            'loss': average_loss,
            'accuracy': average_accuracy,
            'precision': average_precision,
            'recall': average_recall,
            'f1': average_f1
        }
        
        if save_csv:
            # Save the overall metrics to a CSV file
            with open(f"{save_csv}/metrics_overall.json", 'w') as f:
                json.dump(metrics['average'], f, indent=4)
        
        return metrics
    
    @torch.inference_mode()
    def compute_average_accuracy(self) -> float:
        """ Compute the average accuracy across all tasks.
        
        Returns
        -------
        float
            The average accuracy across all tasks.
        """
        #--- Compute the average accuracy ---#
        average_accuracy = sum([
            self.metrics[task].compute_metrics()['accuracy'] for task in TASKS.keys()
        ]) / len(TASKS)
        
        return average_accuracy
        
if __name__ == "__main__":
    # --- Example usage --- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the MultiTaskMetrics with the tasks
    metrics = MultiTaskMetrics(device=device)

    # --- Simulate some data --- #
    batch_size = 8
    outputs = {
        task: torch.randn(batch_size, nc).to(device)  # Simulate model outputs
        for task, nc in TASKS.items()
    }
    labels = {
        task: torch.randint(0, nc, (batch_size,)).to(device)  # Simulate labels
        for task, nc in TASKS.items()
    }
    losses = {
        task: torch.randn(1).to(device)  # Simulate loss
        for task in TASKS
    }

    # Process the simulated batch
    metrics.process_batch(outputs, labels, losses)

    # Compute and print the aggregated metrics
    aggregated_metrics = metrics.compute_metrics(save_csv="runs/")