import os
import torch
from dataclasses import dataclass

@dataclass
class MetricsConfig:
    """ Configuration class for the metrics. """
    task: str
    nc: int
    device: torch.device
    csv_dir : str | None = None
    
    def __post_init__(self):
        """ Post initialization to validate attributes. """
        if self.nc <= 0:
            raise ValueError(f"Number of classes must be positive, got {self.nc}.")
        
        self.csv = f"{self.csv_dir}/metrics_task_{self.task}.csv" if self.csv_dir else None

class Metrics:
    """ Class to compute and store the metrics for a given task. """
    def __init__(self, config: MetricsConfig):
        """ Initialize the metrics class.
        
        Parameters
        ----------
        config : MetricsConfig
            The configuration object for the metrics.
        """
        self.task = config.task
        self.nc = config.nc
        self.csv = config.csv
        
        # Initialize tracking variables
        self.loss = 0.0
        self.total_samples = 0
        self.correct_predictions = 0

    @torch.inference_mode()
    def process_batch(self, outputs: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> None:
        """ Process a batch of data and update metrics.
        
        Parameters
        ----------
        outputs : torch.Tensor
            The model outputs.
        labels : torch.Tensor
            The ground truth labels.
        loss : torch.Tensor
            The loss value.
        """
        batch_size = labels.size(0)
        
        # Update loss
        self.loss += loss.item() * batch_size
        self.total_samples += batch_size

        # Update accuracy (more efficient than confusion matrix)
        if self.nc > 1:
            preds = torch.argmax(outputs, dim=1)
        else:
            preds = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()

        # Count correct predictions
        self.correct_predictions += (preds == labels).sum().item()
            
    @torch.inference_mode()
    def compute_metrics(self, epoch_idx: int | None = None) -> dict:
        """ Compute the metrics.
        
        Parameters
        ----------
        epoch_idx : int | None
            The index of the current processing epoch.
        
        Returns
        -------
        dict
            A dictionary containing the computed metrics.
        """        
        # Compute accuracy
        accuracy = self.correct_predictions / max(self.total_samples, 1)
        
        # Normalize the epoch loss
        avg_loss = self.loss / max(self.total_samples, 1)

        # Create metrics dictionary (simplified)
        metrics = {
            'epoch': epoch_idx if epoch_idx is not None else -1,
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        return metrics
    
    def _save_metrics(self, metrics: dict) -> None:
        """ Save the metrics to a CSV file.
                
        Parameters
        ----------
        metrics : dict
            The metrics to save.
        """
        # Get the keys and values from the metrics dictionary
        file_exists = os.path.exists(self.csv)
        
        with open(self.csv, "a", buffering=8192) as f:  # Use buffering
            if not file_exists:
                header = ",".join(f"{k:>15s}" for k in metrics.keys())
                f.write(header + "\n")
            
            values = ",".join(f"{v:>15.5g}" if isinstance(v, float) else f"{v:>15d}" for v in metrics.values())
            f.write(values + "\n")

    def reset(self) -> None:
        """ Reset the metrics for the next epoch. """
        self.loss = 0.0
        self.total_samples = 0
        self.correct_predictions = 0

class MultiTaskMetrics():
    """ Class to compute and store the metrics for multiple tasks. """
    def __init__(self, tasks: dict, device : torch.device, csv_dir : str = "runs/") -> None:
        """ Initialize the metrics class.
        
        Parameters
        ----------
        tasks: dict 
            A dictionary containing the tasks and their number of classes.
        device : torch.device
            The device to use for the computations.
        csv_dir : str
            The directory to save the CSV files.
        """
        self.csv_dir = csv_dir
        self.tasks = tasks
        
        if self.csv_dir:
            os.makedirs(self.csv_dir, exist_ok=True)
        
        # Initialize the metrics for each task
        self.metrics = {
            task: Metrics(
                config=MetricsConfig(
                    task=task,
                    nc=nc,
                    device=device,
                    csv_dir=csv_dir
                )
            )
            for task, nc in self.tasks.items()
        }
        
        self.epoch_idx = 0
        
    @torch.inference_mode()
    def process_batch(self, outputs: dict, labels: dict, loss: dict) -> None:
        """ Process a batch of data for all tasks.
        
        Parameters
        ----------
        outputs : dict
            The model outputs.
        labels : dict
            The ground truth labels.
        loss : dict
            The loss value.
        """
        for task in self.tasks.keys():
            if task not in outputs or task not in labels or task not in loss:
                raise ValueError(f"Task {task} not found in outputs, labels or loss.")
            self.metrics[task].process_batch(outputs[task], labels[task], loss[task])
            
    @torch.inference_mode()
    def compute_metrics(self, epoch_idx : int) -> dict:
        """ Compute the metrics for all tasks.
        
        Parameters
        ----------
        epoch_idx : int
            The current epoch to compute the metrics for.
        
        Returns
        -------
        dict
            A dictionary containing all the metrics computed for the current epoch.
        """        
        self.epoch_idx = epoch_idx
        
        # Compute the metrics for each task
        results = {
            task: self.metrics[task].compute_metrics(epoch_idx=epoch_idx)
            for task in self.tasks.keys()
        }
        
        # Compute the overall metrics (simplified)
        num_tasks = len(self.tasks)
        average_accuracy = sum(results[task]['accuracy'] for task in self.tasks.keys()) / num_tasks
        average_loss = sum(results[task]['loss'] for task in self.tasks.keys()) / num_tasks
        
        # Add the overall metrics to the dictionary
        results['average'] = {
            'loss': average_loss,
            'accuracy': average_accuracy
        }
        
        # Save the overall metrics to a CSV file
        if self.csv_dir:
            self._save_metrics(results)                
        return results
    
    def _save_metrics(self, results: dict) -> None:
        """ Save metrics with optimized I/O. 
        
        Parameters
        ----------
        results : dict
            A dictionary containing the metrics to save.
        """
        # Save overall metrics to a CSV file
        csv_path = f"{self.csv_dir}/metrics_overall.csv"
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, "a", buffering=8192) as f:
            if not file_exists:
                header = ",".join(f"{k:>15s}" for k in results['average'].keys())
                f.write(header + "\n")
            
            values = ",".join(f"{v:>15.5g}" for v in results['average'].values())
            f.write(values + "\n")
            
        # Save the metrics for each task
        for task, metric in self.metrics.items():
            metric._save_metrics({
                'epoch': self.epoch_idx,
                **results[task]
            })
    
    @torch.inference_mode()
    def compute_average_accuracy(self) -> float:
        """ Compute the average accuracy across all tasks.
        
        Returns
        -------
        float
            The average accuracy across all tasks.
        """
        average_accuracy = sum(
            self.metrics[task].compute_metrics()['accuracy'] for task in self.tasks.keys()
        ) / len(self.tasks)
        
        return average_accuracy
    
    def reset(self) -> None:
        """ Reset the metrics for the next epoch. """
        for task in self.tasks.keys():
            self.metrics[task].reset()
        
        self.epoch_idx = 0
    
if __name__ == "__main__":
    
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = {'task1': 2, 'task2': 3}
    metrics = MultiTaskMetrics(tasks=tasks, device=device, csv_dir="runs/metrics")
    
    # Simulate a batch
    outputs = {
        'task1': torch.randn(10, 2, device=device),
        'task2': torch.randn(10, 3, device=device)
    }
    labels = {
        'task1': torch.randint(0, 2, (10,), device=device),
        'task2': torch.randint(0, 3, (10,), device=device)
    }
    loss = {
        'task1': torch.tensor(0.5, device=device),
        'task2': torch.tensor(0.7, device=device)
    }
    
    metrics.process_batch(outputs, labels, loss)
    results = metrics.compute_metrics(epoch_idx=1)
    
    print(results)