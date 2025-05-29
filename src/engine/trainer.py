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
        task_weights: dict[str, float], 
        csv_dir: str = "runs/train/"
    ) -> None:
        super().__init__(model, device, tasks, losses, optimizer, scheduler, csv_dir)
        self.task_weights = task_weights  
                
    def run_epoch(self, epoch_idx: int, dataloader: torch.utils.data.DataLoader) -> dict[str, any]:
        self.model.train()
        progress_bar = TQDM(dataloader, desc=f"🏋️  Training Epoch no. {epoch_idx}")

        for batch_idx, samples in enumerate(progress_bar):
            inputs = samples['inputs'].to(self.device)
            labels = {task: samples['attributes'][task].to(self.device) for task in self.tasks}
            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda"):
                outputs = self.model(**inputs)
                task_losses = {
                    task: self.losses[task](outputs[task], labels[task])
                    for task in self.tasks
                }

                # 🎯 Applica i pesi
                batch_loss = sum(self.task_weights[task] * task_losses[task] for task in self.tasks)
                self.metrics.process_batch(outputs, labels, task_losses)

            if torch.isfinite(batch_loss):
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                LOGGER.error(f"❌ Non-finite loss encountered: {batch_loss}")
                raise ValueError(f"❌ Non-finite loss encountered: {batch_loss}")

            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            progress_bar.set_postfix({
                "📉 Loss": f"{batch_loss:.4f}",
                "📈 Accuracy": f"{self.metrics.compute_average_accuracy():.2%}"
            })

        return self.metrics.compute_metrics(epoch_idx=epoch_idx)