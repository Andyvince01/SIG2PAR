import torch

from src.engine.base_runner import BaseRunner
from src.utils import LOGGER, TQDM

class Validator(BaseRunner):
    def __init__(self, 
        model: torch.nn.Module, 
        device: torch.device, 
        tasks: list, 
        losses: dict,
        task_weights: dict[str, float],  # <-- nuovo parametro
        csv_dir: str = "runs/valid/"
    ) -> None:
        super().__init__(model, device, tasks, losses, csv_dir=csv_dir)
        self.task_weights = task_weights

    @torch.inference_mode()
    def run_epoch(self, epoch_idx: int, dataloader: torch.utils.data.DataLoader) -> dict[str, any]:
        self.model.eval()
        progress_bar = TQDM(dataloader, desc=f"🚀 Validating Epoch no. {epoch_idx}")

        for batch_idx, samples in enumerate(progress_bar):
            inputs = samples['inputs'].to(self.device)
            labels = {task: samples['attributes'][task].to(self.device) for task in self.tasks}

            with torch.autocast(device_type="cuda"):
                outputs = self.model(**inputs)
                task_losses = {
                    task: self.losses[task](outputs[task], labels[task])
                    for task in self.tasks
                }

                # 🎯 Applica i pesi
                batch_loss = sum(self.task_weights[task] * task_losses[task] for task in self.tasks)
                self.metrics.process_batch(outputs, labels, task_losses)

            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    "📉 Loss": f"{batch_loss:.4f}",
                    "📈 Accuracy": f"{self.metrics.compute_average_accuracy():.2%}"
                })

        return self.metrics.compute_metrics(epoch_idx=epoch_idx)
