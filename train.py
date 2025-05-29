from typing import Optional, Dict, Any, Union
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.data import PARDataset
from src.models import SIG2PAR
from src.engine import AsymmetricLoss, Trainer, Validator
from src.utils import *

class TrainingManager:
    """
    A class to manage the training process, including saving and loading checkpoints.
    
    Attributes:
        model: The neural network model to train
        optimizer: The optimizer for training
        scheduler: Learning rate scheduler
        best_val_loss: Best validation loss achieved
        patience: Current patience counter for early stopping
        epoch: Current epoch number
        max_patience: Maximum patience before early stopping
        device: Device where model is located
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer, 
        scheduler: Union[torch.optim.lr_scheduler.ReduceLROnPlateau, Any],
        max_patience: int = 5
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_loss = float('inf')
        self.patience = 0
        self.epoch = 1
        self.max_patience = max_patience
        
    def save_checkpoint(self, filename: Union[str, Path], epoch: Optional[int] = None) -> None:
        """
        Save the model checkpoint.

        Args:
            filename: The path to save the checkpoint
            epoch: The current epoch number for logging
        """
        epoch_idx = epoch if epoch is not None else -1
        checkpoint = {
            'epoch': epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(checkpoint, filename)
        LOGGER.debug(f"\t...{filename} model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, filename: Union[str, Path]) -> None:
        """
        Load a checkpoint from a file.
        
        Args:
            filename: The path to load the checkpoint from
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint {filename} not found.")
            
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Move optimizer state to the correct device
        optimizer_state = checkpoint['optimizer_state_dict']
        for state in optimizer_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.optimizer.load_state_dict(optimizer_state)
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.epoch = checkpoint.get('epoch', 7) + 1
        
        del checkpoint
        torch.cuda.empty_cache()
        LOGGER.info(f"\t...checkpoint loaded from {filename}.")
    
    @staticmethod
    def print_metrics(epoch: int, train_results: Dict[str, Any], val_results: Dict[str, Any]) -> None:
        """
        Display training and validation metrics in an optimized format.
        
        Args:
            epoch: Current epoch number
            train_results: Training metrics dictionary
            val_results: Validation metrics dictionary
        """
        train_avg = train_results.get('average', {})
        val_avg = val_results.get('average', {})
        
        # Header
        results = f"{'='*45}\n"
        results += f"{'Average':<12} {'Phase':<10} {'Loss':<9} {'Accuracy':<15}\n"

        # Average metrics
        results += (
            f"{'/':<12} {'Train':<10} "
            f"{train_avg.get('loss', 0):<9.4f} "
            f"{train_avg.get('accuracy', 0):<15.2%}\n"
            f"{'/':<12} {'Valid':<10} "
            f"{val_avg.get('loss', 0):<9.4f} "
            f"{val_avg.get('accuracy', 0):<15.2%}\n"
            f"{'='*45}\n"
        )
        
        # Task-specific metrics
        results += f"{'Task':<12} {'Phase':<10} {'Loss':<9} {'Accuracy':<15}\n"
        results += f"{'-'*45}\n"
        
        tasks = [task for task in train_results.keys() if task != 'average']
        
        for task in tasks:
            train_metrics = train_results.get(task, {})
            val_metrics = val_results.get(task, {})
            
            results += (
                f"{task:<12} {'Train':<10} "
                f"{train_metrics.get('loss', 0):<9.4f} "
                f"{train_metrics.get('accuracy', 0):<15.2%}\n"
                f"{'':<12} {'Valid':<10} "
                f"{val_metrics.get('loss', 0):<9.4f} "
                f"{val_metrics.get('accuracy', 0):<15.2%}\n"
                f"{'-'*45}\n"
            )
            
        LOGGER.info(f"📊 EPOCH {epoch} RESULTS:\n{results}")
        
    def train(
        self, 
        trainer: Trainer, 
        validator: Validator, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        num_epochs: int, 
        checkpoint_path: Union[str, Path] = 'checkpoints'
    ) -> None:
        """
        Train the model for a specified number of epochs.
        
        Args:
            trainer: The Trainer object for training
            validator: The Validator object for validation
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_path: Directory to save checkpoints
        """
        # Create checkpoint directory
        checkpoint_dir = Path(checkpoint_path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress bar
        progress_bar = TQDM(
            range(self.epoch, num_epochs + 1), 
            desc="⭐ Training Progress"
        )
                
        # Monitor training resources
        with Timer() as timer, GPUStats() as gpu_stats:
            LOGGER.debug(
                f"🏋️ Training started with {gpu_stats.start_reserved:.2f} GB "
                f"reserved, {gpu_stats.start_allocated:.2f} GB allocated."
            )
            
            for epoch in progress_bar:
                # Training phase
                train_results = trainer.run_epoch(
                    epoch_idx=epoch, 
                    dataloader=train_dataloader
                )

                # Validation phase
                val_results = validator.run_epoch(
                    epoch_idx=epoch, 
                    dataloader=val_dataloader
                )
                
                # Extract metrics
                val_loss = val_results.get('average', {}).get('loss', float('inf'))
                val_accuracy = val_results.get('average', {}).get('accuracy', 0.0)
                
                # Update learning rate (skip OneCycleLR)
                if (hasattr(self.scheduler, '__class__') and 
                    self.scheduler.__class__.__name__ != "OneCycleLR"):
                    self.scheduler.step(val_loss)
                
                # Display metrics
                self.print_metrics(epoch, train_results, val_results)
                
                # Save checkpoints
                last_checkpoint = checkpoint_dir / 'last_model.pth'
                self.save_checkpoint(last_checkpoint, epoch=epoch)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "🕰️ Epoch": f"{epoch}/{num_epochs}",
                    "🏅 Val Loss": f"{val_loss:.4f}",
                    "📈 Val Acc": f"{val_accuracy:.1%}",
                    "⏱️ Patience": f"{self.patience}/{self.max_patience}"
                })
                
                # Early stopping logic
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_checkpoint = checkpoint_dir / 'best_model.pth'
                    self.save_checkpoint(best_checkpoint, epoch=epoch)
                    self.patience = 0
                    LOGGER.info(f"✅ New best model saved! Validation loss: {val_loss:.4f}")
                else:
                    self.patience += 1
                    if self.patience > self.max_patience:
                        LOGGER.info(f'⏲️ Early stopping triggered after {epoch} epochs')
                        break
                
                # Reset metrics for next epoch
                trainer.metrics.reset()
                validator.metrics.reset()
                    
        LOGGER.info(
            f"🎉 Training completed!\n"
            f"► Final GPU usage: {gpu_stats.end_reserved:.1f} GB reserved, "
            f"{gpu_stats.end_allocated:.1f} GB allocated\n"
            f"► Total training time: {timer.duration:.2f} seconds\n"
            f"► Best validation loss: {self.best_val_loss:.4f}"
        )


def _create_criterions() -> Dict[str, AsymmetricLoss]:
    """Create loss functions for each task."""
    return {
        'upper_color': AsymmetricLoss(
            gamma_neg=torch.tensor([1, 2, 4, 2, 3, 5, 5, 5, 2, 2, 4]), 
            gamma_pos=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            num_classes=11
        ),
        'lower_color': AsymmetricLoss(
            gamma_neg=torch.tensor([1, 2, 4, 2, 3, 5, 5, 5, 4, 4, 4]), 
            gamma_pos=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            num_classes=11
        ),
        'gender': AsymmetricLoss(gamma_neg=0, gamma_pos=2, num_classes=1),
        'bag': AsymmetricLoss(gamma_neg=0, gamma_pos=3, num_classes=1),
        'hat': AsymmetricLoss(gamma_neg=0, gamma_pos=3, num_classes=1),
    }


def _create_optimizer_groups(model: SIG2PAR) -> list:
    """Create parameter groups for optimizer with different learning rates."""
    vision_layers = model.vision_model.encoder.layers
    num_layers = len(vision_layers)
    
    return [
        # Group 1: 4th-to-last and 5th-to-last layers
        {
            'params': [
                p for n, p in model.vision_model.named_parameters()
                if (n.startswith('encoder.layers') and 
                    int(n.split('.')[2]) in [num_layers - 5, num_layers - 4])
            ], 
            'lr': 5e-6,            # dimezzato rispetto a prima
            'weight_decay': 5e-5   # dimezzato rispetto a prima
        },
        # Group 2: 1st-to-last and 3rd-to-last layers
        {
            'params': [
                p for n, p in model.vision_model.named_parameters()
                if (n.startswith('encoder.layers') and 
                    int(n.split('.')[2]) in [num_layers - 3, num_layers - 1])
            ], 
            'lr': 1e-4,            # abbassato di 2 ordini di grandezza
            'weight_decay': 1e-4   # ridotto un po' rispetto a 5e-3
        },
        # Group 3: Post LayerNorm and Head
        {
            'params': [
                p for n, p in model.vision_model.named_parameters()
                if 'post_layernorm' in n or 'head' in n
            ], 
            'lr': 1e-5,            # dimezzato
            'weight_decay': 5e-5   # dimezzato
        },
        # Group 4: Task-specific heads
        {
            'params': model.heads.parameters(), 
            'lr': 1e-4,            # ridotto di circa 5 volte (più stabile)
            'weight_decay': 1e-4
        }
    ]



def _create_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    # Dataset paths
    data_root = Path(ROOT) / "data" / "mivia_par"
    
    train_dataset = PARDataset(
        data_folder=data_root / "training_set", 
        annotation_path=data_root / "training_set.txt", 
        augment=True
    )

    val_dataset = PARDataset(
        data_folder=data_root / "validation_set", 
        annotation_path=data_root / "validation_set.txt", 
        augment=False
    )

    # DataLoader parameters
    dataloader_params = {
        "batch_size": 84,
        "num_workers": min(8, os.cpu_count()), 
        "pin_memory": True,
        "persistent_workers": torch.cuda.is_available(),
        "prefetch_factor": 2 if torch.cuda.is_available() else None,
    }
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **dataloader_params)
    
    return train_dataloader, val_dataloader


def train(resume: Optional[str] = None) -> None:
    """
    Train the SIG2PAR model on the MIVIA-PAR dataset.
    
    Args:
        resume: Path to checkpoint file to resume from. If None, starts fresh.
        
    Raises:
        FileNotFoundError: If resume path is provided but doesn't exist
    """
    # Create data loaders
    train_dataloader, val_dataloader = _create_dataloaders()

    # Initialize model
    load_weights = resume is not None
    model = SIG2PAR(load_weights=load_weights)
    
    if not load_weights:
        model.to(DEVICE)
    
    # Create loss functions
    criterions = _create_criterions()

    # Create optimizer with parameter groups
    param_groups = _create_optimizer_groups(model)
    optimizer = optim.AdamW(param_groups)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    # Determine checkpoint directory
    # if resume:
    #     checkpoint_path = Path(resume)
    #     if not checkpoint_path.exists():
    #         raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
    #     CKPT_DIR = checkpoint_path.parent
    
    # Initialize trainer and validator
    trainer = Trainer(
        model=model,
        device=DEVICE,
        tasks=TASKS,
        losses=criterions,
        optimizer=optimizer,
        scheduler=scheduler,
        csv_dir=CKPT_DIR / "train",
        task_weights=TASK_WEIGHTS
    )
    
    validator = Validator(
        model=model,
        device=DEVICE,
        tasks=TASKS,
        losses=criterions,
        csv_dir=CKPT_DIR / "val",
        task_weights=TASK_WEIGHTS
    )

    # Initialize training manager
    training_manager = TrainingManager(
        model=model,
        device=DEVICE,
        optimizer=optimizer,
        scheduler=scheduler,
        max_patience=5
    )
    
    # Load checkpoint if resuming
    if resume and checkpoint_path.exists():
        training_manager.load_checkpoint(checkpoint_path)
    
    #--- Start the training process ---#
    training_manager.train(
        trainer=trainer,
        validator=validator,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=20,
        checkpoint_path=CKPT_DIR
    )

def test(checkpoint_path: str, test_data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Test the SIG2PAR model on the test dataset.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        test_data_path: Optional path to test data. If None, uses validation set
        
    Returns:
        Dictionary containing test results and metrics
        
    Raises:
        FileNotFoundError: If checkpoint or test data path doesn't exist
    """
    # Validate checkpoint path
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
    
    # Create test dataloader
    if test_data_path:
        test_data_root = Path(test_data_path)
        if not test_data_root.exists():
            raise FileNotFoundError(f"Test data path {test_data_path} not found.")
            
        test_dataset = PARDataset(
            data_folder=test_data_root / "test_set", 
            annotation_path=test_data_root / "test_set.txt", 
            augment=False
        )
    else:
        # Use validation set as test set if no test path provided
        data_root = Path(ROOT) / "data" / "mivia_par"
        test_dataset = PARDataset(
            data_folder=data_root / "validation_set", 
            annotation_path=data_root / "validation_set.txt", 
            augment=False
        )
    
    # Create test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=torch.cuda.is_available(),
        prefetch_factor=2 if torch.cuda.is_available() else None,
    )
    
    # Initialize model
    model = SIG2PAR(load_weights=False)
    model.to(DEVICE)
    
    # Create loss functions
    criterions = _create_criterions()
    
    # Load checkpoint
    LOGGER.info(f"🔄 Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create validator for testing
    test_validator = Validator(
        model=model,
        device=DEVICE,
        tasks=TASKS,
        losses=criterions,
        csv_dir=checkpoint_file.parent / "test"
    )
    
    # Run testing
    LOGGER.info(f"🧪 Starting model testing on {len(test_dataset)} samples...")
    
    with Timer() as timer, GPUStats() as gpu_stats:
        LOGGER.debug(
            f"🔬 Testing started with {gpu_stats.start_reserved:.2f} GB "
            f"reserved, {gpu_stats.start_allocated:.2f} GB allocated."
        )
        
        # Run test evaluation
        test_results = test_validator.run_epoch(
            epoch_idx=0,
            dataloader=test_dataloader,
        )
        
        # Extract metrics
        test_avg = test_results.get('average', {})
        test_loss = test_avg.get('loss', 0.0)
        test_accuracy = test_avg.get('accuracy', 0.0)
        
        # Display detailed test results
        _print_test_results(test_results)
        
    LOGGER.info(
        f"🎯 Testing completed!\n"
        f"► Final GPU usage: {gpu_stats.end_reserved:.1f} GB reserved, "
        f"{gpu_stats.end_allocated:.1f} GB allocated\n"
        f"► Total testing time: {timer.duration:.2f} seconds\n"
        f"► Test loss: {test_loss:.4f}\n"
        f"► Test accuracy: {test_accuracy:.2%}"
    )
    
    # Cleanup
    del checkpoint
    torch.cuda.empty_cache()
    
    return test_results

def _print_test_results(test_results: Dict[str, Any]) -> None:
    """
    Display test results in a formatted table.
    
    Args:
        test_results: Test metrics dictionary
    """
    test_avg = test_results.get('average', {})
    
    # Header
    results = f"{'='*50}\n"
    results += f"{'🧪 TEST RESULTS':<50}\n"
    results += f"{'='*50}\n"
    
    # Overall metrics
    results += (
        f"{'Overall Performance':<20} {'Loss':<12} {'Accuracy':<15}\n"
        f"{'-'*50}\n"
        f"{'Average':<20} "
        f"{test_avg.get('loss', 0):<12.4f} "
        f"{test_avg.get('accuracy', 0):<15.2%}\n"
        f"{'='*50}\n"
    )
    
    # Task-specific metrics
    results += f"{'Task Performance':<20} {'Loss':<12} {'Accuracy':<15}\n"
    results += f"{'-'*50}\n"
    
    tasks = [task for task in test_results.keys() if task != 'average']
    
    for task in tasks:
        task_metrics = test_results.get(task, {})
        results += (
            f"{task:<20} "
            f"{task_metrics.get('loss', 0):<12.4f} "
            f"{task_metrics.get('accuracy', 0):<15.2%}\n"
        )
    
    results += f"{'='*50}\n"
    
    LOGGER.info(f"\n{results}")

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    train()      # Set to a checkpoint path to resume training, or None to start fresh
    #test(checkpoint_path='runs/2025-05-24_20-18-05/best_model.pth', test_data_path=None)  # Set to a checkpoint path to test, or None to use validation set