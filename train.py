import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import PARDataset
from src.models import SIG2PAR
from src.engine import AsymmetricLoss, Trainer, Validator
from src.utils import *

class TrainingManager:
    def __init__(self, model, optimizer, scheduler, max_patience=5):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_loss = float('inf')
        self.patience = 0
        self.max_patience = max_patience
        self.device = next(model.parameters()).device
        
    def save_checkpoint(self, filename):
        """
        Save the model checkpoint.

        Arguments:
            - filename (str): The name of the file to save the checkpoint to.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(checkpoint, filename)
        LOGGER.debug(f"\t...{filename} model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, filename):
        """
        Load a checkpoint from a file.
        
        Arguments:
            - filename (str): The name of the file to load the checkpoint from.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        #--- Move optimizer state to the correct device ---#
        optimizer_state = checkpoint['optimizer_state_dict']
        for state in optimizer_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.optimizer.load_state_dict(optimizer_state)
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        del checkpoint
        torch.cuda.empty_cache()
        LOGGER.info(f"\t...checkpoint loaded from {filename}.")
    
    @staticmethod
    def print_metrics(epoch, train_loss, val_loss, train_acc, val_acc, train_metrics, val_metrics):
        """
        Display training and validation metrics.
        """
        LOGGER.debug(
            f'\n\tEPOCH ({epoch}) --> '
            f'TRAINING LOSS: {train_loss:.4f}, '
            f'TRAINING ACCURACY: {train_acc:.4f},\n' 
            f'\t\tupper body loss: {train_metrics["upper_color"]["loss"]:.4f}, '
            f'lower body loss: {train_metrics["lower_color"]["loss"]:.4f}, '
            f'gender loss: {train_metrics["gender"]["loss"]:.4f}, '
            f'bag loss: {train_metrics["bag"]["loss"]:.4f}, ' 
            f'hat loss: {train_metrics["hat"]["loss"]:.4f}\n'
            f'\t\tupper body acc: {train_metrics["upper_color"]["accuracy"]:.4f}, '
            f'lower body acc: {train_metrics["lower_color"]["accuracy"]:.4f}, '
            f'gender acc: {train_metrics["gender"]["accuracy"]:.4f}, '
            f'bag acc: {train_metrics["bag"]["accuracy"]:.4f}, ' 
            f'hat acc: {train_metrics["hat"]["accuracy"]:.4f}'
        )
        LOGGER.debug(
            f'\tEPOCH ({epoch}) --> '
            f'VALIDATION LOSS: {val_loss:.4f}, '
            f'VALIDATION ACCURACY: {val_acc:.4f},\n' 
            f'\t\tupper body loss: {val_metrics["upper_color"]["loss"]:.4f}, '
            f'lower body loss: {val_metrics["lower_color"]["loss"]:.4f}, '
            f'gender loss: {val_metrics["gender"]["loss"]:.4f}, '
            f'bag loss: {val_metrics["bag"]["loss"]:.4f}, ' 
            f'hat loss: {val_metrics["hat"]["loss"]:.4f}\n'
            f'\t\tupper body acc: {val_metrics["upper_color"]["accuracy"]:.4f}, '
            f'lower Body acc: {val_metrics["lower_color"]["accuracy"]:.4f}, '
            f'gender acc: {val_metrics["gender"]["accuracy"]:.4f}, '
            f'bag acc: {val_metrics["bag"]["accuracy"]:.4f}, ' 
            f'hat acc: {val_metrics["hat"]["accuracy"]:.4f}'
        )
        
    def train(self, trainer, validator, train_dataloader, val_dataloader, num_epochs, checkpoint_path='checkpoints'):
        """
        Train the model for a specified number of epochs.
        
        Arguments:
            - trainer: The trainer object
            - validator: The validator object
            - train_dataloader: DataLoader for training data
            - val_dataloader: DataLoader for validation data
            - num_epochs: Number of epochs to train for
            - checkpoint_path: Directory to save checkpoints in
        """
        os.makedirs(checkpoint_path, exist_ok=True)
                
        with Timer() as timer, GPUStats() as gpu_stats:
            LOGGER.debug(f"► Training started with {gpu_stats.start_reserved} GB of memory reserved and {gpu_stats.start_allocated} GB allocated.")
        
            progress_bar = tqdm(range(num_epochs), desc="⭐ Epoch", leave=True, dynamic_ncols=True)
            
            for epoch in progress_bar:
                # Training phase
                train_loss, train_acc, train_metrics = trainer.run_epoch(train_dataloader)
                
                # Validation phase
                val_loss, val_acc, val_metrics = validator.run_epoch(val_dataloader)
                
                # Update scheduler if needed
                if getattr(self.scheduler, '__class__', None) and self.scheduler.__class__.__name__ != "OneCycleLR":
                    self.scheduler.step(val_loss)
                
                # Print metrics
                self.print_metrics(epoch, train_loss, val_loss, train_acc, val_acc, train_metrics, val_metrics)
                
                # Save checkpoints
                last_checkpoint = os.path.join(checkpoint_path, 'last_model.pth')
                self.save_checkpoint(last_checkpoint)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_checkpoint = os.path.join(checkpoint_path, 'best_model.pth')
                    self.save_checkpoint(best_checkpoint)
                    self.patience = 0
                else:
                    self.patience += 1
                    if self.patience > self.max_patience:
                        LOGGER.info('Early stopping triggered')
                        break
                    
        LOGGER.debug(f"► Training finished with {gpu_stats.end_reserved} GB of memory reserved and {gpu_stats.end_allocated} GB allocated. Total time: {timer.duration:.4f} seconds.")

def train():
    """ Main function to train the model. """
    # Initialize the Dataset
    train_dataset = PARDataset(
        data_folder=os.path.join(ROOT, "data", "old_mivia", "training_set"), 
        annotation_path=os.path.join(ROOT, "data", "old_mivia", "training_set.txt"), 
        augment=True
    )

    val_dataset = PARDataset(
        data_folder=os.path.join(ROOT, "data", "old_mivia", "validation_set"), 
        annotation_path=os.path.join(ROOT, "data", "old_mivia", "validation_set.txt"), 
        augment=False
    )

    # Initialize the DataLoader
    dataloader_params = {
        "batch_size": 64,
        "num_workers": 8, 
        "pin_memory": True
    }
    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **dataloader_params)

    # Initialize the model and move it to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIG2PAR().to(device)

    # Define loss functions for each task
    criterions = {
        'upper_color': AsymmetricLoss(
            gamma_neg=torch.tensor([1, 2, 4, 2, 3, 4, 4, 4, 2, 2, 4]), 
            gamma_pos=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            num_classes=11
        ),
        'lower_color': AsymmetricLoss(
            gamma_neg=torch.tensor([1, 2, 4, 2, 3, 5, 5, 5, 4, 4, 4]), 
            gamma_pos=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            num_classes=11
        ),
        'gender': AsymmetricLoss(gamma_neg=0, gamma_pos=2, num_classes=1),
        'bag': AsymmetricLoss(gamma_neg=0, gamma_pos=2, num_classes=1),
        'hat': AsymmetricLoss(gamma_neg=0, gamma_pos=3, num_classes=1),
    }

    # Define parameter groups for optimizer
    param_groups = [
        # Group 1: Second-to-last and third-to-last layers
        {'params': [
            p for n, p in model.vision_model.named_parameters()
            if n.startswith('encoder.layers') and int(n.split('.')[2]) in [len(model.vision_model.encoder.layers) - 3, len(model.vision_model.encoder.layers) - 2]
        ], 'lr': 5e-6, 'weight_decay': 5e-5},

        # Group 2: Final layer only
        {'params': [
            p for n, p in model.vision_model.named_parameters()
            if n.startswith('encoder.layers') and int(n.split('.')[2]) == len(model.vision_model.encoder.layers) - 1
        ], 'lr': 1e-5, 'weight_decay': 1e-4},
        
        # Group 3: Post LayerNorm and Head
        {'params': [
            p for n, p in model.vision_model.named_parameters()
            if 'post_layernorm' in n or 'head' in n
        ], 'lr': 2e-5, 'weight_decay': 1e-4},
        
        # Group 4: Task-specific heads
        {'params': model.heads.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4}
    ]

    # Set the optimizer to AdamW with weight decay    
    optimizer = optim.AdamW(param_groups)
    
    # Set the learning rate scheduler to ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    # Initialize the Trainer and Validator
    trainer = Trainer(
        model=model,
        device=device,
        tasks=criterions.keys(),
        losses=criterions,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    validator = Validator(
        model=model,
        device=device,
        tasks=criterions.keys(),
        losses=criterions,
    )

    # Create training manager and start training
    training_manager = TrainingManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        max_patience=5
    )
    
    # Start training
    training_manager.train(
        trainer=trainer,
        validator=validator,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=20,
        checkpoint_path=CKPT_DIR
    )

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    train()


# [2025-05-02 00:14:28] (base) -  SIGPAR Model Initialized:  SIGPAR Model Summary
# -------------------------
# Vision Encoder: Siglip2VisionTransformer
# Hidden Size:    1152

#  Trainable Parameters:
#   Total:   427,916,889
#   Trainable: 60,987,993
#   Frozen:    366,928,896
# -------------------------
# Train Loader size: 1278 | Train Loader dataset size: 81772
# Validation Loader size: 145 | Validation Loader dataset size: 9219
# [2025-05-02 00:14:28] (base) -  Training started with 1636.0 GB of memory reserved and 1632.380859375 GB allocated.
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [29:09<00:00,  1.37s/it, loss=0.263, accuracy=0.87]
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [29:06<00:00,  1.16s/it, loss=0.263, accuracy=0.87]
#         EPOCH (0) --> TRAINING LOSS: 0.0570, TRAINING ACCURACY: 0.8704,
#                 upper body loss: 0.0776, lower body loss: 0.0579, gender loss: 0.0376, bag loss: 0.0805, hat loss: 0.0314
#                 upper body acc: 0.7372, lower body acc: 0.8045, gender acc: 0.9570, bag acc: 0.9044, hat acc: 0.9488
#         EPOCH (0) --> VALIDATION LOSS: 0.0337, VALIDATION ACCURACY: 0.9232,
#                 upper body loss: 0.0501, lower body loss: 0.0341, gender loss: 0.0265, bag loss: 0.0487, hat loss: 0.0090
#                 upper body acc: 0.8320, lower Body acc: 0.8868, gender acc: 0.9720, bag acc: 0.9460, hat acc: 0.9791
#         ...last_model.pth model saved with validation loss: inf
#         ...best_model.pth model saved with validation loss: 0.0337
# Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [28:44<00:00,  1.35s/it, loss=0.259, accuracy=0.899]
# Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [28:43<00:00,  1.16s/it, loss=0.259, accuracy=0.899] 
#         EPOCH (1) --> TRAINING LOSS: 0.0446, TRAINING ACCURACY: 0.8987,                                                                                                                             
#                 upper body loss: 0.0576, lower body loss: 0.0436, gender loss: 0.0307, bag loss: 0.0674, hat loss: 0.0239
#                 upper body acc: 0.8009, lower body acc: 0.8472, gender acc: 0.9647, bag acc: 0.9200, hat acc: 0.9609
#         EPOCH (1) --> VALIDATION LOSS: 0.0337, VALIDATION ACCURACY: 0.9219,
#                 upper body loss: 0.0501, lower body loss: 0.0322, gender loss: 0.0254, bag loss: 0.0518, hat loss: 0.0088
#                 upper body acc: 0.8263, lower Body acc: 0.8893, gender acc: 0.9726, bag acc: 0.9396, hat acc: 0.9817
#         ...last_model.pth model saved with validation loss: 0.0337
#         ...best_model.pth model saved with validation loss: 0.0337
# Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [28:43<00:00,  1.35s/it, loss=0.203, accuracy=0.904]
# Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [28:41<00:00,  1.16s/it, loss=0.203, accuracy=0.904]
#         EPOCH (2) --> TRAINING LOSS: 0.0417, TRAINING ACCURACY: 0.9044,
#                 upper body loss: 0.0542, lower body loss: 0.0408, gender loss: 0.0290, bag loss: 0.0628, hat loss: 0.0218
#                 upper body acc: 0.8110, lower body acc: 0.8548, gender acc: 0.9666, bag acc: 0.9255, hat acc: 0.9644
#         EPOCH (2) --> VALIDATION LOSS: 0.0334, VALIDATION ACCURACY: 0.9250,
#                 upper body loss: 0.0454, lower body loss: 0.0316, gender loss: 0.0281, bag loss: 0.0522, hat loss: 0.0099
#                 upper body acc: 0.8422, lower Body acc: 0.8908, gender acc: 0.9734, bag acc: 0.9423, hat acc: 0.9762
#         ...last_model.pth model saved with validation loss: 0.0337
#         ...best_model.pth model saved with validation loss: 0.0334
# Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [28:40<00:00,  1.35s/it, loss=0.187, accuracy=0.908]
# Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [28:39<00:00,  1.16s/it, loss=0.187, accuracy=0.908]
#         EPOCH (3) --> TRAINING LOSS: 0.0399, TRAINING ACCURACY: 0.9077,
#                 upper body loss: 0.0522, lower body loss: 0.0391, gender loss: 0.0277, bag loss: 0.0601, hat loss: 0.0202
#                 upper body acc: 0.8168, lower body acc: 0.8590, gender acc: 0.9681, bag acc: 0.9282, hat acc: 0.9663
#         EPOCH (3) --> VALIDATION LOSS: 0.0347, VALIDATION ACCURACY: 0.9221,
#                 upper body loss: 0.0481, lower body loss: 0.0312, gender loss: 0.0284, bag loss: 0.0544, hat loss: 0.0115
#                 upper body acc: 0.8295, lower Body acc: 0.8900, gender acc: 0.9751, bag acc: 0.9427, hat acc: 0.9733
#         ...last_model.pth model saved with validation loss: 0.0334
# [2025-05-02 09:21:06] (base) -  Training started with 0.0 GB of memory reserved and 0.0 GB allocated.
#         ...checkpoint loaded from last_model.pth.
# Training: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:14<00:00,  1.23s/it, loss=0.23, accuracy=0.911]
# Training: 100%|████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:11<00:00,  1.06s/it, loss=0.23, accuracy=0.911]
#         EPOCH (4) --> TRAINING LOSS: 0.0382, TRAINING ACCURACY: 0.9106,
#                 upper body loss: 0.0502, lower body loss: 0.0380, gender loss: 0.0263, bag loss: 0.0576, hat loss: 0.0189
#                 upper body acc: 0.8219, lower body acc: 0.8631, gender acc: 0.9692, bag acc: 0.9309, hat acc: 0.9681
#         EPOCH (4) --> VALIDATION LOSS: 0.0335, VALIDATION ACCURACY: 0.9246,
#                 upper body loss: 0.0458, lower body loss: 0.0307, gender loss: 0.0273, bag loss: 0.0525, hat loss: 0.0113
#                 upper body acc: 0.8423, lower Body acc: 0.8924, gender acc: 0.9713, bag acc: 0.9414, hat acc: 0.9755
#         ...last_model.pth model saved with validation loss: 0.0334
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:07<00:00,  1.23s/it, loss=0.115, accuracy=0.913]
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:06<00:00,  1.05s/it, loss=0.115, accuracy=0.913]
#         EPOCH (5) --> TRAINING LOSS: 0.0368, TRAINING ACCURACY: 0.9132,
#                 upper body loss: 0.0486, lower body loss: 0.0371, gender loss: 0.0250, bag loss: 0.0556, hat loss: 0.0177
#                 upper body acc: 0.8265, lower body acc: 0.8651, gender acc: 0.9705, bag acc: 0.9338, hat acc: 0.9701
#         EPOCH (5) --> VALIDATION LOSS: 0.0343, VALIDATION ACCURACY: 0.9228,
#                 upper body loss: 0.0449, lower body loss: 0.0300, gender loss: 0.0281, bag loss: 0.0504, hat loss: 0.0183
#                 upper body acc: 0.8438, lower Body acc: 0.8961, gender acc: 0.9702, bag acc: 0.9440, hat acc: 0.9601
#         ...last_model.pth model saved with validation loss: 0.0334
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:27<00:00,  1.24s/it, loss=0.2, accuracy=0.919]
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:25<00:00,  1.06s/it, loss=0.2, accuracy=0.919]
#         EPOCH (6) --> TRAINING LOSS: 0.0341, TRAINING ACCURACY: 0.9187,
#                 upper body loss: 0.0462, lower body loss: 0.0352, gender loss: 0.0231, bag loss: 0.0506, hat loss: 0.0156
#                 upper body acc: 0.8355, lower body acc: 0.8732, gender acc: 0.9722, bag acc: 0.9394, hat acc: 0.9731
#         EPOCH (6) --> VALIDATION LOSS: 0.0324, VALIDATION ACCURACY: 0.9285,
#                 upper body loss: 0.0435, lower body loss: 0.0301, gender loss: 0.0262, bag loss: 0.0540, hat loss: 0.0081
#                 upper body acc: 0.8509, lower Body acc: 0.8933, gender acc: 0.9754, bag acc: 0.9402, hat acc: 0.9826
#         ...last_model.pth model saved with validation loss: 0.0334
#         ...best_model.pth model saved with validation loss: 0.0324
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:24<00:00,  1.24s/it, loss=0.105, accuracy=0.921]
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:22<00:00,  1.09s/it, loss=0.105, accuracy=0.921]
#         EPOCH (7) --> TRAINING LOSS: 0.0332, TRAINING ACCURACY: 0.9207,
#                 upper body loss: 0.0452, lower body loss: 0.0345, gender loss: 0.0223, bag loss: 0.0490, hat loss: 0.0147
#                 upper body acc: 0.8395, lower body acc: 0.8753, gender acc: 0.9734, bag acc: 0.9408, hat acc: 0.9743
#         EPOCH (7) --> VALIDATION LOSS: 0.0336, VALIDATION ACCURACY: 0.9252,
#                 upper body loss: 0.0451, lower body loss: 0.0308, gender loss: 0.0306, bag loss: 0.0512, hat loss: 0.0103
#                 upper body acc: 0.8434, lower Body acc: 0.8928, gender acc: 0.9678, bag acc: 0.9428, hat acc: 0.9790
#         ...last_model.pth model saved with validation loss: 0.0324
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:25<00:00,  1.24s/it, loss=0.103, accuracy=0.922]
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:23<00:00,  1.05s/it, loss=0.103, accuracy=0.922] 
#         EPOCH (8) --> TRAINING LOSS: 0.0322, TRAINING ACCURACY: 0.9218,                                                                                                                                  
#                 upper body loss: 0.0445, lower body loss: 0.0339, gender loss: 0.0212, bag loss: 0.0470, hat loss: 0.0143
#                 upper body acc: 0.8403, lower body acc: 0.8757, gender acc: 0.9740, bag acc: 0.9435, hat acc: 0.9753
#         EPOCH (8) --> VALIDATION LOSS: 0.0331, VALIDATION ACCURACY: 0.9257,
#                 upper body loss: 0.0458, lower body loss: 0.0303, gender loss: 0.0267, bag loss: 0.0518, hat loss: 0.0110
#                 upper body acc: 0.8413, lower Body acc: 0.8934, gender acc: 0.9727, bag acc: 0.9438, hat acc: 0.9771
#         ...last_model.pth model saved with validation loss: 0.0324
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:32<00:00,  1.25s/it, loss=0.139, accuracy=0.924]
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:31<00:00,  1.08s/it, loss=0.139, accuracy=0.924] 
#         EPOCH (9) --> TRAINING LOSS: 0.0312, TRAINING ACCURACY: 0.9236,                                                                                                                                  
#                 upper body loss: 0.0435, lower body loss: 0.0336, gender loss: 0.0204, bag loss: 0.0453, hat loss: 0.0131
#                 upper body acc: 0.8438, lower body acc: 0.8768, gender acc: 0.9756, bag acc: 0.9451, hat acc: 0.9766
#         EPOCH (9) --> VALIDATION LOSS: 0.0343, VALIDATION ACCURACY: 0.9242,
#                 upper body loss: 0.0460, lower body loss: 0.0304, gender loss: 0.0282, bag loss: 0.0542, hat loss: 0.0126
#                 upper body acc: 0.8390, lower Body acc: 0.8945, gender acc: 0.9706, bag acc: 0.9432, hat acc: 0.9740
#         ...last_model.pth model saved with validation loss: 0.0324
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:55<00:00,  1.26s/it, loss=0.137, accuracy=0.927]
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:53<00:00,  1.08s/it, loss=0.137, accuracy=0.927] 
#         EPOCH (10) --> TRAINING LOSS: 0.0297, TRAINING ACCURACY: 0.9268,                                                                                                                                 
#                 upper body loss: 0.0423, lower body loss: 0.0324, gender loss: 0.0188, bag loss: 0.0426, hat loss: 0.0122
#                 upper body acc: 0.8490, lower body acc: 0.8812, gender acc: 0.9769, bag acc: 0.9483, hat acc: 0.9788
#         EPOCH (10) --> VALIDATION LOSS: 0.0339, VALIDATION ACCURACY: 0.9258,
#                 upper body loss: 0.0459, lower body loss: 0.0301, gender loss: 0.0281, bag loss: 0.0561, hat loss: 0.0094
#                 upper body acc: 0.8426, lower Body acc: 0.8922, gender acc: 0.9705, bag acc: 0.9421, hat acc: 0.9816
#         ...last_model.pth model saved with validation loss: 0.0324
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:27<00:00,  1.24s/it, loss=0.186, accuracy=0.928]
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:26<00:00,  1.09s/it, loss=0.186, accuracy=0.928] 
#         EPOCH (11) --> TRAINING LOSS: 0.0290, TRAINING ACCURACY: 0.9281,                                                                                                                                 
#                 upper body loss: 0.0415, lower body loss: 0.0322, gender loss: 0.0185, bag loss: 0.0413, hat loss: 0.0117
#                 upper body acc: 0.8518, lower body acc: 0.8822, gender acc: 0.9773, bag acc: 0.9501, hat acc: 0.9793
#         EPOCH (11) --> VALIDATION LOSS: 0.0342, VALIDATION ACCURACY: 0.9266,
#                 upper body loss: 0.0445, lower body loss: 0.0301, gender loss: 0.0291, bag loss: 0.0571, hat loss: 0.0104
#                 upper body acc: 0.8460, lower Body acc: 0.8946, gender acc: 0.9704, bag acc: 0.9428, hat acc: 0.9794
#         ...last_model.pth model saved with validation loss: 0.0324
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:43<00:00,  1.25s/it, loss=0.112, accuracy=0.929]
# Training: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:40<00:00,  1.07s/it, loss=0.112, accuracy=0.929] 
#         EPOCH (12) --> TRAINING LOSS: 0.0286, TRAINING ACCURACY: 0.9290,                                                                                                                                 
#                 upper body loss: 0.0414, lower body loss: 0.0319, gender loss: 0.0183, bag loss: 0.0400, hat loss: 0.0114
#                 upper body acc: 0.8520, lower body acc: 0.8838, gender acc: 0.9774, bag acc: 0.9523, hat acc: 0.9796
#         EPOCH (12) --> VALIDATION LOSS: 0.0343, VALIDATION ACCURACY: 0.9255,
#                 upper body loss: 0.0466, lower body loss: 0.0307, gender loss: 0.0286, bag loss: 0.0563, hat loss: 0.0092
#                 upper body acc: 0.8398, lower Body acc: 0.8913, gender acc: 0.9715, bag acc: 0.9437, hat acc: 0.9813
#         ...last_model.pth model saved with validation loss: 0.0324
# Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:36<00:00,  1.25s/it, loss=0.124, accuracy=0.93]
# Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1278/1278 [26:34<00:00,  1.06s/it, loss=0.124, accuracy=0.93] 
#         EPOCH (13) --> TRAINING LOSS: 0.0278, TRAINING ACCURACY: 0.9303,                                                                                                                                 
#                 upper body loss: 0.0408, lower body loss: 0.0315, gender loss: 0.0174, bag loss: 0.0382, hat loss: 0.0108
#                 upper body acc: 0.8532, lower body acc: 0.8841, gender acc: 0.9785, bag acc: 0.9540, hat acc: 0.9815
#         EPOCH (13) --> VALIDATION LOSS: 0.0347, VALIDATION ACCURACY: 0.9255,
#                 upper body loss: 0.0457, lower body loss: 0.0303, gender loss: 0.0284, bag loss: 0.0579, hat loss: 0.0112
#                 upper body acc: 0.8437, lower Body acc: 0.8927, gender acc: 0.9709, bag acc: 0.9422, hat acc: 0.9782
#         ...last_model.pth model saved with validation loss: 0.0324
# Early stopping triggered
# Epoch:  56%|█████████████████████████████████████████████████████████████████▊                                                   | 9/16 [4:56:25<3:50:33, 1976.19s/it]