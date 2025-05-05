# >>> sigpar.py
# Original author: Andrea Vincenzo Ricciardi
import torch

from collections import defaultdict
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoConfig, Siglip2VisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from src.utils import LOGGER

TASKS = {
    "upper_color": 11,
    "lower_color": 11,
    "gender": 1,
    "bag": 1,
    "hat": 1
}

class SIG2PAR(nn.Module):
    """ Class for the SIGPAR model. """
        
    def __init__(self):
        """ Initialize the SIGPAR model. """
        super(SIG2PAR, self).__init__()
        
        #--- Load the Vision Encoder model ---#
        config = AutoConfig.from_pretrained("./models/siglip2-so400m-patch16-naflex_VE")
        self.processor = AutoImageProcessor.from_pretrained("./models/siglip2-so400m-patch16-naflex_VE", use_fast=True)
        vision_model = Siglip2VisionModel._from_config(config.vision_config)
        # vision_model = Siglip2VisionModel.from_pretrained("./models/siglip2-so400m-patch16-naflex_VE", device_map="auto")

        self.vision_model = vision_model.vision_model

        # Freeze and Unfreeze specific layers of the vision model
        self.__set_trainable_layers()

        #--- Initialize the task-specific heads ---#
        self.heads = nn.ModuleDict({
            task: TaskHead(input_dim=vision_model.config.hidden_size, output_dim=num_classes)
            for task, num_classes in TASKS.items()
        })        
        
        
        del vision_model
        torch.cuda.empty_cache()
        LOGGER.debug(f"► SIGPAR Model Initialized.")
        LOGGER.debug(f"{self.__str__()}")
        
    @torch.no_grad()
    def generate(self, image : Image.Image) -> dict:
        """ Generate predictions for the input image using the SIGPAR model. 
        
        Parameters
        ----------
        image : Image.Image
            Input image to the model.
            
        Returns
        -------
        dict
            Dictionary containing the outputs for each task.
        """
        #--- Process the image ---#
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        #--- Forward pass through the model ---#
        outputs = self.forward(**inputs)
        
        #--- Get the predictions for each task ---#
        return {
            task: torch.argmax(logits, dim=1).item()
            for task, logits in outputs.items()
        }
        
    def forward(self,
        pixel_values: torch.Tensor | None = None,               # Pixel values of the image
        pixel_attention_mask: torch.Tensor | None = None,       # Attention mask for the pixel values
        spatial_shapes: torch.LongTensor | None = None,         # Spatial shapes of the image
        output_attentions: bool | None = None,                  # Output attentions
        output_hidden_states: bool | None = None,               # Output hidden states
    ):
        """ Forward pass of the SIGPAR model. 
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the model.
            
        Returns
        -------
        dict
            Dictionary containing the outputs for each task.
        """
        pixel_values = pixel_values.squeeze(1) if pixel_values is not None and pixel_values.dim() == 4 else pixel_values
        pixel_attention_mask = pixel_attention_mask.squeeze(1) if pixel_attention_mask is not None and pixel_attention_mask.dim() == 3 else None        
        spatial_shapes = spatial_shapes.squeeze(1) if spatial_shapes is not None and spatial_shapes.dim() == 3 else None
                
        #--- Forward pass through the vision model ---#
        outputs : BaseModelOutputWithPooling = self.vision_model(
            pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        # Get the last hidden state of the vision model
        sequence_output = outputs.last_hidden_state

        #--- Average Pool to patch tokens ---#
        sequence_output = torch.mean(sequence_output, dim=1)

        #--- Forward pass through the task-specific classifier ---#
        logits = {
            task: head(sequence_output)
            for task, head in self.heads.items()
        }
        
        return logits
    
    @property
    def device(self) -> torch.device:
        """ Device property of the model. """
        return next(self.parameters()).device
        
    def __set_trainable_layers(self):
        """ Post initialization method. """
        #--- Freeze the model parameters except for the last layers ---#        
        self.vision_model.requires_grad_(False)
        self.vision_model.encoder.layers[-3:].requires_grad_(True)
        self.vision_model.post_layernorm.requires_grad_(True)
        self.vision_model.head.requires_grad_(True)
        self.vision_model.head.requires_grad_(True)
    
    def __str__(self) -> str:
        """ Returns a string representation of the model summary. 
        
        Returns 
        -------
        str
            String representation of the model summary.
        """
        #--- Lambda functions for formatting ---#
        format_number = lambda n: f"{n:,}"
        key_numeric = lambda s: int(s.split(".")[-1]) if s.split(".")[-1].isdigit() else s

        #--- Create the model summary ---#
        lines = []
        divider = "=" * 96
        lines.append("🧠 SIGPAR Model Summary")
        lines.append(divider)
        
        # Header
        header = f"{'Idx':^5} {'Module':<40} {'#Params':>15} {'Trainable':>12} {'Shape':>20}"
        lines.append(header)
        lines.append(divider)

        #--- Group the parameters by module ---#
        grouped = {
            "vision_model.embeddings": [],
            "vision_model.encoder": defaultdict(list),
            "vision_model.post_layernorm": [],
            "vision_model.head": [],
            "heads": defaultdict(list),
        }

        #--- Count the number of parameters in each module ---#
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        for name, p in self.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
            else:
                non_trainable_params += p.numel()
                
            # Vision Model Encoder
            if name.startswith("vision_model.encoder"):
                number_layer = name.split('.')[3]
                grouped["vision_model.encoder"][number_layer].append((name, p))
            # Heads
            elif name.startswith("heads"):
                task = name.split('.')[1]
                grouped["heads"][task].append((name, p))
            # Other modules
            else:
                key = ".".join(name.split(".")[:2])
                if key in grouped:
                    grouped[key].append((name, p))

        #--- Append the parameters to the summary ---#
        for idx, (group_name, group) in enumerate(grouped.items()):
            if isinstance(group, list):
                total = sum(p.numel() for _, p in group)
                shape = str(list(group[0][1].shape))
                trainable = group[0][1].requires_grad
                lines.append(f"{idx:^5} {group_name:<40} {format_number(total):>15} {str(trainable):>12} {shape:>20}")
            elif isinstance(group, dict):
                total = sum(p.numel() for params in group.values() for _, p in params)
                lines.append(f"{idx:^5} {group_name:<40} {format_number(total):>15} {'/':>12} {'/':>20}")
                lines.append("-" * 96)
                for sub_key, params in sorted(group.items(), key=lambda x: key_numeric(x[0])):
                    if params:
                        sub_total = sum(p.numel() for _, p in params)
                        shape = str(list(params[0][1].shape))
                        trainable = params[0][1].requires_grad
                        lines.append(f"{'':^5} {group_name + '.' + sub_key:<40} {format_number(sub_total):>15} {str(trainable):>12} {shape:>20}")
                lines.append("-" * 96)
                
        #--- Append the total parameters to the summary ---#
        lines.append(f"🔢 Total Parameters     : {format_number(total_params)}")
        lines.append(f"✅ Trainable Parameters : {format_number(trainable_params)}")
        lines.append(f"❌ Frozen Parameters    : {format_number(non_trainable_params)}")
        lines.append(divider)

        return "\n".join(lines)

class TaskHead(nn.Module):
    """ Class for the task-specific head. """
    
    def __init__(self, input_dim : int, output_dim : int, hidden_dim : int = 512, dropout : float = 0.3):
        """ Initialize the task-specific head. 
        
        Parameters
        ----------
        input_dim : int
            Input dimension of the task head.
        output_dim : int
            Output dimension of the task head.
        hidden_dim : int, optional
            Hidden dimension of the task head, by default 512.
        dropout : float, optional
            Dropout rate of the task head, by default 0.3.
        """
        super(TaskHead, self).__init__()
        
        #--- Set the classifier layers ---#
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=input_dim, out_features=hidden_dim),
        #     nn.SiLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(in_features=hidden_dim, out_features=output_dim),
        # )
        self.classifier = nn.Linear(in_features=input_dim, out_features=output_dim)
        
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """ Forward pass of the task head. 
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the task head.
            
        Returns
        -------
        torch.Tensor
            Output tensor from the task head.
        """
        #--- Forward pass through the classifier layers ---#
        return self.classifier(x)

if __name__ == '__main__':
    #--- Example usage of the SIGPAR model ---#
    model = SIG2PAR().to("cuda")
    
# [2025-04-27 11:51:30] (base) -  SIGPAR Model Initialized:  SIGPAR Model Summary
# -------------------------
# Vision Encoder: Siglip2VisionTransformer
# Hidden Size:    1152

#  Trainable Parameters:
#   Total:   427,916,889
#   Trainable: 30,508,985
#   Frozen:    397,407,904
# -------------------------
# Train Loader size: 1278 | Train Loader dataset size: 81772
# Validation Loader size: 145 | Validation Loader dataset size: 9219
# [2025-04-27 11:51:30] (base) -  Training started with 1636.0 GB of memory reserved and 1632.380859375 GB allocated.
# Epoch:   0%|                                                                                                                                                                                  | 0/20 [00:00<?, ?it/s]
#         EPOCH (0) --> TRAINING LOSS: 0.0609, TRAINING ACCURACY: 0.8613,
#                 upper body loss: 0.0869, lower body loss: 0.0633, gender loss: 0.0389, bag loss: 0.0820, hat loss: 0.0333      
#                 upper body acc: 0.7073, lower body acc: 0.7888, gender acc: 0.9564, bag acc: 0.9048, hat acc: 0.9494
#         EPOCH (0) --> VALIDATION LOSS: 0.0346, VALIDATION ACCURACY: 0.9235,
#                 upper body loss: 0.0513, lower body loss: 0.0343, gender loss: 0.0287, bag loss: 0.0469, hat loss: 0.0118      
#                 upper body acc: 0.8394, lower Body acc: 0.8848, gender acc: 0.9741, bag acc: 0.9476, hat acc: 0.9716
#         ...best model saved with validation loss: 0.0346
# Epoch:   5%|████████▏                                                                                                                                                          | 1/20 [1:16:04<24:05:23, 4564.40s/it]
#         EPOCH (1) --> TRAINING LOSS: 0.0473, TRAINING ACCURACY: 0.8952,
#                 upper body loss: 0.0600, lower body loss: 0.0449, gender loss: 0.0334, bag loss: 0.0718, hat loss: 0.0263      
#                 upper body acc: 0.7958, lower body acc: 0.8428, gender acc: 0.9627, bag acc: 0.9153, hat acc: 0.9592
#         EPOCH (1) --> VALIDATION LOSS: 0.0373, VALIDATION ACCURACY: 0.9150,
#                 upper body loss: 0.0520, lower body loss: 0.0334, gender loss: 0.0296, bag loss: 0.0566, hat loss: 0.0150      
#                 upper body acc: 0.8254, lower Body acc: 0.8852, gender acc: 0.9723, bag acc: 0.9336, hat acc: 0.9587
# Epoch:  10%|████████████████▎                                                                                                                                                  | 2/20 [2:31:38<22:43:57, 4546.51s/it]
# EPOCH (2) --> TRAINING LOSS: 0.0456, TRAINING ACCURACY: 0.8984,
#                 upper body loss: 0.0588, lower body loss: 0.0432, gender loss: 0.0325, bag loss: 0.0690, hat loss: 0.0244
#                 upper body acc: 0.7994, lower body acc: 0.8479, gender acc: 0.9643, bag acc: 0.9196, hat acc: 0.9608
#         EPOCH (2) --> VALIDATION LOSS: 0.0357, VALIDATION ACCURACY: 0.9185,
#                 upper body loss: 0.0503, lower body loss: 0.0322, gender loss: 0.0279, bag loss: 0.0541, hat loss: 0.0139
#                 upper body acc: 0.8333, lower Body acc: 0.8884, gender acc: 0.9688, bag acc: 0.9372, hat acc: 0.9649
# Epoch:  15%|████████████████████████▍                                                                                                                                          | 3/20 [3:47:50<21:31:31, 4558.30s/it] 
#         EPOCH (3) --> TRAINING LOSS: 0.0453, TRAINING ACCURACY: 0.8985,
#                 upper body loss: 0.0589, lower body loss: 0.0429, gender loss: 0.0321, bag loss: 0.0684, hat loss: 0.0242
#                 upper body acc: 0.7988, lower body acc: 0.8487, gender acc: 0.9643, bag acc: 0.9191, hat acc: 0.9615
#         EPOCH (3) --> VALIDATION LOSS: 0.0353, VALIDATION ACCURACY: 0.9205,
#                 upper body loss: 0.0510, lower body loss: 0.0333, gender loss: 0.0286, bag loss: 0.0527, hat loss: 0.0107
#                 upper body acc: 0.8290, lower Body acc: 0.8863, gender acc: 0.9692, bag acc: 0.9403, hat acc: 0.9775
# Epoch:  20%|████████████████████████████████▌                                                                                                                                  | 4/20 [5:04:18<20:18:38, 4569.92s/it] 
#         EPOCH (4) --> TRAINING LOSS: 0.0445, TRAINING ACCURACY: 0.8997,
#                 upper body loss: 0.0581, lower body loss: 0.0426, gender loss: 0.0315, bag loss: 0.0670, hat loss: 0.0233
#                 upper body acc: 0.8012, lower body acc: 0.8482, gender acc: 0.9655, bag acc: 0.9207, hat acc: 0.9631
#         EPOCH (4) --> VALIDATION LOSS: 0.0372, VALIDATION ACCURACY: 0.9190,
#                 upper body loss: 0.0501, lower body loss: 0.0316, gender loss: 0.0339, bag loss: 0.0582, hat loss: 0.0122
#                 upper body acc: 0.8311, lower Body acc: 0.8914, gender acc: 0.9664, bag acc: 0                upper body loss: 0.0501, lower body loss: 0.0316, gender loss: 0.0339, bag loss: 0.0582, hat loss: 0.0122
#                 upper body acc: 0.8311, lower Body acc: 0.8914, gender acc: 0.9664, bag acc: 0.9362, hat acc: 0.9701
# Epoch:  25%|████████████████████████████████████████▊                                                                                                                          | 5/20 [6:21:50<19:09:53, 4599.59s/it]
#         EPOCH (5) --> TRAINING LOSS: 0.0432, TRAINING ACCURACY: 0.9017,
#                 upper body loss: 0.0566, lower body loss: 0.0413, gender loss: 0.0305, bag loss: 0.0654, hat loss: 0.0224
#                 upper body acc: 0.8041, lower body acc: 0.8526, gender acc: 0.9654, bag acc: 0.9227, hat acc: 0.9639
#         EPOCH (5) --> VALIDATION LOSS: 0.0369, VALIDATION ACCURACY: 0.9197,
#                 upper body loss: 0.0479, lower body loss: 0.0336, gender loss: 0.0282, bag loss: 0.0578, hat loss: 0.0170
#                 upper body acc: 0.8407, lower Body acc: 0.8822, gender acc: 0.9756, bag acc: 0.9411, hat acc: 0.9589
# Epoch:  30%|████████████████████████████████████████████████▉                                                                                                                  | 6/20 [7:39:26<17:57:40, 4618.59s/it]
# EPOCH (6) --> TRAINING LOSS: 0.0420, TRAINING ACCURACY: 0.9045,
#                 upper body loss: 0.0554, lower body loss: 0.0406, gender loss: 0.0291, bag loss: 0.0638, hat loss: 0.0211
#                 upper body acc: 0.8087, lower body acc: 0.8561, gender acc: 0.9671, bag acc: 0.9246, hat acc: 0.9660
#         EPOCH (6) --> VALIDATION LOSS: 0.0336, VALIDATION ACCURACY: 0.9250,
#                 upper body loss: 0.0440, lower body loss: 0.0312, gender loss: 0.0290, bag loss: 0.0515, hat loss: 0.0125
#                 upper body acc: 0.8488, lower Body acc: 0.8926, gender acc: 0.9732, bag acc: 0.9416, hat acc: 0.9688
#         ...best model saved with validation loss: 0.0336
# Epoch:  35%|█████████████████████████████████████████████████████████                                                                                                          | 7/20 [8:55:53<16:38:29, 4608.40s/it]
#         EPOCH (7) --> TRAINING LOSS: 0.0409, TRAINING ACCURACY: 0.9064,
#                 upper body loss: 0.0538, lower body loss: 0.0398, gender loss: 0.0282, bag loss: 0.0623, hat loss: 0.0201
#                 upper body acc: 0.8132, lower body acc: 0.8567, gender acc: 0.9683, bag acc: 0.9264, hat acc: 0.9672
#         EPOCH (7) --> VALIDATION LOSS: 0.0338, VALIDATION ACCURACY: 0.9223,
#                 upper body loss: 0.0489, lower body loss: 0.0320, gender loss: 0.0261, bag loss: 0.0510, hat loss: 0.0113
#                 upper body acc: 0.8341, lower Body acc: 0.8896, gender acc: 0.9735, bag acc: 0.9421, hat acc: 0.9723
# Epoch:  40%|████████████████████████████████████████████████████████████████▊                                                                                                 | 8/20 [10:10:39<15:13:54, 4569.51s/it] 
#         EPOCH (8) --> TRAINING LOSS: 0.0396, TRAINING ACCURACY: 0.9084,                                                                                                                                               
#                 upper body loss: 0.0523, lower body loss: 0.0389, gender loss: 0.0275, bag loss: 0.0601, hat loss: 0.0193
#                 upper body acc: 0.8171, lower body acc: 0.8590, gender acc: 0.9686, bag acc: 0.9295, hat acc: 0.9679
#         EPOCH (8) --> VALIDATION LOSS: 0.0355, VALIDATION ACCURACY: 0.9232,
#                 upper body loss: 0.0500, lower body loss: 0.0326, gender loss: 0.0276, bag loss: 0.0555, hat loss: 0.0118
#                 upper body acc: 0.8354, lower Body acc: 0.8883, gender acc: 0.9757, bag acc: 0.9399, hat acc: 0.9766
# Epoch:  45%|████████████████████████████████████████████████████████████████████████▉                                                                                         | 9/20 [11:28:05<14:02:06, 4593.27s/it] 
#         EPOCH (9) --> TRAINING LOSS: 0.0386, TRAINING ACCURACY: 0.9108,                                                                                                                                               
#                 upper body loss: 0.0508, lower body loss: 0.0379, gender loss: 0.0267, bag loss: 0.0594, hat loss: 0.0183
#                 upper body acc: 0.8227, lower body acc: 0.8626, gender acc: 0.9699, bag acc: 0.9293, hat acc: 0.9697
#         EPOCH (9) --> VALIDATION LOSS: 0.0348, VALIDATION ACCURACY: 0.9229,
#                 upper body loss: 0.0509, lower body loss: 0.0306, gender loss: 0.0281, bag loss: 0.0557, hat loss: 0.0087
#                 upper body acc: 0.8283, lower Body acc: 0.8943, gender acc: 0.9709, bag acc: 0.9388, hat acc: 0.9823
# -------------------------------------------------------------------------------------------------------------------------
#         EPOCH (7) --> TRAINING LOSS: 0.0409, TRAINING ACCURACY: 0.9062,
#                 upper body loss: 0.0540, lower body loss: 0.0393, gender loss: 0.0286, bag loss: 0.0622, hat loss: 0.0203
#                 upper body acc: 0.8116, lower body acc: 0.8592, gender acc: 0.9669, bag acc: 0.9267, hat acc: 0.9668
#         EPOCH (7) --> VALIDATION LOSS: 0.0344, VALIDATION ACCURACY: 0.9223,
#                 upper body loss: 0.0486, lower body loss: 0.0314, gender loss: 0.0285, bag loss: 0.0514, hat loss: 0.0122
#                 upper body acc: 0.8383, lower Body acc: 0.8924, gender acc: 0.9724, bag acc: 0.9389, hat acc: 0.9694
#         ...best model saved with validation loss: 0.0336
#                 EPOCH (8) --> TRAINING LOSS: 0.0398, TRAINING ACCURACY: 0.9081,
#                 upper body loss: 0.0523, lower body loss: 0.0388, gender loss: 0.0276, bag loss: 0.0611, hat loss: 0.0194
#                 upper body acc: 0.8165, lower body acc: 0.8608, gender acc: 0.9686, bag acc: 0.9273, hat acc: 0.9675
#         EPOCH (8) --> VALIDATION LOSS: 0.0351, VALIDATION ACCURACY: 0.9222,
#                 upper body loss: 0.0484, lower body loss: 0.0313, gender loss: 0.0275, bag loss: 0.0526, hat loss: 0.0155
#                 upper body acc: 0.8422, lower Body acc: 0.8954, gender acc: 0.9729, bag acc: 0.9402, hat acc: 0.9605
#         ...last_model.pth model saved with validation loss: 0.0336
#         EPOCH (9) --> TRAINING LOSS: 0.0386, TRAINING ACCURACY: 0.9103,                                                                                                                                               
#                 upper body loss: 0.0510, lower body loss: 0.0380, gender loss: 0.0265, bag loss: 0.0585, hat loss: 0.0187
#                 upper body acc: 0.8198, lower body acc: 0.8626, gender acc: 0.9698, bag acc: 0.9296, hat acc: 0.9694
#         EPOCH (9) --> VALIDATION LOSS: 0.0328, VALIDATION ACCURACY: 0.9247,
#                 upper body loss: 0.0464, lower body loss: 0.0319, gender loss: 0.0279, bag loss: 0.0473, hat loss: 0.0105
#                 upper body acc: 0.8413, lower Body acc: 0.8877, gender acc: 0.9737, bag acc: 0.9454, hat acc: 0.9753
#         ...last_model.pth model saved with validation loss: 0.0336
#         ...best_model.pth model saved with validation loss: 0.0328
#         EPOCH (10) --> TRAINING LOSS: 0.0374, TRAINING ACCURACY: 0.9128,                                                                                                                                              
#                 upper body loss: 0.0495, lower body loss: 0.0370, gender loss: 0.0256, bag loss: 0.0572, hat loss: 0.0179
#                 upper body acc: 0.8253, lower body acc: 0.8657, gender acc: 0.9705, bag acc: 0.9326, hat acc: 0.9700
#         EPOCH (10) --> VALIDATION LOSS: 0.0330, VALIDATION ACCURACY: 0.9282,
#                 upper body loss: 0.0452, lower body loss: 0.0312, gender loss: 0.0285, bag loss: 0.0508, hat loss: 0.0094
#                 upper body acc: 0.8490, lower Body acc: 0.8927, gender acc: 0.9733, bag acc: 0.9435, hat acc: 0.9825
#         ...last_model.pth model saved with validation loss: 0.0328
#         EPOCH (11) --> TRAINING LOSS: 0.0362, TRAINING ACCURACY: 0.9150,                                                                                                                                              
#                 upper body loss: 0.0479, lower body loss: 0.0362, gender loss: 0.0248, bag loss: 0.0555, hat loss: 0.0168
#                 upper body acc: 0.8302, lower body acc: 0.8679, gender acc: 0.9714, bag acc: 0.9334, hat acc: 0.9722
#         EPOCH (11) --> VALIDATION LOSS: 0.0342, VALIDATION ACCURACY: 0.9264,
#                 upper body loss: 0.0464, lower body loss: 0.0310, gender loss: 0.0291, bag loss: 0.0520, hat loss: 0.0126
#                 upper body acc: 0.8471, lower Body acc: 0.8934, gender acc: 0.9721, bag acc: 0.9426, hat acc: 0.9768
#         ...last_model.pth model saved with validation loss: 0.0328
#         EPOCH (12) --> TRAINING LOSS: 0.0348, TRAINING ACCURACY: 0.9180,
#                 upper body loss: 0.0463, lower body loss: 0.0351, gender loss: 0.0237, bag loss: 0.0534, hat loss: 0.0156
#                 upper body acc: 0.8362, lower body acc: 0.8717, gender acc: 0.9718, bag acc: 0.9361, hat acc: 0.9741
#         EPOCH (12) --> VALIDATION LOSS: 0.0353, VALIDATION ACCURACY: 0.9235,
#                 upper body loss: 0.0489, lower body loss: 0.0311, gender loss: 0.0285, bag loss: 0.0563, hat loss: 0.0117
#                 upper body acc: 0.8379, lower Body acc: 0.8929, gender acc: 0.9728, bag acc: 0.9413, hat acc: 0.9727
#         ...last_model.pth model saved with validation loss: 0.0328
# Epoch:  12%|████████████████████▋                                                                                                                                                | 1/8 [1:10:32<8:13:46, 4232.29s/it]
#         EPOCH (13) --> TRAINING LOSS: 0.0336, TRAINING ACCURACY: 0.9203,                                                                                                                                              
#                 upper body loss: 0.0446, lower body loss: 0.0341, gender loss: 0.0226, bag loss: 0.0517, hat loss: 0.0148
#                 upper body acc: 0.8403, lower body acc: 0.8737, gender acc: 0.9738, bag acc: 0.9383, hat acc: 0.9754
#         EPOCH (13) --> VALIDATION LOSS: 0.0339, VALIDATION ACCURACY: 0.9272,
#                 upper body loss: 0.0466, lower body loss: 0.0305, gender loss: 0.0285, bag loss: 0.0532, hat loss: 0.0107
#                 upper body acc: 0.8465, lower Body acc: 0.8963, gender acc: 0.9730, bag acc: 0.9421, hat acc: 0.9780
#         ...last_model.pth model saved with validation loss: 0.0328
# Epoch:  25%|█████████████████████████████████████████▎                                                                                                                           | 2/8 [2:20:55<7:02:41, 4226.90s/it]
#         EPOCH (14) --> TRAINING LOSS: 0.0320, TRAINING ACCURACY: 0.9236,                                                                                                                                              
#                 upper body loss: 0.0429, lower body loss: 0.0331, gender loss: 0.0213, bag loss: 0.0493, hat loss: 0.0133
#                 upper body acc: 0.8465, lower body acc: 0.8774, gender acc: 0.9746, bag acc: 0.9415, hat acc: 0.9778
#         EPOCH (14) --> VALIDATION LOSS: 0.0346, VALIDATION ACCURACY: 0.9268,
#                 upper body loss: 0.0469, lower body loss: 0.0312, gender loss: 0.0282, bag loss: 0.0568, hat loss: 0.0101
#                 upper body acc: 0.8471, lower Body acc: 0.8935, gender acc: 0.9717, bag acc: 0.9410, hat acc: 0.9810
#         ...last_model.pth model saved with validation loss: 0.0328
# Epoch:  38%|█████████████████████████████████████████████████████████████▉                                                                                                       | 3/8 [3:30:58<5:51:19, 4215.82s/it]
#         EPOCH (15) --> TRAINING LOSS: 0.0309, TRAINING ACCURACY: 0.9261,                                                                                                                                              
#                 upper body loss: 0.0415, lower body loss: 0.0321, gender loss: 0.0204, bag loss: 0.0479, hat loss: 0.0128
#                 upper body acc: 0.8514, lower body acc: 0.8814, gender acc: 0.9762, bag acc: 0.9428, hat acc: 0.9788
#         EPOCH (15) --> VALIDATION LOSS: 0.0363, VALIDATION ACCURACY: 0.9225,
#                 upper body loss: 0.0481, lower body loss: 0.0320, gender loss: 0.0334, bag loss: 0.0550, hat loss: 0.0130
#                 upper body acc: 0.8411, lower Body acc: 0.8907, gender acc: 0.9654, bag acc: 0.9411, hat acc: 0.9742
#         ...last_model.pth model saved with validation loss: 0.0328
# Epoch:  50%|██████████████████████████████████████████████████████████████████████████████████▌                                                                                  | 4/8 [4:41:09<4:40:55, 4214.00s/it]
#         EPOCH (16) --> TRAINING LOSS: 0.0297, TRAINING ACCURACY: 0.9285,                                                                                                                                              
#                 upper body loss: 0.0401, lower body loss: 0.0314, gender loss: 0.0191, bag loss: 0.0459, hat loss: 0.0118
#                 upper body acc: 0.8568, lower body acc: 0.8835, gender acc: 0.9773, bag acc: 0.9446, hat acc: 0.9800
#         EPOCH (16) --> VALIDATION LOSS: 0.0349, VALIDATION ACCURACY: 0.9256,
#                 upper body loss: 0.0489, lower body loss: 0.0310, gender loss: 0.0276, bag loss: 0.0549, hat loss: 0.0122
#                 upper body acc: 0.8402, lower Body acc: 0.8948, gender acc: 0.9723, bag acc: 0.9433, hat acc: 0.9774
#         ...last_model.pth model saved with validation loss: 0.0328