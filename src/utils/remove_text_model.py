# >>> remove_text_model.py
# Original author: Andrea Vincenzo Ricciardi

import os
from safetensors import safe_open
from safetensors.torch import save_file

def remove_text_model_weights(filepath : str, output_path : str):
    """Remove text model weights from a safetensors file and save the filtered tensors to a new file.
    
    Parameters
    ----------
    filepath : str
        Path to the original safetensors file.
    output_path : str
        Path to save the filtered safetensors file.
    """
    #--- Check if the input file exists ---#
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input file {filepath} does not exist.")

    #--- Check if the output directory exists, if not create it ---#
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    #--- Read the original safetensors file and filter out unwanted tensors ---#
    with safe_open(filepath, framework="pt", device="cpu") as f:
        filtered_tensors = {
            key: f.get_tensor(key) for key in f.keys() if 'text_model' not in key and "logit_" not in key
        }

    #--- Save the filtered tensors to a new safetensors file ---#
    save_file(filtered_tensors, output_path)
    
if __name__ == '__main__':
    #--- Example usage ---#
    input_path = "models/siglip2-so400m-patch16-naflex/model.safetensors"
    output_path = "models/siglip2-so400m-patch16-naflex_VE/model.safetensors"
    
    remove_text_model_weights(input_path, output_path)