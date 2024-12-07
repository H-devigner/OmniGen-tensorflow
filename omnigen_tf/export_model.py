import torch
import os
from huggingface_hub import hf_hub_download
import logging
import json
from pathlib import Path
import tensorflow as tf
from safetensors import safe_open
from safetensors.torch import load_file

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_tensorflow(state_dict, output_path):
    """Convert PyTorch state dict to TensorFlow format"""
    logger.info("Converting model to TensorFlow...")
    
    class TFWrapper(tf.keras.Model):
        def __init__(self):
            super().__init__()
            # Initialize layers based on state dict structure
            self.build_layers(state_dict)
            
        def build_layers(self, state_dict):
            # Map PyTorch state dict to TensorFlow layers
            for key, value in state_dict.items():
                if 'weight' in key:
                    layer_name = key.replace('.weight', '')
                    shape = value.shape
                    if len(shape) == 2:  # Linear layer
                        setattr(self, layer_name, tf.keras.layers.Dense(
                            units=shape[0],
                            input_shape=(shape[1],),
                            use_bias=False
                        ))
                    elif len(shape) == 4:  # Conv layer
                        setattr(self, layer_name, tf.keras.layers.Conv2D(
                            filters=shape[0],
                            kernel_size=shape[2:],
                            use_bias=False
                        ))
            
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, 4, 64, 64], dtype=tf.float32, name='x'),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name='timestep'),
            tf.TensorSpec(shape=[None, 77], dtype=tf.int32, name='input_ids'),
            tf.TensorSpec(shape=[None, 77], dtype=tf.int32, name='attention_mask'),
            tf.TensorSpec(shape=[None, 77], dtype=tf.int32, name='position_ids')
        ])
        def call(self, x, timestep, input_ids, attention_mask, position_ids):
            # Forward pass implementation
            # This is a placeholder - actual implementation will need to match OmniGen architecture
            return x  # For now, just return input as placeholder
    
    # Create wrapper and save
    tf_model = TFWrapper()
    tf.saved_model.save(tf_model, output_path)
    logger.info(f"Model converted and saved to {output_path}")
    return output_path

def download_and_convert(model_name="Shitao/omnigen-v1", output_dir="./"):
    """
    Download the OmniGen model and convert it to TensorFlow format.
    
    Args:
        model_name (str): HuggingFace model name/path
        output_dir (str): Directory to save the converted model
    """
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Download model files
        logger.info(f"Downloading model files from {model_name}...")
        
        try:
            # Try safetensors first
            weights_path = hf_hub_download(model_name, filename="model.safetensors")
            state_dict = load_file(weights_path)
        except:
            # Fall back to PyTorch weights
            weights_path = hf_hub_download(model_name, filename="pytorch_model.bin")
            state_dict = torch.load(weights_path)
        
        # Convert to TensorFlow
        tf_output_path = output_dir / "tf_model"
        convert_to_tensorflow(state_dict, str(tf_output_path))
        
        return str(tf_output_path)
        
    except Exception as e:
        logger.error(f"Error during model conversion: {str(e)}")
        raise

if __name__ == "__main__":
    download_and_convert()
