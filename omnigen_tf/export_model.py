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

class TFOmniGen(tf.keras.Model):
    """TensorFlow implementation of OmniGen"""
    def __init__(self, state_dict):
        super().__init__()
        self.layers_dict = {}
        self.build_model_layers(state_dict)
        
    def build_model_layers(self, state_dict):
        """Build model layers from state dict"""
        # Track layer dimensions for proper initialization
        layer_dims = {}
        
        # First pass: collect layer dimensions
        for key, value in state_dict.items():
            if 'weight' in key:
                base_name = key.replace('.weight', '')
                layer_dims[base_name] = value.shape
        
        # Second pass: create layers
        for base_name, shape in layer_dims.items():
            if len(shape) == 2:  # Linear layer
                self.layers_dict[base_name] = tf.keras.layers.Dense(
                    units=shape[0],
                    use_bias=False,
                    name=base_name
                )
            elif len(shape) == 4:  # Conv layer
                self.layers_dict[base_name] = tf.keras.layers.Conv2D(
                    filters=shape[0],
                    kernel_size=shape[2:],
                    use_bias=False,
                    name=base_name
                )
    
    def build(self, input_shape):
        """Build the model (required by Keras)"""
        # Create a dummy input to build all layers
        dummy_inputs = {
            'x': tf.keras.Input(shape=(4, 64, 64), dtype=tf.float32),
            'timestep': tf.keras.Input(shape=(), dtype=tf.int32),
            'input_ids': tf.keras.Input(shape=(77,), dtype=tf.int32),
            'attention_mask': tf.keras.Input(shape=(77,), dtype=tf.int32),
            'position_ids': tf.keras.Input(shape=(77,), dtype=tf.int32)
        }
        self.call(dummy_inputs)
    
    @tf.function
    def call(self, inputs):
        """Forward pass"""
        x = inputs['x']
        timestep = inputs['timestep']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = inputs['position_ids']
        
        # Placeholder implementation
        # TODO: Implement actual forward pass logic
        return x

def convert_to_tensorflow(state_dict, output_path):
    """Convert PyTorch state dict to TensorFlow format"""
    logger.info("Converting model to TensorFlow...")
    
    # Create model
    model = TFOmniGen(state_dict)
    
    # Create input signature
    input_signature = {
        'x': tf.TensorSpec(shape=(None, 4, 64, 64), dtype=tf.float32, name='x'),
        'timestep': tf.TensorSpec(shape=(None,), dtype=tf.int32, name='timestep'),
        'input_ids': tf.TensorSpec(shape=(None, 77), dtype=tf.int32, name='input_ids'),
        'attention_mask': tf.TensorSpec(shape=(None, 77), dtype=tf.int32, name='attention_mask'),
        'position_ids': tf.TensorSpec(shape=(None, 77), dtype=tf.int32, name='position_ids')
    }
    
    # Save the model
    tf.saved_model.save(
        model,
        output_path,
        signatures={'serving_default': model.call.get_concrete_function(input_signature)}
    )
    
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
            logger.info("Loaded weights from safetensors")
        except Exception as e:
            logger.warning(f"Failed to load safetensors: {e}")
            # Fall back to PyTorch weights
            weights_path = hf_hub_download(model_name, filename="pytorch_model.bin")
            state_dict = torch.load(weights_path)
            logger.info("Loaded weights from PyTorch checkpoint")
        
        # Convert to TensorFlow
        tf_output_path = output_dir / "tf_model"
        convert_to_tensorflow(state_dict, str(tf_output_path))
        
        return str(tf_output_path)
        
    except Exception as e:
        logger.error(f"Error during model conversion: {str(e)}")
        raise

if __name__ == "__main__":
    download_and_convert()
