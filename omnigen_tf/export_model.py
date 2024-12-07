import torch
import os
from huggingface_hub import hf_hub_download
import logging
import json
from pathlib import Path
import tensorflow as tf
from transformers import AutoConfig, AutoModelForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_tensorflow(model, output_path):
    """Convert PyTorch model to TensorFlow format"""
    logger.info("Converting model to TensorFlow...")
    
    class TFWrapper(tf.keras.Model):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, 4, 64, 64], dtype=tf.float32, name='x'),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name='timestep'),
            tf.TensorSpec(shape=[None, 77], dtype=tf.int32, name='input_ids'),
            tf.TensorSpec(shape=[None, 77], dtype=tf.int32, name='attention_mask'),
            tf.TensorSpec(shape=[None, 77], dtype=tf.int32, name='position_ids')
        ])
        def call(self, x, timestep, input_ids, attention_mask, position_ids):
            return self.model(
                x=x,
                timestep=timestep,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
    
    # Create wrapper and save
    tf_model = TFWrapper(model)
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
        
        # Download configuration
        config = AutoConfig.from_pretrained(model_name)
        
        # Load PyTorch model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32
        )
        model.eval()
        
        # Convert to TensorFlow
        tf_output_path = output_dir / "tf_model"
        convert_to_tensorflow(model, str(tf_output_path))
        
        return str(tf_output_path)
        
    except Exception as e:
        logger.error(f"Error during model conversion: {str(e)}")
        raise

if __name__ == "__main__":
    download_and_convert()
