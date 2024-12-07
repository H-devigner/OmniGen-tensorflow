import torch
import os
from huggingface_hub import hf_hub_download
import logging
import json
from pathlib import Path
import onnx
import onnx_tf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_tensorflow(onnx_model_path, output_path):
    """Convert ONNX model to TensorFlow format"""
    logger.info("Converting ONNX model to TensorFlow...")
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(output_path)
    logger.info(f"Model converted and saved to {output_path}")

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
        
        # Download the ONNX model if available
        try:
            onnx_path = hf_hub_download(model_name, filename="model.onnx")
            logger.info("Found ONNX model, using it directly")
        except Exception as e:
            logger.error(f"ONNX model not found: {e}")
            logger.info("Please provide an ONNX model or use the PyTorch conversion script")
            raise
            
        # Convert to TensorFlow
        tf_output_path = output_dir / "tf_model"
        convert_to_tensorflow(onnx_path, str(tf_output_path))
        
        return str(tf_output_path)
        
    except Exception as e:
        logger.error(f"Error during model conversion: {str(e)}")
        raise

if __name__ == "__main__":
    download_and_convert()
