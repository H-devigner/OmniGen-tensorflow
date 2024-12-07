import torch
import os
import sys
from huggingface_hub import snapshot_download
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup paths for Kaggle environment"""
    # Get the current working directory
    current_dir = Path.cwd()
    
    # Add the OmniGen directory to sys.path
    omnigen_dir = current_dir.parent if "OmniGen-tensorflow" in str(current_dir) else current_dir
    if str(omnigen_dir) not in sys.path:
        sys.path.append(str(omnigen_dir))
    
    return current_dir

def export_model_to_onnx(model_name="Shitao/omnigen-v1", output_dir=None):
    """
    Export the OmniGen PyTorch model to ONNX format.
    
    Args:
        model_name (str): HuggingFace model name/path
        output_dir (str, optional): Directory to save the ONNX model. If None, uses current directory.
    """
    try:
        # Setup paths
        current_dir = setup_paths()
        
        # Import OmniGen modules after path setup
        from OmniGen.model import OmniGen
        from OmniGen.transformer import Phi3Config
        
        # Set output directory
        if output_dir is None:
            output_dir = current_dir
        output_path = os.path.join(output_dir, "omnigen_model.onnx")
        
        # Download the model
        logger.info(f"Downloading model from {model_name}...")
        cache_dir = os.path.join(current_dir, "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = snapshot_download(model_name, cache_dir=cache_dir)
        
        # Initialize model configuration
        config = Phi3Config(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=2048,
            layer_norm_eps=1e-6,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        # Load the model
        logger.info("Loading PyTorch model...")
        model = OmniGen.from_pretrained(model_path)
        model.eval()
        
        # Create dummy inputs
        batch_size = 1
        channels = 4
        height = 64
        width = 64
        seq_length = 77  # Standard sequence length for text tokens
        
        dummy_inputs = {
            'x': torch.randn(batch_size, channels, height, width),
            'timestep': torch.zeros(batch_size, dtype=torch.long),
            'input_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
            'input_img_latents': None,
            'input_image_sizes': None,
            'attention_mask': torch.ones(batch_size, seq_length, dtype=torch.long),
            'position_ids': torch.arange(seq_length)[None].expand(batch_size, -1)
        }
        
        # Export the model
        logger.info(f"Exporting model to ONNX format at {output_path}...")
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            output_path,
            input_names=list(dummy_inputs.keys()),
            output_names=['output'],
            dynamic_axes={
                'x': {0: 'batch_size'},
                'timestep': {0: 'batch_size'},
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'position_ids': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            },
            opset_version=12,
            export_params=True,
            do_constant_folding=True,
            verbose=False
        )
        logger.info(f"Model exported successfully to {output_path}!")
        return output_path
        
    except Exception as e:
        logger.error(f"Error during model export: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage in Kaggle
    export_model_to_onnx()
