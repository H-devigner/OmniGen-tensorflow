import torch
import os
import sys
from huggingface_hub import snapshot_download
import logging
from pathlib import Path
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_kaggle_environment():
    """Setup the Kaggle environment and required paths"""
    try:
        current_dir = Path.cwd()
        logger.info(f"Current working directory: {current_dir}")
        
        # Check if we're in the correct directory structure
        if "OmniGen-tensorflow" in str(current_dir):
            base_dir = current_dir.parent
        else:
            base_dir = current_dir
            
        # Add base directory to Python path
        if str(base_dir) not in sys.path:
            sys.path.append(str(base_dir))
            logger.info(f"Added {base_dir} to Python path")
            
        # Create cache directory
        cache_dir = current_dir / "model_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Verify OmniGen module is available
        omnigen_dir = base_dir / "OmniGen"
        if not omnigen_dir.exists():
            raise ImportError(f"OmniGen directory not found at {omnigen_dir}")
            
        if str(omnigen_dir) not in sys.path:
            sys.path.append(str(omnigen_dir))
            logger.info(f"Added {omnigen_dir} to Python path")
            
        return current_dir, cache_dir
        
    except Exception as e:
        logger.error(f"Error setting up Kaggle environment: {e}")
        raise

def verify_dependencies():
    """Verify all required dependencies are installed"""
    required_packages = [
        'torch',
        'onnx',
        'onnx-tf',
        'tensorflow',
        'huggingface_hub',
        'transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(f"Missing required packages: {', '.join(missing_packages)}")

def export_model_to_onnx(model_name="Shitao/omnigen-v1", output_dir=None):
    """
    Export the OmniGen PyTorch model to ONNX format.
    
    Args:
        model_name (str): HuggingFace model name/path
        output_dir (str, optional): Directory to save the ONNX model. If None, uses current directory.
    """
    try:
        # Verify dependencies first
        verify_dependencies()
        
        # Setup Kaggle environment
        current_dir, cache_dir = setup_kaggle_environment()
        
        # Now try to import OmniGen modules
        try:
            from OmniGen.model import OmniGen
            from OmniGen.transformer import Phi3Config
        except ImportError as e:
            logger.error(f"Failed to import OmniGen modules: {e}")
            logger.info("Checking Python path:")
            for path in sys.path:
                logger.info(f"  {path}")
            raise
        
        # Set output directory
        if output_dir is None:
            output_dir = current_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "omnigen_model.onnx"
        
        # Download the model
        logger.info(f"Downloading model from {model_name}...")
        model_path = snapshot_download(model_name, cache_dir=str(cache_dir))
        logger.info(f"Model downloaded to {model_path}")
        
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
        seq_length = 77
        
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
            str(output_path),
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
            verbose=True  # Enable verbose output for debugging
        )
        
        # Verify the exported model
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed!")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error during model export: {str(e)}")
        raise

if __name__ == "__main__":
    # For Kaggle notebook, you should run this after setting up the environment:
    # !git clone https://github.com/Shitao/OmniGen.git
    # !pip install onnx onnx-tf
    export_model_to_onnx()
