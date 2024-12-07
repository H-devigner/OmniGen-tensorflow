import torch
import os
from huggingface_hub import snapshot_download
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_model_to_onnx(model_name="Shitao/omnigen-v1", output_path="omnigen_model.onnx"):
    """
    Export the OmniGen PyTorch model to ONNX format.
    
    Args:
        model_name (str): HuggingFace model name/path
        output_path (str): Path to save the ONNX model
    """
    try:
        # Download the model
        logger.info(f"Downloading model from {model_name}...")
        model_path = snapshot_download(model_name)
        
        # Import OmniGen modules
        from OmniGen.model import OmniGen
        from OmniGen.transformer import Phi3Config
        
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
            verbose=True
        )
        
        # Verify the exported model
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed!")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error during model export: {str(e)}")
        raise

if __name__ == "__main__":
    export_model_to_onnx()
