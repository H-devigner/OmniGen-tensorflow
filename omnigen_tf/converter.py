import os
import tensorflow as tf
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

class WeightConverter:
    """Converts PyTorch weights to TensorFlow format."""
    
    def __init__(self):
        pass
        
    def download_pytorch_weights(self, model_name):
        """Download PyTorch weights from HuggingFace hub."""
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_folder,
                ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
            )
        else:
            model_path = model_name
            
        # Load safetensors weights
        if os.path.exists(os.path.join(model_path, 'model.safetensors')):
            print("Loading safetensors weights...")
            weights = load_file(os.path.join(model_path, 'model.safetensors'))
        else:
            raise ValueError(f"No model weights found in {model_path}")
            
        # Load config
        config = self._load_config(model_path)
        return weights, config
        
    def _load_config(self, model_path):
        """Load model configuration."""
        config_path = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_path):
            raise ValueError(f"No config.json found in {model_path}")
            
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
        
    def convert_weights(self, pytorch_weights, layer_mapping):
        """Convert PyTorch weights to TensorFlow format."""
        tf_weights = {}
        
        for pt_name, tf_name in layer_mapping.items():
            if pt_name in pytorch_weights:
                weight = pytorch_weights[pt_name].numpy()
                
                # Handle different layer types
                if 'conv' in tf_name.lower():
                    # Convert NCHW to NHWC format for conv layers
                    weight = np.transpose(weight, (2, 3, 1, 0))
                elif 'dense' in tf_name.lower() or 'linear' in tf_name.lower():
                    # Transpose weights for dense/linear layers
                    if len(weight.shape) == 2:
                        weight = weight.transpose()
                        
                tf_weights[tf_name] = weight
                
        return tf_weights
    
    def save_tf_weights(self, tf_weights, output_path):
        """Save converted weights in TensorFlow format."""
        np.savez(output_path, **tf_weights)
        return output_path

    def load_local_weights(self, weights_path: str):
        """Load weights from local safetensors file."""
        return load_file(weights_path)
        
    def save_weights(self, weights, weights_path: str):
        """Save weights to local safetensors file."""
        save_file(weights, weights_path)
