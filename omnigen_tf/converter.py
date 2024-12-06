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
    
    def convert_torch_to_tf(self, state_dict):
        """Convert PyTorch state dict to TensorFlow weights.
        
        Args:
            state_dict: PyTorch state dictionary
            
        Returns:
            List of TensorFlow weight tensors
        """
        # Define layer mapping between PyTorch and TensorFlow
        layer_mapping = {
            'x_embedder': 'x_embedder/proj',
            'input_x_embedder': 'input_x_embedder/proj',
            'time_token': 'time_token',
            't_embedder': 't_embedder',
            'pos_embed': 'pos_embed',
            'final_layer': 'final_layer',
            'llm': 'llm'
        }
        
        tf_weights = []
        
        # Convert each layer's weights
        for pt_name, weight in state_dict.items():
            # Convert to numpy array
            weight_np = weight.numpy()
            
            # Handle convolution weights
            if 'proj' in pt_name:
                # PyTorch conv weights are [out_channels, in_channels, height, width]
                # TensorFlow expects [height, width, in_channels, out_channels]
                weight_np = np.transpose(weight_np, (2, 3, 1, 0))
                
            # Handle linear layer weights
            elif 'linear' in pt_name or 'dense' in pt_name:
                # PyTorch linear weights are [out_features, in_features]
                # TensorFlow expects [in_features, out_features]
                weight_np = np.transpose(weight_np)
                
            # Handle attention weights
            elif any(x in pt_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                # Attention weights need special handling
                if weight_np.ndim == 2:
                    weight_np = np.transpose(weight_np)
                    
            # Handle layer norm weights
            elif 'norm' in pt_name:
                # Layer norm weights typically don't need transformation
                pass
                
            # Convert to TensorFlow tensor
            tf_tensor = tf.convert_to_tensor(weight_np)
            tf_weights.append(tf_tensor)
            
        return tf_weights
        
    def save_tf_weights(self, tf_weights, output_path):
        """Save converted TensorFlow weights."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Save weights in TensorFlow format
        np.save(os.path.join(output_path, 'tf_weights.npy'), tf_weights)
        
    def load_local_weights(self, weights_path: str):
        """Load weights from local safetensors file."""
        return load_file(weights_path)
        
    def save_weights(self, weights, weights_path: str):
        """Save weights to local safetensors file."""
        save_file(weights, weights_path)
