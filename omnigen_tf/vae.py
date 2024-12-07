"""VAE model for OmniGen."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, initializers
from typing import Optional, Tuple, Dict, Any
import json
import os
from safetensors import safe_open
from huggingface_hub import snapshot_download

from .utils import convert_torch_to_tf

class AutoencoderKL(tf.keras.Model):
    """VAE model with KL regularization."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.latent_channels = latent_channels
        self.sample_size = sample_size
        self.scaling_factor = scaling_factor
        
        # Initialize with truncated normal
        kernel_init = initializers.TruncatedNormal(stddev=0.02)
        
        # Build encoder
        self.encoder = self.build_encoder(
            kernel_initializer=kernel_init,
            bias_initializer='zeros'
        )
        
        # Build decoder
        self.decoder = self.build_decoder(
            kernel_initializer=kernel_init,
            bias_initializer='zeros'
        )
        
        # Build the model
        self.build((None, None, None, in_channels))
        
    def build(self, input_shape):
        """Build the model and initialize weights."""
        super().build(input_shape)
        
        # Initialize all layers
        for layer in self.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.assign(
                    layer.kernel_initializer(layer.kernel.shape)
                )
            if hasattr(layer, 'bias_initializer') and layer.use_bias:
                layer.bias.assign(
                    layer.bias_initializer(layer.bias.shape)
                )

    def build_encoder(self, kernel_initializer, bias_initializer):
        """Build encoder network."""
        layers = []
        in_channels = self.in_channels
        
        # Initial convolution
        layers.append(
            tf.keras.layers.Conv2D(
                self.block_out_channels[0],
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name="conv_in"
            )
        )
        
        # Down blocks
        output_channel = self.block_out_channels[0]
        for i, down_block_type in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = self.block_out_channels[min(i + 1, len(self.block_out_channels) - 1)]
            is_final_block = i == len(self.down_block_types) - 1
            
            down_block = []
            
            # Add layers
            for _ in range(self.layers_per_block):
                down_block.append(
                    tf.keras.layers.Conv2D(
                        output_channel,
                        kernel_size=3,
                        strides=2 if not is_final_block else 1,
                        padding="same",
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer
                    )
                )
                down_block.append(tf.keras.layers.Activation(self.act_fn))
                
            layers.extend(down_block)
            
        return tf.keras.Sequential(layers, name="encoder")
        
    def build_decoder(self, kernel_initializer, bias_initializer):
        """Build decoder network."""
        layers = []
        
        in_channels = self.latent_channels
        out_channels = self.block_out_channels[-1]
        
        # Initial convolution
        layers.append(
            tf.keras.layers.Conv2D(
                out_channels,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name="conv_in"
            )
        )
        
        # Up blocks
        for i, up_block_type in enumerate(self.up_block_types):
            input_channel = out_channels
            out_channels = self.block_out_channels[-(i + 2)]
            is_final_block = i == len(self.up_block_types) - 1
            
            up_block = []
            
            # Add layers
            for _ in range(self.layers_per_block):
                up_block.append(
                    tf.keras.layers.Conv2DTranspose(
                        out_channels,
                        kernel_size=3,
                        strides=2 if not is_final_block else 1,
                        padding="same",
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer
                    )
                )
                up_block.append(tf.keras.layers.Activation(self.act_fn))
                
            layers.extend(up_block)
            
        # Final convolution
        layers.append(
            tf.keras.layers.Conv2D(
                self.out_channels,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name="conv_out"
            )
        )
        
        return tf.keras.Sequential(layers, name="decoder")
    
    def encode(self, x: tf.Tensor, return_dict: bool = True) -> tf.Tensor:
        """Encode input tensor to latent space."""
        h = self.encoder(x)
        moments = tf.keras.layers.Conv2D(
            2 * self.latent_channels, 1, padding="same", name="quant_conv"
        )(h)
        mean, logvar = tf.split(moments, 2, axis=-1)
        
        # Sample from latent distribution
        std = tf.exp(0.5 * logvar)
        latents = mean + std * tf.random.normal(tf.shape(mean))
        latents = latents * self.scaling_factor
        
        if return_dict:
            return {
                "latent_dist": {"mean": mean, "std": std},
                "latents": latents
            }
        return latents
    
    def decode(self, z: tf.Tensor, return_dict: bool = False) -> tf.Tensor:
        """Decode latent tensor to image space."""
        z = z / self.scaling_factor
        z = tf.keras.layers.Conv2D(
            self.latent_channels, 1, padding="same", name="post_quant_conv"
        )(z)
        dec = self.decoder(z)
        
        if return_dict:
            return {"sample": dec}
        return dec
    
    def call(self, x: tf.Tensor, training: bool = True) -> Dict[str, tf.Tensor]:
        """Forward pass."""
        latents = self.encode(x)["latents"]
        dec = self.decode(latents)
        return {"sample": dec, "latents": latents}
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        **kwargs
    ) -> "AutoencoderKL":
        """Load pretrained model.
        
        Args:
            pretrained_model_path: Path to pretrained model
            **kwargs: Additional arguments to pass to __init__
            
        Returns:
            Loaded model
        """
        if not os.path.exists(pretrained_model_path):
            pretrained_model_path = snapshot_download(pretrained_model_path)
            
        # Load config
        config_path = os.path.join(pretrained_model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}
            
        # Update config with kwargs
        config.update(kwargs)
        
        # Create model
        model = cls(config=config)
        
        # Load weights
        weight_files = [f for f in os.listdir(pretrained_model_path) if f.endswith(".safetensors")]
        if not weight_files:
            raise ValueError(f"No .safetensors files found in {pretrained_model_path}")
            
        # Load and convert weights
        for weight_file in weight_files:
            weight_path = os.path.join(pretrained_model_path, weight_file)
            with safe_open(weight_path, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    tf_tensor = convert_torch_to_tf(tensor)
                    
                    # Find corresponding layer and set weights
                    layer_name = key.split(".")[0]
                    if hasattr(model, layer_name):
                        layer = getattr(model, layer_name)
                        if isinstance(layer, tf.keras.layers.Layer):
                            layer.set_weights([tf_tensor])
                            
        return model
