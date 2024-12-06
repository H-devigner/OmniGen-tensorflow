"""VAE model for OmniGen."""
import tensorflow as tf
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
        scaling_factor: float = 0.18215,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize VAE model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            down_block_types: Types of down blocks
            up_block_types: Types of up blocks
            block_out_channels: Number of output channels for each block
            layers_per_block: Number of layers per block
            act_fn: Activation function
            latent_channels: Number of latent channels
            scaling_factor: Scaling factor for latents
            config: Model configuration
        """
        super().__init__()
        
        self.config = config or {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "block_out_channels": block_out_channels,
            "layers_per_block": layers_per_block,
            "act_fn": act_fn,
            "latent_channels": latent_channels,
            "scaling_factor": scaling_factor
        }
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Build bottleneck
        self.quant_conv = tf.keras.layers.Conv2D(
            2 * latent_channels, 1, padding="same", name="quant_conv"
        )
        self.post_quant_conv = tf.keras.layers.Conv2D(
            latent_channels, 1, padding="same", name="post_quant_conv"
        )
        
        self.scaling_factor = scaling_factor
        
    def _build_encoder(self) -> tf.keras.Sequential:
        """Build encoder network."""
        layers = []
        in_channels = self.config["in_channels"]
        
        # Initial convolution
        layers.append(
            tf.keras.layers.Conv2D(
                self.config["block_out_channels"][0],
                kernel_size=3,
                strides=1,
                padding="same",
                name="conv_in"
            )
        )
        
        # Down blocks
        output_channel = self.config["block_out_channels"][0]
        for i, down_block_type in enumerate(self.config["down_block_types"]):
            input_channel = output_channel
            output_channel = self.config["block_out_channels"][min(i + 1, len(self.config["block_out_channels"]) - 1)]
            is_final_block = i == len(self.config["down_block_types"]) - 1
            
            down_block = []
            
            # Add layers
            for _ in range(self.config["layers_per_block"]):
                down_block.append(
                    tf.keras.layers.Conv2D(
                        output_channel,
                        kernel_size=3,
                        strides=2 if not is_final_block else 1,
                        padding="same"
                    )
                )
                down_block.append(tf.keras.layers.Activation(self.config["act_fn"]))
                
            layers.extend(down_block)
            
        return tf.keras.Sequential(layers, name="encoder")
        
    def _build_decoder(self) -> tf.keras.Sequential:
        """Build decoder network."""
        layers = []
        
        in_channels = self.config["latent_channels"]
        out_channels = self.config["block_out_channels"][-1]
        
        # Initial convolution
        layers.append(
            tf.keras.layers.Conv2D(
                out_channels,
                kernel_size=3,
                strides=1,
                padding="same",
                name="conv_in"
            )
        )
        
        # Up blocks
        for i, up_block_type in enumerate(self.config["up_block_types"]):
            input_channel = out_channels
            out_channels = self.config["block_out_channels"][-(i + 2)]
            is_final_block = i == len(self.config["up_block_types"]) - 1
            
            up_block = []
            
            # Add layers
            for _ in range(self.config["layers_per_block"]):
                up_block.append(
                    tf.keras.layers.Conv2DTranspose(
                        out_channels,
                        kernel_size=3,
                        strides=2 if not is_final_block else 1,
                        padding="same"
                    )
                )
                up_block.append(tf.keras.layers.Activation(self.config["act_fn"]))
                
            layers.extend(up_block)
            
        # Final convolution
        layers.append(
            tf.keras.layers.Conv2D(
                self.config["out_channels"],
                kernel_size=3,
                strides=1,
                padding="same",
                name="conv_out"
            )
        )
        
        return tf.keras.Sequential(layers, name="decoder")
    
    def encode(self, x: tf.Tensor, return_dict: bool = True) -> tf.Tensor:
        """Encode input tensor to latent space."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
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
        z = self.post_quant_conv(z)
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
