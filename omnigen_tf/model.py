"""OmniGen model implementation in TensorFlow."""
from __future__ import annotations

# Standard library imports
import os
import json
import math
from typing import Optional, Dict, Any, Union, List, Tuple, Callable, TypeVar, TYPE_CHECKING

# Third-party imports
import tensorflow as tf
from tensorflow.keras import layers, initializers
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download

# Local imports
from .converter import WeightConverter
from .peft import PeftAdapterMixin
from .transformer import Phi3TransformerTF

# Type definitions
TensorType = TypeVar('TensorType', bound=tf.Tensor)

def modulate(x, shift, scale):
    """Modulate layer norm output."""
    return x * (1 + tf.expand_dims(scale, 1)) + tf.expand_dims(shift, 1)

class TimestepEmbedder(layers.Layer):
    """Embeds scalar timesteps into vector representations."""
    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        kernel_initializer=None,
        bias_initializer=None
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        
        # Initialize with proper initializers
        self.mlp = tf.keras.Sequential([
            layers.Dense(
                hidden_size,
                use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='fc1'
            ),
            layers.Activation('swish'),
            layers.Dense(
                hidden_size,
                use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='fc2'
            )
        ])
        
    def call(self, t: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        # Create sinusoidal embeddings
        half_dim = self.frequency_embedding_size // 2
        freqs = tf.math.exp(
            -math.log(10000.0) * tf.range(0, half_dim, dtype=tf.float32) / half_dim
        )
        args = tf.cast(t, tf.float32)[:, None] * freqs[None]
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], axis=-1)
        
        # Project to hidden size
        return self.mlp(embedding)

class FinalLayer(layers.Layer):
    """The final layer of OmniGen."""
    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        kernel_initializer=None,
        bias_initializer=None
    ):
        super().__init__()
        self.norm_final = layers.LayerNormalization(
            epsilon=1e-6,
            center=False,
            scale=False,
            name='norm_final'
        )
        self.linear = layers.Dense(
            patch_size * patch_size * out_channels,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name='linear'
        )
        self.adaLN_modulation = tf.keras.Sequential([
            layers.Activation('swish'),
            layers.Dense(
                2 * hidden_size,
                use_bias=True,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name='adaLN_modulation'
            )
        ])
        
    def call(self, x: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        shift, scale = tf.split(self.adaLN_modulation(c), 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class PatchEmbedMR(layers.Layer):
    """2D Image to Patch Embedding."""
    def __init__(
        self,
        patch_size=2,
        in_chans=4,
        embed_dim=768,
        bias=True,
        kernel_initializer=None,
        bias_initializer=None
    ):
        super().__init__()
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            data_format='channels_last',
            name='proj'
        )
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass."""
        B, H, W, C = tf.shape(x)
        x = self.proj(x)
        Hp, Wp = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, [B, Hp * Wp, -1])
        return x

class OmniGenTF(tf.keras.Model, PeftAdapterMixin):
    """OmniGen model in TensorFlow."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__()
        
        # Model configuration
        self.hidden_size = config.get('hidden_size', 768)
        self.patch_size = 2
        self.in_channels = 4
        self.out_channels = 4
        self.pos_embed_max_size = 192
        self.pe_interpolation = 1.0
        
        # Create embedders with proper initialization
        kernel_init = initializers.TruncatedNormal(stddev=0.02)
        
        self.x_embedder = PatchEmbedMR(
            self.patch_size, 
            self.in_channels,
            self.hidden_size,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='x_embedder'
        )
        
        self.input_x_embedder = PatchEmbedMR(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='input_x_embedder'
        )
        
        # Time embeddings
        self.time_token = TimestepEmbedder(
            self.hidden_size,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='time_token'
        )
        
        self.t_embedder = TimestepEmbedder(
            self.hidden_size,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='t_embedder'
        )
        
        # Position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            self.pos_embed_max_size,
            interpolation_scale=self.pe_interpolation,
            base_size=64
        )
        self.pos_embed = tf.Variable(
            initial_value=pos_embed[None],
            trainable=False,
            dtype=tf.float32,
            name='pos_embed'
        )
        
        # Final layer
        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            kernel_initializer=kernel_init,
            bias_initializer='zeros',
            name='final_layer'
        )
        
        # Initialize Phi3 transformer
        self.llm = Phi3TransformerTF(config)
        self.llm.use_cache = False
        
    def call(
        self,
        x: tf.Tensor,
        timestep: tf.Tensor,
        input_ids: Optional[tf.Tensor] = None,
        input_img_latents: Optional[tf.Tensor] = None,
        input_image_sizes: Optional[List[Tuple[int, int]]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        padding_latent: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[Tuple[tf.Tensor]]] = None,
        return_past_key_values: bool = True,
        offload_model: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, List[Tuple[tf.Tensor]]]]:
        """Forward pass."""
        # Get batch size and device
        batch_size = tf.shape(x)[0]
        
        # Process input through patch embedding
        x = self.patch_multiple_resolutions(x, padding_latent)
        
        # Get time embeddings
        t = self.t_embedder(timestep)
        
        # Get position embeddings
        pos_embed = tf.repeat(self.pos_embed, batch_size, axis=0)
        
        # Combine embeddings
        h = x + pos_embed
        
        # Add time token
        time_token = self.time_token(timestep)
        h = tf.concat([time_token[:, None, :], h], axis=1)
        
        # Process through transformer
        if return_past_key_values:
            h, present_key_values = self.llm(
                h,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
        else:
            h = self.llm(
                h,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=False
            )
            present_key_values = None
            
        # Remove time token
        h = h[:, 1:]
        
        # Get output size
        H = W = int(math.sqrt(tf.shape(h)[1]))
        
        # Final layer
        h = self.final_layer(h, t)
        
        # Reshape output
        h = tf.reshape(h, [-1, H, W, self.patch_size, self.patch_size, self.out_channels])
        h = tf.transpose(h, [0, 1, 3, 2, 4, 5])
        h = tf.reshape(h, [-1, H * self.patch_size, W * self.patch_size, self.out_channels])
        
        if return_past_key_values:
            return h, present_key_values
        return h
        
    def forward_with_cfg(
        self,
        x: tf.Tensor,
        timestep: tf.Tensor,
        input_ids: Optional[tf.Tensor] = None,
        input_img_latents: Optional[tf.Tensor] = None,
        input_image_sizes: Optional[List[Tuple[int, int]]] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        cfg_scale: float = 3.0,
        use_img_cfg: bool = True,
        img_cfg_scale: float = 1.6,
        past_key_values: Optional[List[Tuple[tf.Tensor]]] = None,
        use_kv_cache: bool = True,
        offload_model: bool = False
    ) -> tf.Tensor:
        """Forward with classifier-free guidance."""
        # Get batch size
        half = tf.shape(x)[0] // 2
        
        # Split input for classifier-free guidance
        x_cond, x_uncond = tf.split(x, 2)
        
        # Forward pass for conditional
        if use_kv_cache:
            out_cond, present_key_values = self.call(
                x_cond,
                timestep,
                input_ids=input_ids,
                input_img_latents=input_img_latents,
                input_image_sizes=input_image_sizes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                return_past_key_values=True,
                offload_model=offload_model
            )
        else:
            out_cond = self.call(
                x_cond,
                timestep,
                input_ids=input_ids,
                input_img_latents=input_img_latents,
                input_image_sizes=input_image_sizes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                return_past_key_values=False,
                offload_model=offload_model
            )
            present_key_values = None
            
        # Forward pass for unconditional
        out_uncond = self.call(
            x_uncond,
            timestep,
            input_ids=None,
            input_img_latents=None,
            input_image_sizes=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            return_past_key_values=False,
            offload_model=offload_model
        )
        
        # Apply classifier-free guidance
        out = out_uncond + cfg_scale * (out_cond - out_uncond)
        
        if use_kv_cache:
            return out, present_key_values
        return out
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        use_mixed_precision: bool = False,
        **kwargs
    ) -> "OmniGenTF":
        """Load pretrained model."""
        # Download model if needed
        if not os.path.isdir(pretrained_model_path):
            pretrained_model_path = snapshot_download(
                pretrained_model_path,
                allow_patterns=["*.safetensors", "*.json", "*.bin"]
            )
            
        # Load config
        config_file = os.path.join(pretrained_model_path, "config.json")
        with open(config_file) as f:
            config = json.load(f)
            
        # Update config with kwargs
        config.update(kwargs)
        
        # Create model
        model = cls(config)
        
        # Set mixed precision if requested
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
        # Load weights
        weight_files = [f for f in os.listdir(pretrained_model_path) if f.endswith('.safetensors')]
        if not weight_files:
            raise ValueError(f"No safetensors weights found in {pretrained_model_path}")
            
        weight_file = os.path.join(pretrained_model_path, weight_files[0])
        converter = WeightConverter()
        
        with safe_open(weight_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                tf_tensor = converter.convert(tensor, key)
                
                # Find corresponding layer and weight
                layer_name = key.split('.')[0]
                layer = model.get_layer(layer_name)
                
                if isinstance(layer, tf.keras.layers.Layer):
                    if key.endswith('.weight'):
                        layer.kernel.assign(tf_tensor)
                    elif key.endswith('.bias'):
                        layer.bias.assign(tf_tensor)
                        
        return model

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=1):
    """Generate 2D sinusoidal positional embeddings."""
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate 2D positional embeddings from a grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D positional embeddings."""
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
