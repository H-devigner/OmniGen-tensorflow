"""OmniGen model implementation in TensorFlow."""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import math
from typing import Optional, Dict, Any, Union, List, Tuple
from safetensors import safe_open
from huggingface_hub import snapshot_download

from .converter import WeightConverter
from .peft import PeftAdapterMixin
import os
import json

def modulate(x, shift, scale):
    """Modulate layer norm output."""
    return x * (1 + tf.expand_dims(scale, 1)) + tf.expand_dims(shift, 1)

class TimestepEmbedder(layers.Layer):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_size, use_bias=True),
            layers.Activation('swish'),  # SiLU/Swish activation
            layers.Dense(hidden_size, use_bias=True)
        ])

    def timestep_embedding(self, t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = tf.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.cast(t[:, None], tf.float32) * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def call(self, t, dtype=tf.float32):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = tf.cast(t_freq, dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class FinalLayer(layers.Layer):
    """The final layer of OmniGen."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = layers.LayerNormalization(
            epsilon=1e-6,
            center=False,
            scale=False
        )
        self.linear = layers.Dense(
            patch_size * patch_size * out_channels,
            use_bias=True
        )
        self.adaLN_modulation = tf.keras.Sequential([
            layers.Activation('swish'),
            layers.Dense(2 * hidden_size, use_bias=True)
        ])

    def call(self, x, c):
        shift, scale = tf.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class PatchEmbedMR(layers.Layer):
    """2D Image to Patch Embedding."""
    def __init__(self, patch_size=2, in_chans=4, embed_dim=768, bias=True):
        super().__init__()
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=bias,
            data_format='channels_last'
        )

    def call(self, x):
        # Input: [B, H, W, C]
        x = self.proj(x)
        # Reshape to [B, H*W, C]
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B, H * W, C])
        return x

class OmniGenTF(tf.keras.Model):
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
        
        # Create embedders
        self.x_embedder = PatchEmbedMR(
            self.patch_size, 
            self.in_channels,
            self.hidden_size,
            bias=True
        )
        self.input_x_embedder = PatchEmbedMR(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bias=True
        )
        
        # Time embeddings
        self.time_token = TimestepEmbedder(self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        
        # Position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            self.pos_embed_max_size,
            interpolation_scale=self.pe_interpolation,
            base_size=64
        )
        self.pos_embed = tf.constant(pos_embed, dtype=tf.float32)[None]
        
        # Final layer
        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels
        )
        
        # Initialize Phi3 transformer
        self.llm = Phi3TransformerTF(config)
        self.llm.use_cache = False
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize weights following PyTorch implementation."""
        def _basic_init(layer):
            if isinstance(layer, layers.Dense):
                # Use Glorot (Xavier) uniform initialization
                layer.kernel_initializer = 'glorot_uniform'
                if layer.use_bias:
                    layer.bias_initializer = 'zeros'
            elif isinstance(layer, layers.Conv2D):
                layer.kernel_initializer = 'glorot_uniform'
                if layer.use_bias:
                    layer.bias_initializer = 'zeros'
                    
        self.apply(_basic_init)
        
    def unpatchify(self, x, h, w):
        """Reverse patch embedding."""
        # x: [B, L, patch_size**2 * C]
        # Return: [B, H, W, C]
        patch_size = self.patch_size
        c = self.out_channels
        
        B, L, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, [B, h//patch_size, w//patch_size, patch_size, patch_size, c])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, h, w, c])
        return x
        
    def patch_multiple_resolutions(self, latents, padding_latent=None, is_input_images=False):
        """Process latents at multiple resolutions."""
        embedder = self.input_x_embedder if is_input_images else self.x_embedder
        
        # Get dimensions
        B, H, W, C = tf.shape(latents)[0], tf.shape(latents)[1], tf.shape(latents)[2], tf.shape(latents)[3]
        
        # Embed latents
        x = embedder(latents)
        
        # Get position embeddings for this resolution
        pos_embed = self.pos_embed[:, :H*W//4]  # Divide by 4 due to patch size of 2
        
        return x, pos_embed

    def call(
        self,
        x,
        timestep,
        input_ids=None,
        input_img_latents=None,
        input_image_sizes=None,
        attention_mask=None,
        position_ids=None,
        padding_latent=None,
        past_key_values=None,
        return_past_key_values=True,
        offload_model=False
    ):
        """Forward pass matching PyTorch implementation."""
        # Process input through patch embedding
        x, pos_embed = self.patch_multiple_resolutions(x, padding_latent)
        x = x + tf.cast(pos_embed, x.dtype)
        
        # Get timestep embedding
        t_emb = self.t_embedder(timestep)
        
        # Process input images if provided
        if input_img_latents is not None:
            input_x, _ = self.patch_multiple_resolutions(
                input_img_latents,
                padding_latent,
                is_input_images=True
            )
            x = tf.concat([x, input_x], axis=1)
        
        # Add time token
        time_token = self.time_token(timestep)
        time_token = tf.expand_dims(time_token, axis=1)
        x = tf.concat([time_token, x], axis=1)
        
        # Process through transformer
        hidden_states = self.llm(
            input_ids=input_ids,
            inputs_embeds=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=return_past_key_values,
            training=False
        )
        
        if return_past_key_values:
            x = hidden_states[0]
            past_key_values = hidden_states[1]
        else:
            x = hidden_states[0]
            past_key_values = None
            
        # Final layer processing
        x = self.final_layer(x[:, 1:], t_emb)  # Remove time token
        
        if return_past_key_values:
            return x, past_key_values
        return x
        
    def forward_with_cfg(
        self,
        x,
        timestep,
        input_ids=None,
        input_img_latents=None,
        input_image_sizes=None,
        attention_mask=None,
        position_ids=None,
        cfg_scale=3.0,
        use_img_cfg=True,
        img_cfg_scale=1.6,
        past_key_values=None,
        use_kv_cache=True,
        offload_model=False
    ):
        """Forward with classifier-free guidance."""
        # Double the batch for CFG
        x_in = tf.concat([x] * 2, axis=0)
        t_in = tf.concat([timestep] * 2, axis=0)
        
        if input_ids is not None:
            input_ids_in = tf.concat([input_ids, tf.zeros_like(input_ids)], axis=0)
            attention_mask_in = tf.concat([attention_mask, tf.zeros_like(attention_mask)], axis=0) if attention_mask is not None else None
            position_ids_in = tf.concat([position_ids, tf.zeros_like(position_ids)], axis=0) if position_ids is not None else None
        else:
            input_ids_in = None
            attention_mask_in = None
            position_ids_in = None
            
        if input_img_latents is not None and use_img_cfg:
            img_latents_in = tf.concat([input_img_latents, tf.zeros_like(input_img_latents)], axis=0)
            img_sizes_in = tf.concat([input_image_sizes, tf.zeros_like(input_image_sizes)], axis=0) if input_image_sizes is not None else None
        else:
            img_latents_in = None
            img_sizes_in = None
            
        # Get predictions
        if use_kv_cache and past_key_values is not None:
            noise_pred, past_key_values = self(
                x_in, t_in,
                input_ids_in,
                img_latents_in,
                img_sizes_in,
                attention_mask_in,
                position_ids_in,
                past_key_values=past_key_values,
                return_past_key_values=True,
                offload_model=offload_model
            )
        else:
            noise_pred = self(
                x_in, t_in,
                input_ids_in,
                img_latents_in,
                img_sizes_in,
                attention_mask_in,
                position_ids_in,
                past_key_values=None,
                return_past_key_values=False,
                offload_model=offload_model
            )
            
        # Split predictions
        noise_pred_uncond, noise_pred_text = tf.split(noise_pred, 2, axis=0)
        
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        
        if input_img_latents is not None and use_img_cfg:
            noise_pred_img = noise_pred_text  # Use text-conditioned prediction for image guidance
            noise_pred = noise_pred + img_cfg_scale * (noise_pred_img - noise_pred)
            
        if use_kv_cache and past_key_values is not None:
            return noise_pred, past_key_values
        return noise_pred

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
