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
    """Modulate layer output."""
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
            -math.log(max_period) * tf.range(half, dtype=tf.float32) / half
        )
        args = tf.cast(t, tf.float32)[:, None] * freqs[None]
        embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
        if dim % 2:
            embedding = tf.pad(embedding, [[0, 0], [0, 1]])
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
        self.norm_final = layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)
        self.linear = layers.Dense(patch_size * patch_size * out_channels, use_bias=True)
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
    def __init__(self, patch_size=2, in_chans=4, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Create the Conv2D layer with correct input shape
        self.proj = None
    
    def build(self, input_shape):
        # Initialize Conv2D layer when input shape is known
        self.proj = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding='valid',
            data_format='channels_last',
            dtype=tf.float32,
            name='conv2d'
        )
        # Build the Conv2D layer
        self.proj.build(input_shape)
        self.built = True
    
    def call(self, x):
        if not self.built:
            self.build(x.shape)
        # Input is [B, H, W, C], which is already in TensorFlow's channels_last format
        # No need to transpose
        x = self.proj(x)
        # Reshape to [B, H*W, C]
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H * W, C])
        return x

class Transformer(layers.Layer):
    """Transformer block with pre-norm architecture."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
            dropout=dropout_rate
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(int(hidden_size * mlp_ratio)),
            layers.Activation('gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(hidden_size),
            layers.Dropout(dropout_rate)
        ])
        
    def call(self, x, training=False):
        # Attention
        attn_output = self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + attn_output
        
        # MLP
        mlp_output = self.mlp(self.norm2(x), training=training)
        x = x + mlp_output
        
        return x

class OmniGenTF(tf.keras.Model, PeftAdapterMixin):
    """TensorFlow implementation of OmniGen."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        self.num_layers = config.get('num_hidden_layers', 12)
        self.num_heads = config.get('num_attention_heads', 12)
        self.use_kv_cache = False
        self.offload_kv_cache = False
        
        # Create embedders first
        self.x_embedder = PatchEmbedMR(
            patch_size=2,
            in_chans=4,
            embed_dim=self.hidden_size
        )
        self.input_x_embedder = PatchEmbedMR(
            patch_size=2,
            in_chans=4,
            embed_dim=self.hidden_size
        )
        
        # Build embedders
        dummy_input = tf.random.uniform((1, 64, 64, 4))
        _ = self.x_embedder(dummy_input)
        _ = self.input_x_embedder(dummy_input)
        
        # Create other components
        self.time_token = TimestepEmbedder(self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        
        # Position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            192,  # pos_embed_max_size
            interpolation_scale=1.0,
            base_size=64
        )
        self.pos_embed = tf.constant(pos_embed, dtype=tf.float32)[None]
        
        # Final layer
        self.final_layer = FinalLayer(self.hidden_size, 2, 4)
        
        # Initialize transformer
        self.transformer_blocks = [
            Transformer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                name=f'transformer_block_{i}'
            )
            for i in range(self.num_layers)
        ]
        
        # Build transformer
        dummy_seq = tf.random.uniform((1, 256, self.hidden_size))
        for block in self.transformer_blocks:
            _ = block(dummy_seq)
        
        # Initialize weights
        self.initialize_weights()
        
        # KV cache
        self.past_key_values = None
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        adapter_path: Optional[str] = None,
        **kwargs
    ) -> "OmniGenTF":
        """Load pretrained model.
        
        Args:
            pretrained_model_path: Path to pretrained model
            adapter_path: Path to adapter weights (optional)
            **kwargs: Additional arguments
            
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
        model = cls(config)
        
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
                            
        # Load adapter if provided
        if adapter_path is not None:
            model.load_adapter(adapter_path)
            
        return model
        
    def patch_multiple_resolutions(
        self,
        latents: tf.Tensor,
        padding_latent: Optional[tf.Tensor] = None,
        is_input_images: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Process latents at multiple resolutions."""
        embedder = self.input_x_embedder if is_input_images else self.x_embedder
        
        # Get latent dimensions
        batch_size = tf.shape(latents)[0]
        height = tf.shape(latents)[1]
        width = tf.shape(latents)[2]
        
        # Embed latents
        x = embedder(latents)
        
        # Get position embeddings
        pos_embed = self.cropped_pos_embed(height, width)
        
        # Add position embeddings
        x = x + pos_embed
        
        # Reshape to sequence
        x = tf.reshape(x, [batch_size, -1, self.hidden_size])
        
        return x, pos_embed
        
    def forward_with_cfg(
        self,
        x: tf.Tensor,
        timestep: tf.Tensor,
        input_ids: Optional[tf.Tensor] = None,
        input_img_latents: Optional[tf.Tensor] = None,
        input_image_sizes: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        cfg_scale: float = 3.0,
        use_img_cfg: bool = True,
        img_cfg_scale: float = 1.6,
        past_key_values: Optional[List[Tuple[tf.Tensor]]] = None,
        use_kv_cache: bool = True,
        offload_model: bool = False
    ) -> tf.Tensor:
        """Forward pass with classifier-free guidance."""
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
            img_latents_in = input_img_latents
            img_sizes_in = input_image_sizes
            
        # Get predictions
        noise_pred = self.call(
            x_in,
            t_in,
            input_ids_in,
            img_latents_in,
            img_sizes_in,
            attention_mask_in,
            position_ids_in,
            past_key_values=past_key_values,
            use_kv_cache=use_kv_cache,
            offload_model=offload_model
        )
        
        # Split predictions
        noise_pred_uncond, noise_pred_text = tf.split(noise_pred, 2, axis=0)
        
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        
        if input_img_latents is not None and use_img_cfg:
            noise_pred_img = noise_pred_text  # Use text-conditioned prediction for image guidance
            noise_pred = noise_pred + img_cfg_scale * (noise_pred_img - noise_pred)
            
        return noise_pred
        
    def forward_with_separate_cfg(
        self,
        x: tf.Tensor,
        timestep: tf.Tensor,
        input_ids: Optional[tf.Tensor] = None,
        input_img_latents: Optional[tf.Tensor] = None,
        input_image_sizes: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        cfg_scale: float = 3.0,
        use_img_cfg: bool = True,
        img_cfg_scale: float = 1.6,
        past_key_values: Optional[List[Tuple[tf.Tensor]]] = None,
        use_kv_cache: bool = True,
        offload_model: bool = False
    ) -> tf.Tensor:
        """Forward pass with separate classifier-free guidance."""
        # Get unconditional prediction
        noise_pred_uncond = self.call(
            x,
            timestep,
            None,
            None,
            None,
            None,
            None,
            past_key_values=past_key_values,
            use_kv_cache=use_kv_cache,
            offload_model=offload_model
        )
        
        # Get text-conditioned prediction
        noise_pred_text = self.call(
            x,
            timestep,
            input_ids,
            None,
            None,
            attention_mask,
            position_ids,
            past_key_values=past_key_values,
            use_kv_cache=use_kv_cache,
            offload_model=offload_model
        )
        
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        
        if input_img_latents is not None and use_img_cfg:
            # Get image-conditioned prediction
            noise_pred_img = self.call(
                x,
                timestep,
                input_ids,
                input_img_latents,
                input_image_sizes,
                attention_mask,
                position_ids,
                past_key_values=past_key_values,
                use_kv_cache=use_kv_cache,
                offload_model=offload_model
            )
            noise_pred = noise_pred + img_cfg_scale * (noise_pred_img - noise_pred)
            
        return noise_pred
        
    def call(
        self,
        x: tf.Tensor,
        timesteps: tf.Tensor,
        input_ids: Optional[tf.Tensor] = None,
        input_img_latents: Optional[tf.Tensor] = None,
        input_image_sizes: Optional[tf.Tensor] = None,
        attention_mask: Optional[tf.Tensor] = None,
        position_ids: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[Tuple[tf.Tensor]]] = None,
        use_kv_cache: bool = True,
        offload_model: bool = False,
        training: bool = False
    ) -> tf.Tensor:
        """Model forward pass."""
        # Process input latents
        x, pos_embed = self.patch_multiple_resolutions(x)
        
        # Get timestep embeddings
        t_emb = self.t_embedder(timesteps)
        
        # Process input images if provided
        if input_img_latents is not None:
            input_x, _ = self.patch_multiple_resolutions(input_img_latents, is_input_images=True)
            x = tf.concat([x, input_x], axis=1)
            
        # Add time token
        time_token = self.time_token(timesteps)
        time_token = tf.expand_dims(time_token, axis=1)
        x = tf.concat([time_token, x], axis=1)
        
        # Apply transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if offload_model:
                block.to("GPU:0")
                
            if use_kv_cache and past_key_values is not None:
                kv_cache = past_key_values[i]
            else:
                kv_cache = None
                
            x = block(x, kv_cache=kv_cache)
            
            if offload_model:
                block.to("CPU:0")
                
            if use_kv_cache:
                if self.past_key_values is None:
                    self.past_key_values = [None] * len(self.transformer_blocks)
                self.past_key_values[i] = x
                
        # Apply final layer
        x = self.final_layer(x[:, 1:], t_emb)  # Remove time token
        
        # Reshape output
        batch_size = tf.shape(x)[0]
        sequence_length = tf.shape(x)[1]
        x = tf.reshape(x, [batch_size, sequence_length, 2, 2, 4])
        x = tf.transpose(x, [0, 1, 3, 2, 4])
        x = tf.reshape(x, [batch_size, -1, 4])
        
        return x

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
