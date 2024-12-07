"""Phi3 Transformer implementation in TensorFlow."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, initializers
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple

class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.num_attention_heads = config.get('num_attention_heads', 32)
        self.hidden_size = config.get('hidden_size', 2048)
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        
        # Initialize with truncated normal
        kernel_init = initializers.TruncatedNormal(stddev=0.02)
        
        self.query = layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer='zeros'
        )
        self.key = layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer='zeros'
        )
        self.value = layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer='zeros'
        )
        self.out = layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=kernel_init,
            bias_initializer='zeros'
        )
        
    def build(self, input_shape):
        """Build the layer."""
        super().build(input_shape)
        
        # Initialize weights
        for layer in [self.query, self.key, self.value, self.out]:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.assign(
                    layer.kernel_initializer(layer.kernel.shape)
                )
            if hasattr(layer, 'bias_initializer') and layer.use_bias:
                layer.bias.assign(
                    layer.bias_initializer(layer.bias.shape)
                )

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_length, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, [0, 2, 1, 3])
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[Tuple[tf.Tensor, ...], tf.Tensor]:
        batch_size = tf.shape(hidden_states)[0]
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        if past_key_value is not None:
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)
            
        if use_cache:
            present = (key_layer, value_layer)
        
        # Compute attention scores
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(float(self.attention_head_size))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        context_layer = tf.matmul(attention_probs, value_layer)
        
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.hidden_size))
        
        attention_output = self.out(context_layer)
        
        if use_cache:
            return attention_output, present
        return attention_output

class Phi3Block(layers.Layer):
    """Transformer block for Phi3."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = tf.keras.Sequential([
            layers.Dense(config.get('intermediate_size', 8192), activation='gelu', kernel_initializer=initializers.TruncatedNormal(stddev=0.02), bias_initializer='zeros'),
            layers.Dense(config.get('hidden_size', 2048), kernel_initializer=initializers.TruncatedNormal(stddev=0.02), bias_initializer='zeros')
        ])
        self.input_layernorm = layers.LayerNormalization(epsilon=1e-5)
        self.post_attention_layernorm = layers.LayerNormalization(epsilon=1e-5)
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        past_key_value: Optional[Tuple[tf.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[Tuple[tf.Tensor, ...], tf.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if use_cache:
            self_attn_output, present = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=True
            )
        else:
            self_attn_output = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=False
            )
            
        hidden_states = residual + self_attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        if use_cache:
            return hidden_states, present
        return hidden_states

class Phi3TransformerTF(layers.Layer):
    """Phi3 Transformer model."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 2048)
        self.num_hidden_layers = config.get('num_hidden_layers', 32)
        
        # Transformer blocks
        self.blocks = [Phi3Block(config) for _ in range(self.num_hidden_layers)]
        
        # Final layer norm
        self.final_layernorm = layers.LayerNormalization(epsilon=1e-5)
        
        # Cache for key/value states
        self.use_cache = False
        
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
        past_key_values: Optional[List[Tuple[tf.Tensor]]] = None,
        use_cache: Optional[bool] = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, List[Tuple[tf.Tensor]]]]:
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        if past_key_values is None:
            past_key_values = [None] * self.num_hidden_layers
            
        all_presents = [] if use_cache else None
        
        for i, (block, past_key_value) in enumerate(zip(self.blocks, past_key_values)):
            if use_cache:
                hidden_states, present = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=True
                )
                all_presents.append(present)
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=False
                )
                
        hidden_states = self.final_layernorm(hidden_states)
        
        if use_cache:
            return hidden_states, all_presents
        return hidden_states
