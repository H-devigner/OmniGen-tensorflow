"""OmniGen pipeline for text-to-image generation."""
import os
import tensorflow as tf
from PIL import Image
import numpy as np
from typing import List, Union, Optional
import time
from huggingface_hub import snapshot_download
from safetensors import safe_open
import logging
import inspect

from .model import OmniGenTF
from .scheduler import OmniGenScheduler
from .vae import AutoencoderKL
from .utils import convert_torch_to_tf

logger = logging.getLogger(__name__)

class OmniGenPipeline:
    """Pipeline for text-to-image generation with OmniGen."""
    
    def __init__(
        self,
        model: OmniGenTF,
        vae: AutoencoderKL,
        scheduler: OmniGenScheduler,
        processor=None,
        device: str = None
    ):
        """Initialize OmniGen pipeline.
        
        Args:
            model: OmniGen model
            vae: VAE model for encoding/decoding images
            scheduler: Noise scheduler
            processor: Text processor (optional)
            device: Device to run on (optional)
        """
        self.model = model
        self.vae = vae
        self.scheduler = scheduler
        self.processor = processor
        
        # Set device
        self.device = device or ("GPU:0" if tf.config.list_physical_devices('GPU') else "CPU:0")
        
        # Memory optimization flags
        self.model_cpu_offload = False
        self.vae_cpu_offload = False
        self.use_kv_cache = True
        self.offload_kv_cache = False
        
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        vae_path: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        mixed_precision: bool = True,
        **kwargs
    ) -> "OmniGenPipeline":
        """Initialize pipeline with memory optimization."""
        
        # Enable mixed precision for memory efficiency
        if mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        # Set memory growth to avoid OOM
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Memory growth setting failed: {e}")
        
        for attempt in range(max_retries):
            try:
                # Download and load model with optimized settings
                model_path = snapshot_download(model_name)
                model = OmniGenTF.from_pretrained(
                    model_path,
                    use_mixed_precision=mixed_precision,
                    **kwargs
                )
                
                # Load VAE with optimization
                if vae_path is None:
                    vae_path = os.path.join(model_path, "vae")
                    if os.path.exists(vae_path):
                        vae = AutoencoderKL.from_pretrained(
                            vae_path,
                            use_mixed_precision=mixed_precision
                        )
                    else:
                        print("No VAE found in model, using default")
                        vae = AutoencoderKL.from_pretrained(
                            "stabilityai/sdxl-vae",
                            use_mixed_precision=mixed_precision
                        )
                else:
                    vae = AutoencoderKL.from_pretrained(
                        vae_path,
                        use_mixed_precision=mixed_precision
                    )
                
                # Initialize processor
                processor = OmniGenTFProcessor.from_pretrained(model_path)
                
                return cls(
                    model=model,
                    vae=vae,
                    processor=processor,
                    **kwargs
                )
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
    
    def enable_model_cpu_offload(self):
        """Enable CPU offloading for model."""
        if not self.model_cpu_offload:
            self.model.to("CPU:0")
            self.model_cpu_offload = True
    
    def disable_model_cpu_offload(self):
        """Disable CPU offloading for model."""
        if self.model_cpu_offload:
            self.model.to(self.device)
            self.model_cpu_offload = False
    
    def enable_vae_cpu_offload(self):
        """Enable CPU offloading for VAE."""
        if not self.vae_cpu_offload:
            self.vae.to("CPU:0")
            self.vae_cpu_offload = True
    
    def disable_vae_cpu_offload(self):
        """Disable CPU offloading for VAE."""
        if self.vae_cpu_offload:
            self.vae.to(self.device)
            self.vae_cpu_offload = False
    
    def to(self, device: str):
        """Move pipeline to specified device."""
        self.device = device
        if not self.model_cpu_offload:
            self.model.to(device)
        if not self.vae_cpu_offload:
            self.vae.to(device)
    
    def vae_encode(self, x: tf.Tensor) -> tf.Tensor:
        """Encode images using VAE."""
        if self.vae_cpu_offload:
            self.vae.to(self.device)
        
        # Encode
        latents = self.vae.encode(x)
        
        if self.vae_cpu_offload:
            self.vae.to("CPU:0")
            
        return latents
    
    def vae_decode(self, latents: tf.Tensor) -> tf.Tensor:
        """Decode latents using VAE."""
        if self.vae_cpu_offload:
            self.vae.to(self.device)
            
        # Decode
        images = self.vae.decode(latents)
        
        if self.vae_cpu_offload:
            self.vae.to("CPU:0")
            
        return images
    
    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: tf.DType = tf.float32,
        seed: Optional[int] = None
    ) -> tf.Tensor:
        """Prepare random latents for generation."""
        # Calculate latent size
        vae_scale = self.vae.config.scaling_factor
        latents_height = height // vae_scale
        latents_width = width // vae_scale
        
        # Generate random latents
        shape = (batch_size, latents_height, latents_width, 4)
        if seed is not None:
            tf.random.set_seed(seed)
        latents = tf.random.normal(shape, dtype=dtype)
        
        # Scale latents
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[tf.random.Generator] = None,
        latents: Optional[tf.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[callable] = None,
        callback_steps: int = 1,
        **kwargs
    ):
        """Generate images from text prompt."""
        
        # 0. Process inputs
        device = self.device
        batch_size = 1
        if isinstance(prompt, list):
            batch_size = len(prompt)
        
        height = height or 1024
        width = width or 1024
        
        # 1. Process text
        text_inputs = self.processor(
            prompt,
            padding=True,
            max_length=77,
            truncation=True,
            return_tensors="tf"
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        position_ids = tf.range(0, tf.shape(input_ids)[1])[None].repeat(batch_size, axis=0)
        
        # 2. Define timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # 3. Prepare latents
        latents_shape = (batch_size, height//8, width//8, 4)
        if latents is None:
            latents = tf.random.normal(
                latents_shape,
                seed=generator.seed() if generator else None,
                dtype=tf.float32
            )
        
        # 4. Prepare extra parameters
        extra_step_kwargs = {}
        if "eta" in inspect.signature(self.scheduler.step).parameters:
            extra_step_kwargs["eta"] = eta
            
        # 5. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        past_key_values = None
        
        with tf.device(device):
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                latent_model_input = tf.concat([latents] * 2, axis=0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise residual
                if self.use_kv_cache and past_key_values is not None:
                    noise_pred, past_key_values = self.model.forward_with_cfg(
                        latent_model_input,
                        tf.repeat(t, batch_size),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        cfg_scale=guidance_scale,
                        use_img_cfg=use_img_guidance,
                        img_cfg_scale=img_guidance_scale,
                        past_key_values=past_key_values,
                        use_kv_cache=True,
                        offload_model=self.model_cpu_offload
                    )
                else:
                    noise_pred = self.model.forward_with_cfg(
                        latent_model_input,
                        tf.repeat(t, batch_size),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        cfg_scale=guidance_scale,
                        use_img_cfg=use_img_guidance,
                        img_cfg_scale=img_guidance_scale,
                        past_key_values=None,
                        use_kv_cache=False,
                        offload_model=self.model_cpu_offload
                    )
                
                # Compute previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # Call callback if needed
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
        
        # 6. Post-processing
        image = self.vae.decode(latents / 0.18215).sample
        image = (image / 2 + 0.5).clip(0, 1)
        
        # 7. Convert to output format
        image = tf.transpose(image, [0, 2, 3, 1])
        
        if output_type == "pil":
            image = tf.cast(image * 255, tf.uint8)
            image = [Image.fromarray(img.numpy()) for img in image]
            
        return image
