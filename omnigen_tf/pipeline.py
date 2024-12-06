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
        input_images: Optional[Union[List[str], List[List[str]]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        offload_model: bool = False,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = False,
        use_input_image_size_as_output: bool = False,
        seed: Optional[int] = None,
        output_type: str = "pil",
    ) -> List[Image.Image]:
        """Generate images from text prompt.
        
        Args:
            prompt: Text prompt(s)
            input_images: Optional input reference images
            height: Output height
            width: Output width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            use_img_guidance: Whether to use image guidance
            img_guidance_scale: Image guidance scale
            max_input_image_size: Maximum size for input images
            separate_cfg_infer: Whether to run CFG inference separately
            offload_model: Whether to offload model to CPU
            use_kv_cache: Whether to use KV caching
            offload_kv_cache: Whether to offload KV cache
            use_input_image_size_as_output: Use input image size as output
            seed: Random seed
            output_type: Output format ("pil" or "tensor")
            
        Returns:
            List of generated images
        """
        # Input validation
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("Height and width must be multiples of 8")
            
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        
        # Process input images if provided
        input_latents = None
        if input_images is not None:
            if isinstance(input_images[0], str):
                input_images = [input_images]
            input_latents = []
            for img_list in input_images:
                img_latents = []
                for img_path in img_list:
                    img = Image.open(img_path).convert("RGB")
                    img = tf.convert_to_tensor(np.array(img))
                    img = tf.image.resize(img, (max_input_image_size, max_input_image_size))
                    img = (img / 127.5) - 1.0
                    latents = self.vae_encode(img[None])
                    img_latents.append(latents)
                input_latents.append(img_latents)
        
        # Set up memory optimization
        if offload_model:
            self.enable_model_cpu_offload()
        self.use_kv_cache = use_kv_cache
        self.offload_kv_cache = offload_kv_cache
        
        # Prepare initial latents
        latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            seed=seed
        )
        
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Generation loop
        for t in timesteps:
            # Move model to device if needed
            if self.model_cpu_offload:
                self.model.to(self.device)
                
            # Prepare model inputs
            model_input = {
                "sample": latents,
                "timestep": tf.fill((batch_size,), t),
                "encoder_hidden_states": None,  # Add text embeddings here
                "input_image_latents": input_latents
            }
            
            # Model inference
            if separate_cfg_infer:
                noise_pred = self.model.forward_with_separate_cfg(
                    **model_input,
                    guidance_scale=guidance_scale,
                    use_img_guidance=use_img_guidance,
                    img_guidance_scale=img_guidance_scale
                )
            else:
                noise_pred = self.model.forward_with_cfg(
                    **model_input,
                    guidance_scale=guidance_scale,
                    use_img_guidance=use_img_guidance,
                    img_guidance_scale=img_guidance_scale
                )
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents)
            
            # Offload model if needed
            if self.model_cpu_offload:
                self.model.to("CPU:0")
        
        # Decode latents
        images = self.vae_decode(latents)
        images = ((images + 1) * 127.5).numpy().astype(np.uint8)
        
        # Convert to PIL
        if output_type == "pil":
            images = [Image.fromarray(img) for img in images]
            
        return images
