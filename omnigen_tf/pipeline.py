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
from tqdm.auto import tqdm

from .model import OmniGenTF
from .processor import OmniGenProcessor
from .scheduler import OmniGenScheduler
from .vae import AutoencoderKL
from .utils import convert_torch_to_tf

logger = logging.getLogger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from omnigen_tf import OmniGenPipeline
        >>> pipe = OmniGenPipeline.from_pretrained("Shitao/omnigen-v1")
        >>> prompt = "A woman holds a bouquet of flowers and faces the camera"
        >>> image = pipe(
        ...     prompt,
        ...     guidance_scale=2.5,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("t2i.png")
        ```
"""

class OmniGenPipeline:
    """Pipeline for text-to-image generation with OmniGen."""
    
    def __init__(
        self,
        model: OmniGenTF,
        vae: AutoencoderKL,
        processor: OmniGenProcessor,
        device: str = None
    ):
        """Initialize OmniGen pipeline.
        
        Args:
            model: OmniGen model
            vae: VAE model for encoding/decoding images
            processor: Text and image processor
            device: Device to run on (optional)
        """
        self.model = model
        self.vae = vae
        self.processor = processor
        
        # Set device
        self.device = device or ("GPU:0" if tf.config.list_physical_devices('GPU') else "CPU:0")
        
        # Memory optimization flags
        self.use_kv_cache = True
        
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        vae_path: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 1,
        mixed_precision: bool = True,
        **kwargs
    ):
        """Load pipeline from pretrained model.
        
        Args:
            model_name: Name or path of pretrained model
            vae_path: Path to VAE model (optional)
            max_retries: Maximum number of retries for model loading
            retry_delay: Delay between retries in seconds
            mixed_precision: Whether to use mixed precision
            **kwargs: Additional arguments to pass to model
            
        Returns:
            OmniGenPipeline: Loaded pipeline
        """
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
                vae = AutoencoderKL.from_pretrained(
                    vae_path,
                    use_mixed_precision=mixed_precision
                )
                
                # Load processor
                processor = OmniGenProcessor.from_pretrained(model_path)
                
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
                
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[tf.random.Generator] = None,
        latents: Optional[tf.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, tf.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        use_img_guidance: bool = False,
        img_guidance_scale: float = 1.6,
        guidance_image: Optional[Union[Image.Image, List[Image.Image]]] = None,
    ):
        """Generate images from text prompt.
        
        Args:
            prompt: The prompt to generate images from
            height: Height of output image
            width: Width of output image
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt for guidance
            num_images_per_prompt: Number of images to generate per prompt
            eta: Eta parameter for scheduler
            generator: Random number generator
            latents: Pre-generated latents
            output_type: Output format ('pil' or 'np')
            return_dict: Whether to return dict
            callback: Progress callback
            callback_steps: Steps between callbacks
            cross_attention_kwargs: Cross attention arguments
            use_img_guidance: Whether to use image guidance
            img_guidance_scale: Image guidance scale
            guidance_image: Reference image for guidance
            
        Returns:
            Generated images
        """
        device = self.device
        
        # Default height and width
        height = height or 512
        width = width or 512
        
        # Process inputs
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
            
        # Process text inputs
        text_inputs = self.processor(
            prompt,
            negative_prompt=negative_prompt,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )
        
        # Process image guidance if provided
        if use_img_guidance and guidance_image is not None:
            image_inputs = self.processor.process_images(
                guidance_image,
                height=height,
                width=width,
                return_tensors="tf"
            )
        else:
            image_inputs = None
            
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Generate initial latents
        latents_shape = (batch_size * num_images_per_prompt, height // 8, width // 8, 4)
        if latents is None:
            latents = tf.random.normal(
                latents_shape,
                dtype=tf.float32,
                seed=generator.seed() if generator is not None else None
            )
            
        # Scale latents
        latents = latents * self.scheduler.init_noise_sigma
        
        # Prepare extra kwargs
        extra_kwargs = {}
        if cross_attention_kwargs is not None:
            extra_kwargs["cross_attention_kwargs"] = cross_attention_kwargs
            
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                latent_model_input = tf.concat([latents] * 2, axis=0)
                
                # Get model prediction
                noise_pred = self.model(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_inputs.input_ids,
                    input_image_latents=image_inputs.pixel_values if image_inputs is not None else None,
                    use_img_guidance=use_img_guidance,
                    **extra_kwargs
                )
                
                # Perform guidance
                noise_pred_uncond, noise_pred_text = tf.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                if use_img_guidance and guidance_image is not None:
                    noise_pred = noise_pred + img_guidance_scale * image_inputs.pixel_values
                    
                # Compute previous noisy sample
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    eta=eta,
                    generator=generator
                ).prev_sample
                
                # Update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                        
        # Decode latents
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5) * 255
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, tf.uint8)
        
        # Convert to PIL images
        image = image.numpy()
        if output_type == "pil":
            image = [Image.fromarray(img) for img in image]
            
        if not return_dict:
            return image
            
        return {"images": image}
        
    def progress_bar(self, total):
        """Get progress bar."""
        return tqdm(total=total, desc="Generating image")
