"""OmniGen pipeline for text-to-image generation."""
from __future__ import annotations

import os
import time
from typing import Optional, List, Union, Dict, Any
import tensorflow as tf
import torch
from PIL import Image
import numpy as np
from huggingface_hub import snapshot_download

from .model import OmniGenTF
from .processor import OmniGenProcessor
from .vae import AutoencoderKL

class OmniGenPipeline:
    """Pipeline for text-to-image generation using OmniGen."""
    
    def __init__(
        self,
        model: OmniGenTF,
        processor: OmniGenProcessor,
        vae: AutoencoderKL,
        device: str = "gpu",
        **kwargs
    ):
        """Initialize pipeline."""
        self.model = model
        self.processor = processor
        self.vae = vae
        self.device = device
        
        # Set up device strategy
        if device == "gpu" and tf.config.list_physical_devices('GPU'):
            self.strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        else:
            self.strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        vae_path: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 1,
        mixed_precision: bool = False,
        **kwargs
    ) -> "OmniGenPipeline":
        """Load pipeline from pretrained model."""
        try:
            # Download and set up processor
            print("Loading processor...")
            processor = OmniGenProcessor.from_pretrained(model_name)
            
            # Set up model with proper error handling
            print("Loading model...")
            model = OmniGenTF.from_pretrained(
                model_name,
                use_mixed_precision=mixed_precision,
                **kwargs
            )
            
            # Set up VAE
            print("Loading VAE...")
            if vae_path is None:
                vae_path = model_name
            vae = AutoencoderKL.from_pretrained(vae_path)
            
            # Initialize pipeline
            return cls(
                model=model,
                processor=processor,
                vae=vae,
                **kwargs
            )
            
        except Exception as e:
            print(f"Error initializing pipeline: {str(e)}")
            raise
    
    def __call__(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images from text prompt."""
        try:
            # Process input
            print("Processing input...")
            inputs = self.processor(
                prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt
            )
            
            # Run inference with strategy
            print("Generating image...")
            with self.strategy.scope():
                # Generate latents
                latents = self.model.generate(
                    **inputs,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    **kwargs
                )
                
                # Decode latents
                images = self.vae.decode(latents)
                
            # Post-process
            images = self.processor.postprocess_images(images)
            
            return images
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            raise

    def progress_bar(self, total):
        """Get progress bar."""
        return tqdm(total=total, desc="Generating image")
