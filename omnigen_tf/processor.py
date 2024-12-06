"""OmniGen processor for text and image inputs."""
import os
from typing import List, Optional, Union
import tensorflow as tf
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer

class OmniGenProcessor:
    """Processor for OmniGen inputs."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_size: Optional[dict] = None,
        prefix_tokens: Optional[List[str]] = None
    ):
        """Initialize processor.
        
        Args:
            tokenizer: Tokenizer for text processing
            image_size: Default image size
            prefix_tokens: Special prefix tokens to add
        """
        self.tokenizer = tokenizer
        self.image_size = image_size or {"height": 512, "width": 512}
        self.prefix_tokens = prefix_tokens or []
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load processor from pretrained model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model
            **kwargs: Additional arguments
            
        Returns:
            OmniGenProcessor: Loaded processor
        """
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            **kwargs
        )
        
        # Load config for image size and prefix tokens
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                image_size = config.get("image_size", None)
                prefix_tokens = config.get("prefix_tokens", None)
        else:
            image_size = None
            prefix_tokens = None
            
        return cls(
            tokenizer=tokenizer,
            image_size=image_size,
            prefix_tokens=prefix_tokens
        )
        
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        truncation: Union[bool, str] = True,
        return_tensors: Optional[str] = None,
    ):
        """Process text inputs.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt for guidance
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Truncation strategy
            return_tensors: Return tensor format
            
        Returns:
            Processed inputs
        """
        if isinstance(prompt, str):
            prompt = [prompt]
            
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            prompt = prompt + negative_prompt
            
        # Add prefix tokens if specified
        if self.prefix_tokens:
            prompt = [f"{' '.join(self.prefix_tokens)} {p}" for p in prompt]
            
        inputs = self.tokenizer(
            prompt,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        return inputs
        
    def process_images(
        self,
        images: Union[Image.Image, List[Image.Image]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        return_tensors: Optional[str] = None
    ):
        """Process image inputs.
        
        Args:
            images: Input images
            height: Target height
            width: Target width
            return_tensors: Return tensor format
            
        Returns:
            Processed images
        """
        if isinstance(images, Image.Image):
            images = [images]
            
        height = height or self.image_size["height"]
        width = width or self.image_size["width"]
        
        processed_images = []
        image_sizes = []
        
        for image in images:
            # Resize image
            if image.size != (width, height):
                image = image.resize((width, height), Image.LANCZOS)
                
            # Convert to numpy array
            image = np.array(image)
            
            # Normalize to [-1, 1]
            image = (image / 127.5) - 1.0
            
            processed_images.append(image)
            image_sizes.append((height, width))
            
        # Stack images
        pixel_values = np.stack(processed_images)
        
        if return_tensors == "tf":
            pixel_values = tf.convert_to_tensor(pixel_values)
            image_sizes = tf.convert_to_tensor(image_sizes)
            
        return {
            "pixel_values": pixel_values,
            "image_sizes": image_sizes
        }
