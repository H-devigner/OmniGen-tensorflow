import os
import re
from typing import Dict, List, Optional, Union
import json

import tensorflow as tf
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from omnigen_tf.utils import crop_arr


class OmniGenTFProcessor:
    def __init__(self, 
                text_tokenizer,
                max_image_size: int = 1024):
        self.text_tokenizer = text_tokenizer
        self.max_image_size = max_image_size

        # Image transformation pipeline
        self.image_transform = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 127.5 - 1.0),
            tf.keras.layers.Lambda(lambda x: tf.ensure_shape(x, [None, None, 3]))
        ])

        self.collator = OmniGenTFCollator()
        self.separate_collator = OmniGenTFSeparateCollator()

    @classmethod
    def from_pretrained(cls, model_name):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                        cache_dir=cache_folder,
                                        allow_patterns="*.json")
        text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(text_tokenizer)

    def process_image(self, image_path):
        """Process a single image."""
        image = Image.open(image_path).convert('RGB')
        image = crop_arr(image, self.max_image_size)
        image = tf.convert_to_tensor(np.array(image))
        return self.image_transform(image)

    def add_prefix_instruction(self, prompt):
        """Add instruction prefix to the prompt."""
        if isinstance(prompt, str):
            user_prompt = 'Create an image based on the following prompt: ' + prompt
            return user_prompt
        elif isinstance(prompt, list):
            return ['Create an image based on the following prompt: ' + p for p in prompt]
        else:
            raise ValueError("Prompt must be a string or list of strings")

    def process_text(self, prompt: Union[str, List[str]], max_length: int = 77) -> Dict[str, tf.Tensor]:
        """Process text prompt."""
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # Add prefix instruction
        prompt = [self.add_prefix_instruction(p) for p in prompt]
        
        # Tokenize
        text_inputs = self.text_tokenizer(
            prompt,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors="tf"
        )
        
        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"]
        }
        
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        num_images_per_prompt: int = 1,
        max_length: int = 77,
        **kwargs
    ) -> Dict[str, tf.Tensor]:
        """Process inputs for generation."""
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` must be of type `str` or `list` but is {type(prompt)}")
            
        # Process text
        text_inputs = self.process_text(prompt, max_length)
        
        # Process image if provided
        if image is not None:
            if isinstance(image, Image.Image):
                image = [image]
                
            if not isinstance(image, list):
                raise ValueError(f"`image` must be of type `PIL.Image.Image` or `list` but is {type(image)}")
                
            image = [self.process_image(img) for img in image]
            image = tf.stack(image)
            
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = tf.expand_dims(image, 0)
                
            if image.shape[-1] != 3:
                raise ValueError(f"Image must have 3 channels but has {image.shape[-1]}")
                
            # Add to inputs
            text_inputs["pixel_values"] = image
            
        return text_inputs

    def save_pretrained(self, save_directory):
        self.text_tokenizer.save_pretrained(save_directory)

    def get_collate_fn(self):
        return self.collate_fn

    def get_separate_collate_fn(self):
        return self.separate_collate_fn

    def get_image_transform(self):
        return self.image_transform

    def get_text_tokenizer(self):
        return self.text_tokenizer

    def process_multi_modal_prompt(self, text, input_images):
        """Process a multi-modal prompt with text and images."""
        text = self.add_prefix_instruction(text)
        if input_images is None or len(input_images) == 0:
            model_inputs = self.text_tokenizer(text)
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)]

        for i in range(1, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text)
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        assert len(unique_image_ids) == len(input_images), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(input_images)} images"

        input_images = [input_images[x-1] for x in image_ids]

        all_input_ids = []
        img_inx = []
        for i in range(len(prompt_chunks)):
            all_input_ids.extend(prompt_chunks[i])
            if i != len(prompt_chunks) - 1:
                start_inx = len(all_input_ids)
                size = tf.shape(input_images[i])[-2] * tf.shape(input_images[i])[-1] // 256  # 16*16
                img_inx.append([start_inx, start_inx + size])
                all_input_ids.extend([0] * size)

        return {"input_ids": all_input_ids, "pixel_values": input_images, "image_sizes": img_inx}

    def process_multimodal_input(self, image_paths, prompts):
        """Process a batch of multimodal inputs (images and text prompts)."""
        images = tf.map_fn(self.process_image, image_paths, dtype=tf.float32)
        input_ids, attention_masks = tf.map_fn(self.process_text, prompts, dtype=(tf.int32, tf.int32))
        return images, input_ids, attention_masks

    def process_multimodal_batch(self, batch):
        """Process a batch of multimodal inputs (images and text prompts)."""
        images, input_ids, attention_masks = self.process_multimodal_input([x[0] for x in batch], [x[1] for x in batch])
        return {
            'image': images,
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }

    def process_multimodal_separate_batch(self, batch):
        """Process a batch of multimodal inputs (images and text prompts) with separate labels."""
        images, input_ids, attention_masks = self.process_multimodal_input([x[0] for x in batch], [x[1] for x in batch])
        return {
            'image': images,
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': tf.constant([x[2] for x in batch], dtype=tf.int32)
        }


class OmniGenTFCollator:
    """Data collator for OmniGen."""
    
    def __init__(self, pad_token_id=2):
        self.pad_token_id = pad_token_id
        
    def __call__(self, features: List[Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        """Collate features."""
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        
        # Determine max length
        max_length = max(x.shape[-1] for x in input_ids)
        
        # Pad input IDs
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            # Calculate padding
            padding_length = max_length - ids.shape[-1]
            
            if padding_length > 0:
                # Pad input IDs
                padded_ids = tf.pad(
                    ids,
                    [[0, 0], [0, padding_length]],
                    constant_values=self.pad_token_id
                )
                padded_input_ids.append(padded_ids)
                
                # Pad attention mask
                padded_mask = tf.pad(
                    mask,
                    [[0, 0], [0, padding_length]],
                    constant_values=0
                )
                padded_attention_mask.append(padded_mask)
            else:
                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
                
        # Stack tensors
        input_ids = tf.stack(padded_input_ids)
        attention_mask = tf.stack(padded_attention_mask)
        
        # Process images if present
        if "pixel_values" in features[0]:
            pixel_values = tf.stack([f["pixel_values"] for f in features])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values
            }
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class OmniGenTFSeparateCollator(OmniGenTFCollator):
    def __call__(self, features):
        """Process a batch of features with separate handling."""
        mllm_inputs = [f[0] for f in features]
        cfg_mllm_inputs = [f[1] for f in features]
        img_cfg_mllm_input = [f[2] for f in features]
        target_img_size = [f[3] for f in features]
        
        all_padded_input_ids, all_position_ids, all_attention_mask, all_padding_images, all_pixel_values, all_image_sizes = [], [], [], [], [], []
        
        # Process main inputs
        padded_input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes = \
            self.process_mllm_input(mllm_inputs, target_img_size)
        all_padded_input_ids.append(padded_input_ids)
        all_position_ids.append(position_ids)
        all_attention_mask.append(attention_mask)
        all_pixel_values.append(pixel_values)
        all_image_sizes.append(image_sizes)
        all_padding_images.append(padding_images)
        
        # Process CFG inputs if present
        if cfg_mllm_inputs[0] is not None:
            padded_input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes = \
                self.process_mllm_input(cfg_mllm_inputs, target_img_size)
            all_padded_input_ids.append(padded_input_ids)
            all_position_ids.append(position_ids)
            all_attention_mask.append(attention_mask)
            all_pixel_values.append(pixel_values)
            all_image_sizes.append(image_sizes)
            all_padding_images.append(padding_images)
        
        # Process image CFG inputs if present
        if img_cfg_mllm_input[0] is not None:
            padded_input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes = \
                self.process_mllm_input(img_cfg_mllm_input, target_img_size)
            all_padded_input_ids.append(padded_input_ids)
            all_position_ids.append(position_ids)
            all_attention_mask.append(attention_mask)
            all_pixel_values.append(pixel_values)
            all_image_sizes.append(image_sizes)
            all_padding_images.append(padding_images)
        
        return {
            "input_ids": all_padded_input_ids,
            "attention_mask": all_attention_mask,
            "position_ids": all_position_ids,
            "input_pixel_values": all_pixel_values,
            "input_image_sizes": all_image_sizes,
            "padding_images": all_padding_images,
        }
