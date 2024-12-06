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

    def process_text(self, prompt):
        """Process a single text prompt."""
        prompt = self.add_prefix_instruction(prompt)
        encoding = self.text_tokenizer.encode_plus(
            prompt,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        return encoding['input_ids'], encoding['attention_mask']

    def process_batch(self, image_paths, prompts):
        """Process a batch of images and text prompts."""
        images = tf.map_fn(self.process_image, image_paths, dtype=tf.float32)
        input_ids, attention_masks = tf.map_fn(self.process_text, prompts, dtype=(tf.int32, tf.int32))
        return images, input_ids, attention_masks

    def collate_fn(self, batch):
        """Collate function for the dataset."""
        images, input_ids, attention_masks = self.process_batch([x[0] for x in batch], [x[1] for x in batch])
        return {
            'image': images,
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }

    def separate_collate_fn(self, batch):
        """Collate function for the separate dataset."""
        images, input_ids, attention_masks = self.process_batch([x[0] for x in batch], [x[1] for x in batch])
        return {
            'image': images,
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': tf.constant([x[2] for x in batch], dtype=tf.int32)
        }

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

    def __call__(self, instructions: List[str], input_images: List[List[str]] = None,
                height: int = 1024, width: int = 1024, 
                negative_prompt: str = "low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers.",
                use_img_cfg: bool = True,
                separate_cfg_input: bool = False,
                use_input_image_size_as_output: bool = False) -> Dict:
        """Process instructions and images for model input."""
        if input_images is None:
            use_img_cfg = False
        if isinstance(instructions, str):
            instructions = [instructions]
            input_images = [input_images]

        input_data = []
        for i in range(len(instructions)):
            cur_instruction = instructions[i]
            cur_input_images = None if input_images is None else input_images[i]
            if cur_input_images is not None and len(cur_input_images) > 0:
                cur_input_images = [self.process_image(x) for x in cur_input_images]
            else:
                cur_input_images = None
                assert "<img><|image_1|></img>" not in cur_instruction

            mllm_input = self.process_multi_modal_prompt(cur_instruction, cur_input_images)

            neg_mllm_input, img_cfg_mllm_input = None, None
            neg_mllm_input = self.process_multi_modal_prompt(negative_prompt, None)
            if use_img_cfg:
                if cur_input_images is not None and len(cur_input_images) >= 1:
                    img_cfg_prompt = [f"<img><|image_{i+1}|></img>" for i in range(len(cur_input_images))]
                    img_cfg_mllm_input = self.process_multi_modal_prompt(" ".join(img_cfg_prompt), cur_input_images)
                else:
                    img_cfg_mllm_input = neg_mllm_input

            if use_input_image_size_as_output:
                input_data.append((mllm_input, neg_mllm_input, img_cfg_mllm_input, 
                                [tf.shape(mllm_input['pixel_values'][0])[-2], tf.shape(mllm_input['pixel_values'][0])[-1]]))
            else:
                input_data.append((mllm_input, neg_mllm_input, img_cfg_mllm_input, [height, width]))

        if separate_cfg_input:
            return self.separate_collator(input_data)
        return self.collator(input_data)

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
    def __init__(self, pad_token_id=2, hidden_size=3072):
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size

    def create_position(self, attention_mask, num_tokens_for_output_images):
        """Create position IDs for input sequence."""
        position_ids = []
        text_length = tf.shape(attention_mask)[-1]
        img_length = tf.reduce_max(num_tokens_for_output_images)
        
        for mask in attention_mask:
            temp_l = tf.reduce_sum(tf.cast(mask, tf.int32))
            pos_ids = tf.range(temp_l + 1 + img_length, dtype=tf.int32)
            
            if temp_l < text_length:
                pad_pos = tf.zeros([text_length - temp_l], dtype=tf.int32)
                pos_ids = tf.concat([pad_pos, pos_ids], axis=0)
            
            position_ids.append(tf.expand_dims(pos_ids, 0))
        
        return tf.concat(position_ids, axis=0)

    def create_mask(self, attention_mask, num_tokens_for_output_images):
        """Create attention mask for transformer."""
        extended_mask = []
        padding_images = []
        text_length = tf.shape(attention_mask)[-1]
        img_length = tf.reduce_max(num_tokens_for_output_images)
        seq_len = text_length + img_length + 1  # +1 for time embedding
        
        for i, mask in enumerate(attention_mask):
            temp_l = tf.reduce_sum(tf.cast(mask, tf.int32))
            pad_l = text_length - temp_l
            
            # Create causal mask for text
            temp_mask = tf.linalg.band_part(tf.ones([temp_l + 1, temp_l + 1]), -1, 0)
            
            # Add image attention
            image_mask = tf.zeros([temp_l + 1, img_length])
            temp_mask = tf.concat([temp_mask, image_mask], axis=-1)
            
            # Allow images to attend to everything
            image_mask = tf.ones([img_length, temp_l + img_length + 1])
            temp_mask = tf.concat([temp_mask, image_mask], axis=0)
            
            # Handle padding
            if pad_l > 0:
                pad_mask = tf.zeros([temp_l + 1 + img_length, pad_l])
                temp_mask = tf.concat([pad_mask, temp_mask], axis=-1)
                
                pad_mask = tf.ones([pad_l, seq_len])
                temp_mask = tf.concat([pad_mask, temp_mask], axis=0)
            
            # Handle image padding
            true_img_length = num_tokens_for_output_images[i]
            pad_img_length = img_length - true_img_length
            if pad_img_length > 0:
                temp_mask = tf.tensor_scatter_nd_update(
                    temp_mask,
                    tf.expand_dims(tf.range(seq_len), 1),
                    tf.zeros([seq_len, pad_img_length])
                )
                temp_padding_imgs = tf.zeros([1, pad_img_length, self.hidden_size])
            else:
                temp_padding_imgs = None
            
            extended_mask.append(tf.expand_dims(temp_mask, 0))
            padding_images.append(temp_padding_imgs)
        
        return tf.concat(extended_mask, axis=0), padding_images

    def adjust_attention_for_input_images(self, attention_mask, image_sizes):
        """Adjust attention mask for input images."""
        if image_sizes is None:
            return attention_mask
        
        for size in image_sizes:
            start_idx, end_idx = size
            attention_mask = tf.tensor_scatter_nd_update(
                attention_mask,
                tf.expand_dims(tf.range(start_idx, end_idx), 1),
                tf.ones([end_idx - start_idx])
            )
        return attention_mask

    def pad_input_ids(self, input_ids, image_sizes):
        """Pad input IDs for batch processing."""
        max_length = max(len(ids) for ids in input_ids)
        padded_ids = []
        attention_mask = []
        
        for ids in input_ids:
            padding_length = max_length - len(ids)
            padded = ids + [self.pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length
            padded_ids.append(padded)
            attention_mask.append(mask)
            
        return tf.constant(padded_ids), tf.constant(attention_mask)

    def process_mllm_input(self, mllm_inputs, target_img_size):
        """Process multi-modal language model inputs."""
        input_ids = [x['input_ids'] for x in mllm_inputs]
        pixel_values = [x['pixel_values'] for x in mllm_inputs]
        image_sizes = [x['image_sizes'] for x in mllm_inputs]
        
        # Pad input IDs and create attention mask
        padded_input_ids, attention_mask = self.pad_input_ids(input_ids, image_sizes)
        
        # Adjust attention mask for input images
        attention_mask = self.adjust_attention_for_input_images(attention_mask, image_sizes[0])
        
        # Create position IDs and extended attention mask
        position_ids = self.create_position(attention_mask, target_img_size)
        extended_mask, padding_images = self.create_mask(attention_mask, target_img_size)
        
        return padded_input_ids, position_ids, extended_mask, padding_images, pixel_values, image_sizes

    def __call__(self, features):
        """Process a batch of features."""
        mllm_inputs = [f[0] for f in features]
        cfg_mllm_inputs = [f[1] for f in features]
        img_cfg_mllm_input = [f[2] for f in features]
        target_img_size = [f[3] for f in features]
        
        if img_cfg_mllm_input[0] is not None:
            mllm_inputs = mllm_inputs + cfg_mllm_inputs + img_cfg_mllm_input
            target_img_size = target_img_size + target_img_size + target_img_size
        else:
            mllm_inputs = mllm_inputs + cfg_mllm_inputs
            target_img_size = target_img_size + target_img_size
        
        all_padded_input_ids, all_position_ids, all_attention_mask, all_padding_images, all_pixel_values, all_image_sizes = \
            self.process_mllm_input(mllm_inputs, target_img_size)
        
        return {
            "input_ids": all_padded_input_ids,
            "attention_mask": all_attention_mask,
            "position_ids": all_position_ids,
            "input_pixel_values": all_pixel_values,
            "input_image_sizes": all_image_sizes,
            "padding_images": all_padding_images,
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
