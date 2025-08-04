"""Image processing utilities"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path

def validate_image(image):
    """Validate input image format and size"""
    if image is None:
        return False, "No image provided"
    
    if isinstance(image, Image.Image):
        if image.size[0] * image.size[1] > 2048 * 2048:
            return False, "Image too large (max 2048x2048)"
        return True, "Valid image"
    
    if isinstance(image, np.ndarray):
        if len(image.shape) not in [2, 3]:
            return False, "Invalid image dimensions"
        if image.shape[0] * image.shape[1] > 2048 * 2048:
            return False, "Image too large (max 2048x2048)"
        return True, "Valid image"
    
    return False, "Unsupported image format"

def prepare_output_dir():
    """Prepare output directories"""
    dirs = ["outputs", "outputs/sketch", "outputs/face_edit", "outputs/deblur", "temp"]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    return True

def save_image(image, filepath, quality=95):
    """Save image to file"""
    try:
        if isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.dim() == 4:  # Batch dimension
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
                image = image.permute(1, 2, 0)
            
            image = image.cpu().numpy()
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(filepath), image)
            else:
                Image.fromarray(image).save(filepath, quality=quality)
                
        elif isinstance(image, Image.Image):
            image.save(filepath, quality=quality)
            
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                Image.fromarray(image_rgb).save(filepath, quality=quality)
            else:
                Image.fromarray(image).save(filepath, quality=quality)
        
        return True, f"Image saved to {filepath}"
    except Exception as e:
        return False, f"Error saving image: {str(e)}"

def resize_image(image, target_size, maintain_aspect=True):
    """Resize image to target size"""
    if isinstance(image, Image.Image):
        if maintain_aspect:
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
        else:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image
    
    elif isinstance(image, np.ndarray):
        if maintain_aspect:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    return image

def preprocess_for_model(image, target_size=(512, 512), normalize=True):
    """Preprocess image for model input"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    # Convert to tensor
    if len(image.shape) == 3:
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    else:
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    return image

def postprocess_from_model(tensor_output, target_size=None):
    """Postprocess model output to image"""
    # Remove batch dimension and convert to numpy
    if tensor_output.dim() == 4:
        output = tensor_output.squeeze(0)
    else:
        output = tensor_output
    
    if output.dim() == 3:
        output = output.permute(1, 2, 0)
    
    output = output.cpu().numpy()
    
    # Denormalize if needed
    if output.max() <= 1:
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
    else:
        output = np.clip(output, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(output.shape) == 3:
        output = Image.fromarray(output)
    else:
        output = Image.fromarray(output, mode='L')
    
    # Resize if needed
    if target_size is not None:
        output = output.resize(target_size, Image.Resampling.LANCZOS)
    
    return output

def create_image_grid(images, rows=None, cols=None, padding=10):
    """Create a grid of images"""
    if not images:
        return None
    
    # Convert all images to PIL if needed
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            img = postprocess_from_model(img)
        pil_images.append(img)
    
    # Calculate grid dimensions
    n_images = len(pil_images)
    if rows is None and cols is None:
        rows = int(np.sqrt(n_images))
        cols = int(np.ceil(n_images / rows))
    elif rows is None:
        rows = int(np.ceil(n_images / cols))
    elif cols is None:
        cols = int(np.ceil(n_images / rows))
    
    # Get max dimensions
    max_width = max(img.width for img in pil_images)
    max_height = max(img.height for img in pil_images)
    
    # Create grid
    grid_width = cols * max_width + (cols - 1) * padding
    grid_height = rows * max_height + (rows - 1) * padding
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    for i, img in enumerate(pil_images):
        row = i // cols
        col = i % cols
        x = col * (max_width + padding)
        y = row * (max_height + padding)
        
        # Center image in cell
        x_offset = (max_width - img.width) // 2
        y_offset = (max_height - img.height) // 2
        
        grid.paste(img, (x + x_offset, y + y_offset))
    
    return grid
