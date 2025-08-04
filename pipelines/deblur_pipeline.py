import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.archs.codeformer_arch import CodeFormer
from utils.face_utils import init_detection_model, get_face_landmarks_5, face_align

class DeblurPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.face_detector = None
        
    def load_model(self):
        if self.model is None:
            print("Loading CodeFormer model...")
            
            # Initialize the model with your custom architecture
            self.model = CodeFormer(
                dim_embd=512, 
                n_head=8, 
                n_layers=9, 
                codebook_size=1024
            ).to(self.device)
            
            # Load pretrained weights
            weights_path = "weights/codeformer.pth"
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                if 'params_ema' in checkpoint:
                    self.model.load_state_dict(checkpoint['params_ema'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                print("CodeFormer weights loaded successfully!")
            else:
                print(f"Warning: CodeFormer weights not found at {weights_path}")
                
            self.model.eval()
            
        # Initialize face detection
        if self.face_detector is None:
            self.face_detector = init_detection_model('retinaface_resnet50', device=self.device)
    
    def preprocess_image(self, image):
        """Preprocess image for CodeFormer"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert RGB to BGR for OpenCV processing
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_image(self, tensor_output):
        """Convert model output back to PIL Image"""
        # Remove batch dimension and convert to numpy
        output = tensor_output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        
        # Clamp and convert to uint8
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(output)
    
    def restore(self, image, fidelity=0.7, background_enhance=True, face_upsample=False):
        """
        Restore a blurred image using CodeFormer
        
        Args:
            image: PIL Image or numpy array
            fidelity: Balance between quality (lower) and fidelity (higher) [0-1]
            background_enhance: Whether to enhance background with Real-ESRGAN
            face_upsample: Whether to upsample faces for high-res images
        """
        try:
            self.load_model()
            
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Resize to model input size (512x512)
            input_tensor = torch.nn.functional.interpolate(
                input_tensor, size=(512, 512), mode='bilinear', align_corners=False
            )
            
            with torch.no_grad():
                # Run CodeFormer restoration
                output, _, _ = self.model(
                    input_tensor, 
                    w=fidelity, 
                    detach_16=True, 
                    code_only=False, 
                    adain=False
                )
            
            # Postprocess output
            restored_image = self.postprocess_image(output)
            
            # Resize back to original dimensions if needed
            if isinstance(image, Image.Image):
                original_size = image.size
                restored_image = restored_image.resize(original_size, Image.Resampling.LANCZOS)
            
            return restored_image
            
        except Exception as e:
            print(f"Error in image restoration: {str(e)}")
            return image  # Return original image on error
