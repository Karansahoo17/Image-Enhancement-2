import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis

# Custom InstantID pipeline (simplified version)
class FaceEditPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = None
        self.face_app = None
        
    def load_model(self):
        if self.pipe is None:
            print("Loading Face Edit pipeline...")
            
            # Load face analysis model
            self.face_app = FaceAnalysis(name='antelopev2', root='weights/', 
                                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Load ControlNet for InstantID
            controlnet = ControlNetModel.from_pretrained(
                "InstantX/InstantID", 
                subfolder="ControlNetModel",
                torch_dtype=torch.float16
            )
            
            # Load SDXL pipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=torch.float16
            ).to(self.device)
            
            # Load IP-Adapter weights
            self.pipe.load_ip_adapter("weights/ip-adapter.bin")
            
            # Enable memory optimizations
            self.pipe.enable_sequential_cpu_offload()
            
            print("Face Edit pipeline loaded successfully!")
    
    def extract_face_features(self, face_image):
        """Extract face embeddings and keypoints"""
        if isinstance(face_image, Image.Image):
            face_array = np.array(face_image)
        else:
            face_array = face_image
            
        # Convert RGB to BGR for face analysis
        face_bgr = cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR)
        
        # Get face info
        faces = self.face_app.get(face_bgr)
        
        if not faces:
            raise ValueError("No face detected in the input image")
        
        # Sort faces by area (largest first)
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        
        # Get the largest face
        face_info = faces[0]
        face_emb = face_info.embedding
        face_kps = face_info.kps
        
        return face_emb, face_kps
    
    def generate(self, face_image, prompt, negative_prompt=None,
                num_inference_steps=30, guidance_scale=5.0,
                ip_adapter_scale=0.8, controlnet_conditioning_scale=0.8,
                seed=None):
        """
        Edit face based on text prompt while preserving identity
        
        Args:
            face_image: PIL Image of the face
            prompt: Text prompt describing desired changes
            negative_prompt: Negative text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            ip_adapter_scale: IP-Adapter influence strength
            controlnet_conditioning_scale: ControlNet influence strength
            seed: Random seed for reproducibility
        """
        try:
            self.load_model()
            
            # Extract face features
            face_emb, face_kps = self.extract_face_features(face_image)
            
            # Set up generator
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Set default negative prompt
            if negative_prompt is None:
                negative_prompt = ("ugly, disfigured, distorted, low quality, blurry, "
                                 "deformed eyes, bad anatomy")
            
            # Generate edited face
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image_embeds=face_emb,
                    image=face_kps,  # Simplified - normally needs keypoint processing
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    ip_adapter_scale=ip_adapter_scale,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=1024,
                    height=1024
                ).images[0]
            
            return result
            
        except Exception as e:
            print(f"Error in face editing: {str(e)}")
            return None
