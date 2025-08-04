import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import cv2

class SketchPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = None
        
    def load_model(self):
        if self.pipe is None:
            print("Loading Sketch-to-Image pipeline...")
            
            # Load ControlNet for scribble
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_scribble", 
                torch_dtype=torch.float16
            )
            
            # Load Stable Diffusion pipeline with ControlNet
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "SG161222/Realistic_Vision_V5.1_noVAE",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None
            ).to(self.device)
            
            # Use faster scheduler
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Enable memory optimizations
            self.pipe.enable_model_cpu_offload()
            
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xFormers memory optimization enabled")
            except ImportError:
                print("xFormers not available, continuing without optimization")
                
            print("Sketch-to-Image pipeline loaded successfully!")
    
    def preprocess_sketch(self, sketch_image):
        """Preprocess sketch image for ControlNet"""
        if isinstance(sketch_image, Image.Image):
            sketch_array = np.array(sketch_image)
        else:
            sketch_array = sketch_image
            
        # Convert to grayscale if needed
        if len(sketch_array.shape) == 3:
            sketch_array = cv2.cvtColor(sketch_array, cv2.COLOR_RGB2GRAY)
        
        # Convert back to PIL Image
        sketch_pil = Image.fromarray(sketch_array).convert("RGB")
        
        return sketch_pil
    
    def generate(self, sketch_image, prompt, negative_prompt=None, 
                num_inference_steps=30, guidance_scale=7.5, 
                controlnet_conditioning_scale=0.75, seed=None):
        """
        Generate image from sketch
        
        Args:
            sketch_image: PIL Image of the sketch
            prompt: Text prompt describing the desired image
            negative_prompt: Negative text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            controlnet_conditioning_scale: ControlNet influence strength
            seed: Random seed for reproducibility
        """
        try:
            self.load_model()
            
            # Preprocess sketch
            control_image = self.preprocess_sketch(sketch_image)
            
            # Set up generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Set default negative prompt if not provided
            if negative_prompt is None:
                negative_prompt = ("deformed, ugly, disfigured, low quality, cartoon, "
                                 "anime, plastic, fake, bad anatomy, extra limbs, mutated")
            
            # Generate image
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=generator,
                    width=512,
                    height=512
                ).images[0]
            
            return result
            
        except Exception as e:
            print(f"Error in sketch generation: {str(e)}")
            return None
