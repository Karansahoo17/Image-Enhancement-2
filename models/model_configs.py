"""Model configurations for different pipelines"""

# Sketch to Image Configuration
SKETCH_CONFIG = {
    "controlnet_model": "lllyasviel/control_v11p_sd15_scribble",
    "base_model": "SG161222/Realistic_Vision_V5.1_noVAE",
    "default_steps": 30,
    "default_guidance": 7.5,
    "default_controlnet_strength": 0.75,
    "image_size": 512
}

# Face Edit Configuration  
FACE_EDIT_CONFIG = {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "controlnet_model": "InstantX/InstantID",
    "ip_adapter_path": "weights/ip-adapter.bin",
    "face_analysis_model": "antelopev2",
    "default_steps": 30,
    "default_guidance": 5.0,
    "default_ip_scale": 0.8,
    "default_controlnet_scale": 0.8,
    "image_size": 1024
}

# Deblur Configuration
DEBLUR_CONFIG = {
    "model_path": "weights/codeformer.pth",
    "detection_model": "weights/facelib/detection/detection_Resnet50_Final.pth", 
    "parsing_model": "weights/facelib/parsing/parsing_parsenet.pth",
    "default_fidelity": 0.7,
    "input_size": 512,
    "device": "cuda"
}

# Global Configuration
GLOBAL_CONFIG = {
    "cache_dir": "weights",
    "output_dir": "outputs", 
    "temp_dir": "temp",
    "max_image_size": 2048,
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "torch_dtype": "float16"
}
