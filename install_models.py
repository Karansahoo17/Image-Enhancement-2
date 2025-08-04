#!/usr/bin/env python3
"""
Automated model download and setup script for AI Image Pipeline
"""

import os
import requests
import gdown
from pathlib import Path
from huggingface_hub import hf_hub_download
import zipfile
import tarfile

def create_directories():
    """Create necessary directories"""
    directories = [
        "weights",
        "weights/facelib/detection",
        "weights/facelib/parsing", 
        "weights/CodeFormer",
        "weights/antelopev2",
        "weights/ControlNetModel"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def download_codeformer_models():
    """Download CodeFormer related models"""
    print("\n📥 Downloading CodeFormer models...")
    
    models = [
        {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "path": "weights/facelib/detection/detection_Resnet50_Final.pth"
        },
        {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth", 
            "path": "weights/facelib/parsing/parsing_parsenet.pth"
        },
        {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            "path": "weights/CodeFormer/codeformer.pth"
        }
    ]
    
    for model in models:
        if not os.path.exists(model["path"]):
            print(f"Downloading {model['path']}...")
            response = requests.get(model["url"], stream=True)
            response.raise_for_status()
            
            with open(model["path"], "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✓ Downloaded {model['path']}")
        else:
            print(f"✓ {model['path']} already exists")

def download_instantid_models():
    """Download InstantID models"""
    print("\n📥 Downloading InstantID models...")
    
    try:
        # ControlNet model
        hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ControlNetModel/config.json",
            local_dir="weights"
        )
        
        hf_hub_download(
            repo_id="InstantX/InstantID", 
            filename="ControlNetModel/diffusion_pytorch_model.safetensors",
            local_dir="weights"
        )
        
        # IP-Adapter
        hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ip-adapter.bin", 
            local_dir="weights"
        )
        
        print("✓ InstantID models downloaded")
        
    except Exception as e:
        print(f"❌ Error downloading InstantID models: {e}")

def download_face_analysis_models():
    """Download face analysis models"""
    print("\n📥 Downloading face analysis models...")
    
    try:
        # AntelopeV2 models
        models = [
            "glintr100.onnx",
            "scrfd_10g_bnkps.onnx"
        ]
        
        for model in models:
            hf_hub_download(
                repo_id="InstantX/InstantID",
                repo_type="space",
                filename=f"models/antelopev2/{model}",
                local_dir="weights"
            )
        
        print("✓ Face analysis models downloaded")
        
    except Exception as e:
        print(f"❌ Error downloading face analysis models: {e}")

def download_additional_models():
    """Download additional required models"""
    print("\n📥 Downloading additional models...")
    
    # This will be handled by the diffusers library automatically
    # when the pipelines are first loaded
    print("✓ Additional models will be downloaded automatically on first use")

def verify_installation():
    """Verify that all required files are present"""
    print("\n🔍 Verifying installation...")
    
    required_files = [
        "weights/facelib/detection/detection_Resnet50_Final.pth",
        "weights/facelib/parsing/parsing_parsenet.pth", 
        "weights/CodeFormer/codeformer.pth",
        "weights/ip-adapter.bin"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {len(missing_files)}")
        return False
    else:
        print("\n🎉 All required files are present!")
        return True

def main():
    """Main installation function"""
    print("🚀 Starting AI Image Pipeline model installation...")
    print("This may take a while depending on your internet connection.\n")
    
    try:
        create_directories()
        download_codeformer_models()
        download_instantid_models() 
        download_face_analysis_models()
        download_additional_models()
        
        if verify_installation():
            print("\n🎉 Installation completed successfully!")
            print("\nYou can now run the pipeline with:")
            print("python main.py")
        else:
            print("\n❌ Installation completed with some missing files.")
            print("Please check the error messages above and try again.")
            
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
