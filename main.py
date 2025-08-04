import gradio as gr
import torch
import gc
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from pipelines.sketch_pipeline import SketchPipeline
from pipelines.face_edit_pipeline import FaceEditPipeline  
from pipelines.deblur_pipeline import DeblurPipeline

class AIImagePipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize pipelines (lazy loading)
        self.sketch_pipeline = None
        self.face_edit_pipeline = None
        self.deblur_pipeline = None
        
        # Prepare directories
        os.makedirs("weights", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
    
    def get_sketch_pipeline(self):
        if self.sketch_pipeline is None:
            self.sketch_pipeline = SketchPipeline(device=self.device)
        return self.sketch_pipeline
    
    def get_face_edit_pipeline(self):
        if self.face_edit_pipeline is None:
            self.face_edit_pipeline = FaceEditPipeline(device=self.device)
        return self.face_edit_pipeline
    
    def get_deblur_pipeline(self):
        if self.deblur_pipeline is None:
            self.deblur_pipeline = DeblurPipeline(device=self.device)
        return self.deblur_pipeline
    
    def sketch_to_image(self, sketch, prompt, neg_prompt, steps, guidance, strength, seed):
        try:
            if sketch is None:
                return None, "Please upload a sketch image"
            
            pipeline = self.get_sketch_pipeline()
            result = pipeline.generate(
                sketch_image=sketch,
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=int(steps),
                guidance_scale=guidance,
                controlnet_conditioning_scale=strength,
                seed=int(seed) if seed >= 0 else None
            )
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            if result is not None:
                return result, "✅ Image generated successfully!"
            else:
                return None, "❌ Generation failed"
                
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def edit_face(self, face_img, prompt, neg_prompt, steps, guidance, ip_scale, ctrl_scale, seed):
        try:
            if face_img is None:
                return None, "Please upload a face image"
            
            pipeline = self.get_face_edit_pipeline()
            result = pipeline.generate(
                face_image=face_img,
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=int(steps),
                guidance_scale=guidance,
                ip_adapter_scale=ip_scale,
                controlnet_conditioning_scale=ctrl_scale,
                seed=int(seed) if seed >= 0 else None
            )
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            if result is not None:
                return result, "✅ Face edited successfully!"
            else:
                return None, "❌ Face editing failed"
                
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def deblur_image(self, blurred_img, fidelity, bg_enhance, face_upsample):
        try:
            if blurred_img is None:
                return None, "Please upload a blurred image"
            
            pipeline = self.get_deblur_pipeline()
            result = pipeline.restore(
                image=blurred_img,
                fidelity=fidelity,
                background_enhance=bg_enhance,
                face_upsample=face_upsample
            )
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            if result is not None:
                return result, "✅ Image restored successfully!"
            else:
                return None, "❌ Image restoration failed"
                
        except Exception as e:
            return None, f"❌ Error: {str(e)}"

def create_interface():
    app = AIImagePipeline()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
        max-width: 1200px !important;
        margin: 0 auto;
    }
    .tab-nav button.selected {
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        color: white;
    }
    """
    
    with gr.Blocks(title="🎨 AI Image Pipeline", theme=gr.themes.Soft(), css=css) as interface:
        
        gr.Markdown("""
        # 🎨 AI Image Processing Pipeline
        ### Transform your images with state-of-the-art AI: Convert sketches to photos, edit faces with text prompts, or restore blurred images
        """)
        
        with gr.Tabs():
            
            # 🖼️ SKETCH TO IMAGE TAB
            with gr.Tab("🖼️ Sketch to Image"):
                gr.Markdown("### Convert your sketches into photorealistic images")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        sketch_input = gr.Image(
                            label="📝 Upload Sketch", 
                            type="pil",
                            height=400
                        )
                        
                        sketch_prompt = gr.Textbox(
                            label="✨ Prompt",
                            placeholder="Describe what you want to create...",
                            lines=3,
                            value="photograph of a beautiful young woman with long, flowing hair, detailed face, expressive eyes, soft lighting, professional portrait, 8k, ultra-realistic, sharp focus"
                        )
                        
                        sketch_negative = gr.Textbox(
                            label="🚫 Negative Prompt",
                            placeholder="What to avoid...",
                            lines=2,
                            value="deformed, ugly, disfigured, low quality, cartoon, anime, plastic, fake, bad anatomy"
                        )
                        
                        with gr.Row():
                            sketch_steps = gr.Slider(1, 50, value=30, step=1, label="Steps")
                            sketch_guidance = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance")
                        
                        with gr.Row():
                            sketch_strength = gr.Slider(0, 1, value=0.75, step=0.05, label="ControlNet Strength")
                            sketch_seed = gr.Number(value=-1, label="Seed (-1 for random)")
                        
                        sketch_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        sketch_output = gr.Image(label="Generated Image", height=400)
                        sketch_status = gr.Textbox(label="Status", interactive=False)
            
            # 👤 FACE EDITING TAB  
            with gr.Tab("👤 Face Editing"):
                gr.Markdown("### Edit faces using text descriptions while preserving identity")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        face_input = gr.Image(
                            label="📸 Upload Face Image", 
                            type="pil",
                            height=400
                        )
                        
                        face_prompt = gr.Textbox(
                            label="✏️ Edit Description",
                            placeholder="Describe the changes you want...",
                            lines=3,
                            value="a photo of a person smiling, professional portrait, cinematic lighting, high quality"
                        )
                        
                        face_negative = gr.Textbox(
                            label="🚫 Negative Prompt",
                            placeholder="What to avoid...",
                            lines=2,
                            value="ugly, disfigured, distorted, low quality, blurry, deformed eyes"
                        )
                        
                        with gr.Row():
                            face_steps = gr.Slider(1, 50, value=30, step=1, label="Steps")
                            face_guidance = gr.Slider(1, 20, value=5.0, step=0.5, label="Guidance")
                        
                        with gr.Row():
                            face_ip_scale = gr.Slider(0, 1, value=0.8, step=0.1, label="Identity Strength")
                            face_ctrl_scale = gr.Slider(0, 1, value=0.8, step=0.1, label="Control Strength")
                        
                        face_seed = gr.Number(value=-1, label="Seed (-1 for random)")
                        face_btn = gr.Button("✨ Edit Face", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        face_output = gr.Image(label="Edited Face", height=400)
                        face_status = gr.Textbox(label="Status", interactive=False)
            
            # 🔧 IMAGE RESTORATION TAB
            with gr.Tab("🔧 Image Restoration"):
                gr.Markdown("### Restore and enhance blurred or degraded face images")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        deblur_input = gr.Image(
                            label="🌫️ Upload Blurred Image", 
                            type="pil",
                            height=400
                        )
                        
                        gr.Markdown("### Settings")
                        
                        fidelity = gr.Slider(
                            0, 1, 
                            value=0.7, 
                            step=0.05, 
                            label="🎯 Fidelity (Quality vs Faithfulness)",
                            info="Lower = Higher quality, Higher = More faithful to original"
                        )
                        
                        bg_enhance = gr.Checkbox(
                            value=True, 
                            label="🌟 Enhance Background", 
                            info="Use Real-ESRGAN for background enhancement"
                        )
                        
                        face_upsample = gr.Checkbox(
                            value=False, 
                            label="⬆️ Face Upsampling", 
                            info="Upsample faces for high-resolution images"
                        )
                        
                        deblur_btn = gr.Button("🔧 Restore Image", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        deblur_output = gr.Image(label="Restored Image", height=400)
                        deblur_status = gr.Textbox(label="Status", interactive=False)
        
        # Event handlers
        sketch_btn.click(
            app.sketch_to_image,
            inputs=[sketch_input, sketch_prompt, sketch_negative, sketch_steps, 
                   sketch_guidance, sketch_strength, sketch_seed],
            outputs=[sketch_output, sketch_status]
        )
        
        face_btn.click(
            app.edit_face,
            inputs=[face_input, face_prompt, face_negative, face_steps, 
                   face_guidance, face_ip_scale, face_ctrl_scale, face_seed],
            outputs=[face_output, face_status]
        )
        
        deblur_btn.click(
            app.deblur_image,
            inputs=[deblur_input, fidelity, bg_enhance, face_upsample],
            outputs=[deblur_output, deblur_status]
        )
        
        # Footer
        gr.Markdown("""
        ---
        💡 **Tips:**
        - **Sketch to Image**: Use clear, simple sketches with good contrast
        - **Face Editing**: Upload clear, front-facing photos for best results  
        - **Image Restoration**: Works best on face images with blur or noise
        
        🚀 Powered by Stable Diffusion, ControlNet, InstantID, and CodeFormer with MobileViT
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
