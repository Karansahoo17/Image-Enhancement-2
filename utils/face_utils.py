"""Face detection and processing utilities"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import torch

def init_detection_model(model_name='retinaface_resnet50', device='cuda'):
    """Initialize face detection model"""
    try:
        face_app = FaceAnalysis(name='antelopev2', root='weights/', 
                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        return face_app
    except Exception as e:
        print(f"Error initializing face detection model: {e}")
        return None

def get_face_landmarks_5(image, face_detector):
    """Get 5-point face landmarks"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if len(image.shape) == 4:  # Batch dimension
        image = image[0]
    
    if image.max() <= 1:  # Normalized image
        image = (image * 255).astype(np.uint8)
    
    # Convert to BGR for face detection
    if len(image.shape) == 3 and image.shape[0] == 3:  # CHW format
        image = np.transpose(image, (1, 2, 0))
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    faces = face_detector.get(image_bgr)
    if faces:
        return faces[0].kps
    else:
        return None

def face_align(image, landmarks):
    """Align face based on landmarks"""
    # Simplified face alignment
    # In practice, you'd want more sophisticated alignment
    return image
