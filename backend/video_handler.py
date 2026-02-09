"""
Video Emotion Handler for AEMER
Handles face detection and emotion prediction from video frames.
Uses ResNet-18 transfer learning model trained on FER2013.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tempfile
from typing import List, Tuple, Dict, Optional


# Emotion labels (same order as training)
EMOTIONS = ['angry', 'happy', 'sad', 'neutral']


class FacialEmotionResNet(nn.Module):
    """ResNet-18 based facial emotion classifier (same as training)."""
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


class VideoEmotionHandler:
    """Handles video emotion detection with face detection."""
    
    def __init__(self, model_path: str = "VideoModel/video_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.face_cascade = None
        
        # Image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self.load_model(model_path)
        
        # Load face detector
        self.load_face_detector()
    
    def load_model(self, model_path: str) -> bool:
        """Load the trained video emotion model."""
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ Video model not found at {model_path}")
                return False
            
            self.model = FacialEmotionResNet(num_classes=4)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Video model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading video model: {e}")
            self.model = None
            return False
    
    def load_face_detector(self):
        """Load OpenCV Haar cascade for face detection."""
        try:
            # Try to find the cascade file
            cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_default.xml'
            ]
            
            for path in cascade_paths:
                if os.path.exists(path):
                    self.face_cascade = cv2.CascadeClassifier(path)
                    print(f"✅ Face detector loaded")
                    return True
            
            # Fallback: use OpenCV's built-in
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print(f"✅ Face detector loaded (default)")
            return True
        except Exception as e:
            print(f"⚠️ Face detector warning: {e}")
            return False
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame using Haar cascade."""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """Preprocess a face crop for the model."""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(face_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_img).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict_emotion(self, face_tensor: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """Predict emotion from preprocessed face tensor."""
        if self.model is None:
            return 'neutral', 0.25, {e: 0.25 for e in EMOTIONS}
        
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = probabilities.argmax().item()
            confidence = probabilities[predicted_idx].item()
        
        emotion = EMOTIONS[predicted_idx]
        all_probs = {EMOTIONS[i]: probabilities[i].item() for i in range(len(EMOTIONS))}
        
        return emotion, confidence, all_probs
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Process a single frame and return emotions for all detected faces."""
        results = []
        
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Preprocess and predict
            face_tensor = self.preprocess_face(face_img)
            emotion, confidence, all_probs = self.predict_emotion(face_tensor)
            
            results.append({
                'bbox': [x, y, w, h],
                'emotion': emotion,
                'confidence': confidence,
                'all_probabilities': all_probs
            })
        
        return results
    
    def process_video(self, video_bytes: bytes, sample_rate: int = 1) -> Dict:
        """
        Process a video file and return aggregated emotion results.
        
        Args:
            video_bytes: Raw video file bytes
            sample_rate: Sample 1 frame per second (default)
        
        Returns:
            Dict with dominant emotion, confidence, and frame-by-frame results
        """
        # Save video to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        try:
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_interval = int(fps / sample_rate) if fps > 0 else 30
            
            all_emotions = []
            frame_results = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at specified rate
                if frame_count % frame_interval == 0:
                    results = self.process_frame(frame)
                    
                    for result in results:
                        all_emotions.append(result['emotion'])
                        frame_results.append({
                            'frame': frame_count,
                            'time': frame_count / fps if fps > 0 else 0,
                            **result
                        })
                
                frame_count += 1
            
            cap.release()
            
            # Aggregate results
            if all_emotions:
                from collections import Counter
                emotion_counts = Counter(all_emotions)
                dominant_emotion = emotion_counts.most_common(1)[0][0]
                confidence = emotion_counts[dominant_emotion] / len(all_emotions)
            else:
                dominant_emotion = 'neutral'
                confidence = 0.0
            
            return {
                'emotion': dominant_emotion,
                'confidence': confidence,
                'total_frames': total_frames,
                'analyzed_frames': len(frame_results),
                'faces_detected': len(all_emotions),
                'frame_results': frame_results[:10]  # Return first 10 for brevity
            }
        
        finally:
            os.unlink(tmp_path)
    
    def process_image(self, image_bytes: bytes) -> Dict:
        """Process a single image and return emotion results."""
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {'error': 'Failed to decode image'}
        
        results = self.process_frame(frame)
        
        if results:
            # Return the first face's emotion
            return {
                'emotion': results[0]['emotion'],
                'confidence': results[0]['confidence'],
                'all_probabilities': results[0]['all_probabilities'],
                'faces_detected': len(results)
            }
        else:
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_probabilities': {e: 0.25 for e in EMOTIONS},
                'faces_detected': 0,
                'warning': 'No face detected in image'
            }


# Test
if __name__ == "__main__":
    handler = VideoEmotionHandler()
    print(f"Model loaded: {handler.model is not None}")
    print(f"Face detector loaded: {handler.face_cascade is not None}")
