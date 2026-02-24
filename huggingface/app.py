"""
AEMER - Accent-aware Multimodal Emotion Recognition
FastAPI Backend for Hugging Face Spaces
With Accent Detection, Text Emotion, and Video Emotion
"""

import io
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, sosfilt
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List, Tuple

# Conditional imports for video processing
try:
    import cv2
    from PIL import Image
    import torchvision.models as models
    import torchvision.transforms as transforms
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ opencv/torchvision not installed - video emotion disabled")

# Conditional import - transformers may not be available
try:
    from transformers import DistilBertTokenizer, DistilBertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not installed - text emotion disabled")


# ============================================
# EMOTION MODEL ARCHITECTURE (AUDIO)
# ============================================

class CNN_BiLSTM(nn.Module):
    """
    CNN + Bidirectional LSTM model for audio emotion recognition.
    Input: Log-Mel Spectrogram (1, 128, time_frames)
    Output: 4 emotion classes (Angry, Happy, Sad, Neutral)
    """
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            input_size=64 * 32,
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop(x)

        # Reshape for LSTM: (batch, time, features)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last timestep
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


# ============================================
# ACCENT MODEL ARCHITECTURE
# ============================================

class CNN_BiLSTM_Accent(nn.Module):
    """
    CNN-BiLSTM for Accent Detection.
    Same architecture as training notebook.
    """
    
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.3))
        self.lstm = nn.LSTM(128*16, 128, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes))
        
    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.permute(0, 3, 1, 2).reshape(x.size(0), -1, 128*16)
        lstm_out, _ = self.lstm(x)
        attn = F.softmax(self.attention(lstm_out), dim=1)
        return self.fc(torch.sum(attn * lstm_out, dim=1))


# ============================================
# TEXT EMOTION MODEL ARCHITECTURE
# ============================================

if TRANSFORMERS_AVAILABLE:
    class TextEmotionClassifier(nn.Module):
        """
        DistilBERT-based text emotion classifier.
        Same architecture as training notebook.
        Uses a deeper classifier head with BatchNorm for better generalization.
        """
        def __init__(self, num_classes=4, dropout=0.3):
            super().__init__()
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.dropout = nn.Dropout(dropout)
            
            # Deeper classifier head (must match training notebook)
            self.classifier = nn.Sequential(
                nn.Linear(768, 384),
                nn.BatchNorm1d(384),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(384, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            x = self.dropout(cls_output)
            return self.classifier(x)


# ============================================
# VIDEO EMOTION MODEL ARCHITECTURE
# ============================================

if CV2_AVAILABLE:
    class FacialEmotionResNet(nn.Module):
        """ResNet-18 based facial emotion classifier."""
        
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
        
        EMOTIONS = ['angry', 'happy', 'sad', 'neutral']
        
        def __init__(self, model_path: str = "video_model.pth"):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.face_cascade = None
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.load_model(model_path)
            self.load_face_detector()
        
        def load_model(self, model_path: str) -> bool:
            if not os.path.exists(model_path):
                print(f"⚠️ Video model not found at {model_path}")
                return False
            
            try:
                self.model = FacialEmotionResNet(num_classes=4)
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ Video model loaded from {model_path}")
                return True
            except Exception as e:
                print(f"❌ Error loading video model: {e}")
                return False
        
        def load_face_detector(self):
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print("✅ Face detector loaded")
            except Exception as e:
                print(f"⚠️ Face detector error: {e}")
        
        def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
            if self.face_cascade is None:
                return []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
            return [(x, y, w, h) for (x, y, w, h) in faces]
        
        def predict_emotion(self, face_img: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
            if self.model is None:
                return 'neutral', 0.25, {e: 0.25 for e in self.EMOTIONS}
            
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                idx = probs.argmax().item()
                conf = probs[idx].item()
            
            return self.EMOTIONS[idx], conf, {self.EMOTIONS[i]: probs[i].item() for i in range(4)}
        
        def check_blur(self, frame: np.ndarray) -> Tuple[bool, float]:
            """Check if image is blurry using Laplacian variance."""
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Lower variance = more blurry. Threshold of 100 works well.
            is_blurry = laplacian_var < 100
            return is_blurry, laplacian_var
        
        def check_brightness(self, frame: np.ndarray) -> Tuple[bool, bool, float]:
            """Check if image is too dark or too bright."""
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            is_dark = avg_brightness < 50
            is_bright = avg_brightness > 200
            return is_dark, is_bright, avg_brightness
        
        def process_image(self, image_bytes: bytes) -> Dict:
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {'emotion': 'neutral', 'confidence': 0.0, 'error': 'Failed to decode image'}
            
            # Check image quality
            is_blurry, blur_score = self.check_blur(frame)
            is_dark, is_bright, brightness = self.check_brightness(frame)
            
            quality_issues = []
            if is_blurry:
                quality_issues.append(f"Image is blurry (sharpness: {blur_score:.0f})")
            if is_dark:
                quality_issues.append("Image is too dark")
            if is_bright:
                quality_issues.append("Image is too bright/overexposed")
            
            faces = self.detect_faces(frame)
            
            if faces:
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                emotion, conf, all_probs = self.predict_emotion(face_img)
                result = {
                    'emotion': emotion,
                    'confidence': conf,
                    'all_probabilities': all_probs,
                    'faces_detected': len(faces),
                    'image_quality': {
                        'blur_score': blur_score,
                        'brightness': brightness
                    }
                }
                if quality_issues:
                    result['quality_warning'] = "; ".join(quality_issues)
                return result
            else:
                quality_issues.append("No face detected - try a clearer image")
                return {
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'all_probabilities': {e: 0.25 for e in self.EMOTIONS},
                    'faces_detected': 0,
                    'quality_warning': "; ".join(quality_issues)
                }
        
        def process_video(self, video_bytes: bytes, sample_rate: int = 1) -> Dict:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name
            
            try:
                cap = cv2.VideoCapture(tmp_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = max(1, int(fps / sample_rate)) if fps > 0 else 30
                
                all_emotions = []
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        faces = self.detect_faces(frame)
                        for (x, y, w, h) in faces:
                            face_img = frame[y:y+h, x:x+w]
                            emotion, _, _ = self.predict_emotion(face_img)
                            all_emotions.append(emotion)
                    
                    frame_count += 1
                
                cap.release()
                
                if all_emotions:
                    from collections import Counter
                    counts = Counter(all_emotions)
                    dominant = counts.most_common(1)[0][0]
                    conf = counts[dominant] / len(all_emotions)
                else:
                    dominant, conf = 'neutral', 0.0
                
                return {
                    'emotion': dominant,
                    'confidence': conf,
                    'faces_detected': len(all_emotions)
                }
            finally:
                os.unlink(tmp_path)


# ============================================
# AUDIO PROCESSOR
# ============================================

class AudioProcessor:
    """Process audio files into mel spectrograms for the models."""
    
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, duration: float = 4.0):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration = duration
        self.target_length = int(sample_rate * duration)
    
    def load_audio(self, audio_bytes: bytes) -> tuple:
        """Load audio from bytes and return (audio_array, sample_rate)."""
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate, mono=True)
        return audio, sr
    
    def check_clipping(self, audio: np.ndarray) -> Tuple[bool, float]:
        """Check if audio has clipping (samples at max amplitude).
        Returns (is_clipped, clipping_percentage)."""
        clipped_samples = np.sum(np.abs(audio) > 0.99)
        clipping_pct = (clipped_samples / len(audio)) * 100
        is_clipped = clipping_pct > 5  # More than 5% clipped
        return is_clipped, clipping_pct
    
    def process_for_emotion(self, audio_bytes: bytes) -> torch.Tensor:
        """Convert audio bytes to normalized mel spectrogram for emotion model (4s).
        Includes noise reduction and bandpass filtering."""
        audio, sr = self.load_audio(audio_bytes)
        
        # Step 1: Noise reduction (spectral gating)
        try:
            audio = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)
        except Exception:
            pass  # Keep original if noise reduction fails
        
        # Step 2: Bandpass filter for speech (80Hz - 3000Hz)
        try:
            nyquist = sr / 2
            low, high = 80 / nyquist, 3000 / nyquist
            if 0 < low < 1 and 0 < high < 1:
                sos = butter(5, [low, high], btype='band', output='sos')
                audio = sosfilt(sos, audio).astype(np.float32)
        except Exception:
            pass  # Keep original if filtering fails
        
        # Step 3: Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Step 4: Pad or trim to target length (4 seconds for emotion)
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.target_length]
        
        # Step 5: Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels,
            n_fft=1024, hop_length=160, win_length=400, fmax=8000
        )
        
        # Convert to log scale and normalize
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        return torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0)
    
    def process_for_accent(self, audio_bytes: bytes) -> torch.Tensor:
        """Convert audio bytes to mel spectrogram for accent model (3s, 301 frames)."""
        audio, sr = self.load_audio(audio_bytes)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # Pad or trim to 3 seconds
        accent_length = int(3 * self.sample_rate)
        if len(audio) < accent_length:
            audio = np.pad(audio, (0, accent_length - len(audio)))
        else:
            audio = audio[:accent_length]
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=128,
            n_fft=1024, hop_length=160
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        # Pad/trim to 301 frames
        max_len = 301
        if log_mel.shape[1] < max_len:
            log_mel = np.pad(log_mel, ((0, 0), (0, max_len - log_mel.shape[1])))
        else:
            log_mel = log_mel[:, :max_len]
        
        return torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0)


# ============================================
# MODEL HANDLER
# ============================================

class ModelHandler:
    """Handle model loading and predictions."""
    
    EMOTIONS = ['angry', 'happy', 'sad', 'neutral']
    ACCENTS = [
        'american', 'british', 'australian', 'indian', 'canadian',
        'irish', 'african', 'filipino', 'hongkong'
    ]
    
    def __init__(self, emotion_model_path: str = "best_model.pth", 
                 accent_model_path: str = "accent_model.pth",
                 text_model_path: str = "text_model.pth"):
        self.device = torch.device("cpu")
        self.emotion_model = None
        self.accent_model = None
        self.text_model = None
        self.tokenizer = None
        self.audio_processor = AudioProcessor()
        
        self.load_emotion_model(emotion_model_path)
        self.load_accent_model(accent_model_path)
        self.load_text_model(text_model_path)
    
    def load_emotion_model(self, model_path: str) -> bool:
        """Load the emotion model."""
        if not os.path.exists(model_path):
            print(f"Warning: Emotion model not found at {model_path}")
            return False
        
        try:
            self.emotion_model = CNN_BiLSTM(num_classes=4)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.emotion_model.load_state_dict(state_dict)
            self.emotion_model.to(self.device)
            self.emotion_model.eval()
            print(f"✅ Emotion model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading emotion model: {e}")
            return False
    
    def load_accent_model(self, model_path: str) -> bool:
        """Load the accent model."""
        if not os.path.exists(model_path):
            print(f"Warning: Accent model not found at {model_path}")
            return False
        
        try:
            self.accent_model = CNN_BiLSTM_Accent(num_classes=len(self.ACCENTS))
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.accent_model.load_state_dict(state_dict)
            self.accent_model.to(self.device)
            self.accent_model.eval()
            print(f"✅ Accent model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading accent model: {e}")
            return False
    
    def load_text_model(self, model_path: str) -> bool:
        """Load the text emotion model."""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ transformers not available - text model disabled")
            return False
            
        if not os.path.exists(model_path):
            print(f"Warning: Text model not found at {model_path}")
            return False
        
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.text_model = TextEmotionClassifier(num_classes=4)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.text_model.load_state_dict(state_dict)
            self.text_model.to(self.device)
            self.text_model.eval()
            print(f"✅ Text emotion model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading text model: {e}")
            return False
    
    def predict(self, audio_bytes: bytes) -> Dict:
        """Make emotion and accent predictions from audio bytes."""
        result = {
            "emotion_label": "unknown",
            "confidence_score": 0.0,
            "detected_accent": None,
            "audio_weight": 1.0,
            "text_weight": 0.0,
            "visual_weight": 0.0,
            "audio_score": 0.0,
            "text_score": 0.0,
            "visual_score": 0.0,
            "all_probabilities": {}
        }
        
        # Emotion prediction
        if self.emotion_model is not None:
            spectrogram = self.audio_processor.process_for_emotion(audio_bytes)
            spectrogram = spectrogram.to(self.device)
            
            with torch.no_grad():
                output = self.emotion_model(spectrogram)
                probabilities = F.softmax(output, dim=1).squeeze()
            
            probs_np = probabilities.cpu().numpy()
            emotion_idx = int(np.argmax(probs_np))
            
            result["emotion_label"] = self.EMOTIONS[emotion_idx]
            result["confidence_score"] = float(probs_np[emotion_idx])
            result["audio_score"] = float(probs_np[emotion_idx])
            result["all_probabilities"] = {
                emotion: float(probs_np[i]) 
                for i, emotion in enumerate(self.EMOTIONS)
            }
        
        # Accent prediction
        if self.accent_model is not None:
            try:
                accent_spec = self.audio_processor.process_for_accent(audio_bytes)
                accent_spec = accent_spec.to(self.device)
                
                with torch.no_grad():
                    accent_output = self.accent_model(accent_spec)
                    accent_probs = F.softmax(accent_output, dim=1).squeeze()
                
                accent_probs_np = accent_probs.cpu().numpy()
                accent_idx = int(np.argmax(accent_probs_np))
                result["detected_accent"] = self.ACCENTS[accent_idx]
            except Exception as e:
                print(f"Accent detection error: {e}")
        
        return result
    
    def predict_text(self, text: str) -> Dict:
        """Predict emotion from text."""
        result = {
            "emotion_label": "unknown",
            "confidence_score": 0.0,
            "all_probabilities": {}
        }
        
        if self.text_model is None or self.tokenizer is None:
            return result
        
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.text_model(input_ids, attention_mask)
                probabilities = F.softmax(outputs, dim=1).squeeze()
            
            probs_np = probabilities.cpu().numpy()
            emotion_idx = int(np.argmax(probs_np))
            
            result["emotion_label"] = self.EMOTIONS[emotion_idx]
            result["confidence_score"] = float(probs_np[emotion_idx])
            result["all_probabilities"] = {
                emotion: float(probs_np[i])
                for i, emotion in enumerate(self.EMOTIONS)
            }
        except Exception as e:
            print(f"Text prediction error: {e}")
        
        return result


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="AEMER API",
    description="Accent-aware Multimodal Emotion Recognition",
    version="1.2.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model handler
model_handler = ModelHandler()

# Initialize video handler
video_handler = None
if CV2_AVAILABLE:
    video_handler = VideoEmotionHandler("video_model.pth")


# ============================================
# RESPONSE MODELS
# ============================================

class PredictionResponse(BaseModel):
    emotion_label: str
    confidence_score: float
    detected_accent: Optional[str] = None
    audio_weight: float = 1.0
    text_weight: float = 0.0
    visual_weight: float = 0.0
    audio_score: float = 0.0
    text_score: float = 0.0
    visual_score: float = 0.0
    all_probabilities: Dict[str, float]


class TextPredictionRequest(BaseModel):
    text: str


class TextPredictionResponse(BaseModel):
    emotion_label: str
    confidence_score: float
    all_probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    accent_model_loaded: bool
    text_model_loaded: bool
    video_model_loaded: bool
    device: str


class ModalityResult(BaseModel):
    emotion_label: str
    confidence_score: float
    all_probabilities: Dict[str, float]
    weight: float = 0.0


class MultimodalPredictionResponse(BaseModel):
    emotion_label: str
    confidence_score: float
    all_probabilities: Dict[str, float]
    fusion_method: str = "adaptive_weighted"
    modalities_used: List[str] = []
    audio_result: Optional[ModalityResult] = None
    text_result: Optional[ModalityResult] = None
    video_result: Optional[ModalityResult] = None
    quality_warning: Optional[str] = None

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "AEMER - Emotion Recognition API",
        "version": "1.3.0",
        "features": ["emotion_detection", "accent_detection", "text_emotion", "video_emotion"],
        "endpoints": {
            "predict": "POST /predict - Upload audio for emotion & accent prediction",
            "predict_text": "POST /predict-text - Text emotion prediction",
            "predict_video": "POST /predict-video - Video/image emotion prediction",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return {
        "status": "healthy",
        "model_loaded": model_handler.emotion_model is not None,
        "accent_model_loaded": model_handler.accent_model is not None,
        "text_model_loaded": model_handler.text_model is not None,
        "video_model_loaded": video_handler is not None and video_handler.model is not None,
        "device": str(model_handler.device)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict emotion and accent from uploaded audio file.
    
    Supported formats: WAV, MP3, OGG, FLAC
    Returns: Detected emotion, accent, confidence score, and all probabilities
    """
    if model_handler.emotion_model is None:
        raise HTTPException(status_code=503, detail="Emotion model not loaded")
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Check for audio clipping before processing
        warnings = []
        try:
            raw_audio, _ = model_handler.audio_processor.load_audio(audio_bytes)
            is_clipped, clip_pct = model_handler.audio_processor.check_clipping(raw_audio)
            if is_clipped:
                warnings.append(f"Audio clipping detected ({clip_pct:.1f}% clipped) - may affect accuracy")
        except:
            pass  # Continue even if clipping check fails
        
        # Get prediction
        result = model_handler.predict(audio_bytes)
        
        # Add quality warning for low confidence
        if result.get("confidence_score", 1.0) < 0.5:
            warnings.append("Low confidence - prediction may be inaccurate")
        
        if warnings:
            result["quality_warning"] = "; ".join(warnings)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-text", response_model=TextPredictionResponse)
async def predict_text(request: TextPredictionRequest):
    """
    Predict emotion from text input.
    
    Args:
        request: JSON body with 'text' field
        
    Returns: Detected emotion, confidence score, and all probabilities
    """
    if model_handler.text_model is None:
        raise HTTPException(status_code=503, detail="Text emotion model not loaded")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Check for short text warning
        warnings = []
        word_count = len(request.text.strip().split())
        if word_count < 3:
            warnings.append("Text is very short - prediction may be less accurate")
        
        result = model_handler.predict_text(request.text)
        
        # Add low confidence warning
        if result.get("confidence_score", 1.0) < 0.5:
            warnings.append("Low confidence - prediction may be inaccurate")
        
        if warnings:
            result["quality_warning"] = "; ".join(warnings)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text prediction failed: {str(e)}")


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    """
    Predict emotion from video or image file.
    
    Supported formats: MP4, AVI, MOV, JPG, PNG
    Returns: Detected emotion and confidence score
    """
    if video_handler is None or video_handler.model is None:
        raise HTTPException(status_code=503, detail="Video emotion model not loaded")
    
    # Validate file type
    file_ext = ""
    if file.filename:
        file_ext = "." + file.filename.lower().split(".")[-1]
    
    allowed_video = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    allowed_image = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    
    is_video = file_ext in allowed_video
    is_image = file_ext in allowed_image
    
    if not is_video and not is_image:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_video + allowed_image}"
        )
    
    try:
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if is_video:
            result = video_handler.process_video(file_bytes)
        else:
            result = video_handler.process_image(file_bytes)
        
        # Add quality warnings
        warnings = []
        faces = result.get("faces_detected", 0)
        if faces == 0:
            warnings.append("No face detected - try a clearer image with visible face")
        elif faces > 1:
            warnings.append(f"Multiple faces detected ({faces}) - analyzing primary face only")
        if result.get("confidence", 1.0) < 0.5:
            warnings.append("Low confidence - prediction may be inaccurate")
        
        if warnings:
            result["quality_warning"] = "; ".join(warnings)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video prediction failed: {str(e)}")


@app.post("/predict-multimodal", response_model=MultimodalPredictionResponse)
async def predict_multimodal(
    audio_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """
    Multimodal emotion prediction: combine audio, text, and video/image inputs.
    
    At least 2 modalities must be provided.
    Returns fused prediction with per-modality breakdowns.
    """
    modality_results = {}
    warnings = []
    
    # --- Process Audio ---
    if audio_file and audio_file.filename:
        try:
            audio_bytes = await audio_file.read()
            if len(audio_bytes) > 0 and model_handler.emotion_model is not None:
                # Clipping check
                try:
                    raw_audio, _ = model_handler.audio_processor.load_audio(audio_bytes)
                    is_clipped, clip_pct = model_handler.audio_processor.check_clipping(raw_audio)
                    if is_clipped:
                        warnings.append(f"Audio clipping detected ({clip_pct:.1f}%)")
                except:
                    pass
                
                result = model_handler.predict(audio_bytes)
                modality_results["audio"] = {
                    "emotion_label": result["emotion_label"],
                    "confidence_score": result["confidence_score"],
                    "all_probabilities": result["all_probabilities"],
                }
        except Exception as e:
            warnings.append(f"Audio processing failed: {str(e)}")
    
    # --- Process Text ---
    if text and text.strip():
        try:
            if model_handler.text_model is not None:
                word_count = len(text.strip().split())
                if word_count < 3:
                    warnings.append("Text is very short")
                
                result = model_handler.predict_text(text)
                modality_results["text"] = {
                    "emotion_label": result["emotion_label"],
                    "confidence_score": result["confidence_score"],
                    "all_probabilities": result["all_probabilities"],
                }
        except Exception as e:
            warnings.append(f"Text processing failed: {str(e)}")
    
    # --- Process Video/Image ---
    if video_file and video_file.filename:
        try:
            if video_handler is not None and video_handler.model is not None:
                video_bytes = await video_file.read()
                if len(video_bytes) > 0:
                    file_ext = video_file.filename.rsplit(".", 1)[-1].lower() if video_file.filename else ""
                    is_video = file_ext in ["mp4", "avi", "mov", "mkv", "webm"]
                    
                    if is_video:
                        result = video_handler.process_video(video_bytes)
                    else:
                        result = video_handler.process_image(video_bytes)
                    
                    faces = result.get("faces_detected", 0)
                    if faces == 0:
                        warnings.append("No face detected in image")
                    elif faces > 1:
                        warnings.append(f"Multiple faces detected ({faces})")
                    
                    modality_results["video"] = {
                        "emotion_label": result["emotion"],
                        "confidence_score": result["confidence"],
                        "all_probabilities": result.get("all_probabilities", {}),
                    }
        except Exception as e:
            warnings.append(f"Video processing failed: {str(e)}")
    
    # --- Validate: at least 2 modalities ---
    if len(modality_results) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"At least 2 modalities required, got {len(modality_results)}: {list(modality_results.keys())}"
        )
    
    # --- Adaptive Weighted Fusion ---
    # Base weights for each modality
    base_weights = {"audio": 0.4, "text": 0.2, "video": 0.4}
    emotions = ["angry", "happy", "sad", "neutral"]
    
    fused_probs = {e: 0.0 for e in emotions}
    total_weight = 0.0
    
    for modality, res in modality_results.items():
        # Adaptive weight = base_weight * confidence (higher confidence = more influence)
        confidence = res["confidence_score"]
        adaptive_weight = base_weights.get(modality, 0.33) * confidence
        total_weight += adaptive_weight
        res["weight"] = adaptive_weight
        
        for emotion in emotions:
            fused_probs[emotion] += res["all_probabilities"].get(emotion, 0.0) * adaptive_weight
    
    # Normalize
    if total_weight > 0:
        for emotion in emotions:
            fused_probs[emotion] /= total_weight
    
    # Get final prediction
    final_emotion = max(fused_probs, key=lambda e: fused_probs[e])
    final_confidence = fused_probs[final_emotion]
    
    # Normalize weights for display
    for modality, res in modality_results.items():
        res["weight"] = res["weight"] / total_weight if total_weight > 0 else 0.0
    
    # Low confidence warning
    if final_confidence < 0.5:
        warnings.append("Low fused confidence - prediction may be inaccurate")
    
    # Build response
    response = {
        "emotion_label": final_emotion,
        "confidence_score": final_confidence,
        "all_probabilities": fused_probs,
        "fusion_method": "adaptive_weighted",
        "modalities_used": list(modality_results.keys()),
    }
    
    if "audio" in modality_results:
        response["audio_result"] = modality_results["audio"]
    if "text" in modality_results:
        response["text_result"] = modality_results["text"]
    if "video" in modality_results:
        response["video_result"] = modality_results["video"]
    
    if warnings:
        response["quality_warning"] = "; ".join(warnings)
    
    return response


# Startup message
@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("  AEMER API Started!")
    print(f"  Emotion model: {'✅' if model_handler.emotion_model else '❌'}")
    print(f"  Accent model: {'✅' if model_handler.accent_model else '❌'}")
    print(f"  Text model: {'✅' if model_handler.text_model else '❌'}")
    print(f"  Video model: {'✅' if video_handler and video_handler.model else '❌'}")
    print(f"  Multimodal fusion: ✅")
    print("=" * 50)

