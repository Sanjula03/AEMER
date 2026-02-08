"""
AEMER - Accent-aware Multimodal Emotion Recognition
FastAPI Backend for Hugging Face Spaces
With Accent Detection and Text Emotion
"""

import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional

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
        """
        def __init__(self, num_classes=4, dropout=0.3):
            super().__init__()
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(768, 256)
            self.fc2 = nn.Linear(256, num_classes)
            self.relu = nn.ReLU()
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            x = self.dropout(cls_output)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)


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
    
    def process_for_emotion(self, audio_bytes: bytes) -> torch.Tensor:
        """Convert audio bytes to normalized mel spectrogram for emotion model (4s)."""
        audio, sr = self.load_audio(audio_bytes)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Pad or trim to target length (4 seconds for emotion)
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.target_length]
        
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
    ACCENTS = ['american', 'british', 'canadian', 'south_asian']
    
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
            self.accent_model = CNN_BiLSTM_Accent(num_classes=4)
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
    device: str


# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "AEMER - Emotion Recognition API",
        "version": "1.2.0",
        "features": ["emotion_detection", "accent_detection", "text_emotion"],
        "endpoints": {
            "predict": "POST /predict - Upload audio for emotion & accent prediction",
            "predict_text": "POST /predict-text - Text emotion prediction",
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
        
        # Get prediction
        result = model_handler.predict(audio_bytes)
        
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
        result = model_handler.predict_text(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text prediction failed: {str(e)}")


# Startup message
@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("  AEMER API Started!")
    print(f"  Emotion model: {'✅' if model_handler.emotion_model else '❌'}")
    print(f"  Accent model: {'✅' if model_handler.accent_model else '❌'}")
    print(f"  Text model: {'✅' if model_handler.text_model else '❌'}")
    print("=" * 50)
