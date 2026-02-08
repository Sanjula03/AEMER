"""
Accent Handler for AEMER
Loads the accent detection model and handles predictions.

Accent Classes:
- 0: American
- 1: British
- 2: Canadian
- 3: South Asian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os
from typing import Tuple, Dict


# Accent class mapping
ACCENT_LABELS = {
    0: "american",
    1: "british",
    2: "canadian",
    3: "south_asian"
}


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


class AccentHandler:
    """Handles loading and running the accent detection model."""
    
    # Audio processing parameters (match training)
    SAMPLE_RATE = 16000
    DURATION = 3
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 160
    MAX_LEN = int(SAMPLE_RATE * DURATION / HOP_LENGTH) + 1  # 301
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load the trained accent model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Accent model not found: {model_path}")
        
        print(f"Loading accent model from: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                self.model = CNN_BiLSTM_Accent(num_classes=4)
                self.model.load_state_dict(state_dict)
            else:
                self.model = checkpoint
            
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Accent model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading accent model: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray, sr: int) -> torch.Tensor:
        """Preprocess audio to log-mel spectrogram."""
        # Resample if needed
        if sr != self.SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.SAMPLE_RATE)
        
        # Convert to mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)
        
        # Trim silence
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Pad or trim to fixed length
        target_samples = int(self.DURATION * self.SAMPLE_RATE)
        if len(audio_data) < target_samples:
            audio_data = np.pad(audio_data, (0, target_samples - len(audio_data)))
        else:
            audio_data = audio_data[:target_samples]
        
        # Extract log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=self.SAMPLE_RATE, n_mels=self.N_MELS,
            n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize spectrogram
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        # Pad/trim to fixed length
        if log_mel.shape[1] < self.MAX_LEN:
            log_mel = np.pad(log_mel, ((0, 0), (0, self.MAX_LEN - log_mel.shape[1])))
        else:
            log_mel = log_mel[:, :self.MAX_LEN]
        
        # Convert to tensor
        tensor = torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
        return tensor
    
    def predict(self, features: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """Run accent prediction on preprocessed audio features."""
        if self.model is None:
            raise RuntimeError("Accent model not loaded.")
        
        features = features.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
        
        accent_label = ACCENT_LABELS.get(predicted_idx, "unknown")
        all_probs = {
            ACCENT_LABELS[i]: float(probabilities[0][i])
            for i in range(len(ACCENT_LABELS))
        }
        
        return accent_label, confidence, all_probs
    
    def predict_from_audio(self, audio_data: np.ndarray, sr: int) -> Tuple[str, float, Dict[str, float]]:
        """Predict accent directly from audio data."""
        features = self.preprocess_audio(audio_data, sr)
        return self.predict(features)


# Test
if __name__ == "__main__":
    model_path = "../AccentModel/accent_model.pth"
    
    if os.path.exists(model_path):
        handler = AccentHandler(model_path)
        
        # Test with dummy input
        dummy = torch.randn(1, 1, 128, 301)
        accent, conf, probs = handler.predict(dummy)
        
        print(f"\nTest prediction:")
        print(f"  Accent: {accent}")
        print(f"  Confidence: {conf:.2%}")
        print(f"  All: {probs}")
    else:
        print(f"Model not found: {model_path}")
