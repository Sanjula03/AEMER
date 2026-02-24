"""
Model Handler for AEMER
Loads the PyTorch emotion recognition model and handles predictions.

Emotion Classes:
- 0: Angry
- 1: Happy
- 2: Sad
- 3: Neutral
- 4: Fear
- 5: Surprise
- 6: Disgust
"""

import torch
import torch.nn as nn
import os
from typing import Tuple, Dict


# Emotion class mapping
EMOTION_LABELS = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "neutral",
    4: "fear",
    5: "surprise",
    6: "disgust"
}


class CNN_BiLSTM(nn.Module):
    """
    CNN + Bidirectional LSTM model for audio emotion recognition.
    
    This is the EXACT architecture from your training code.
    Input: Log-Mel Spectrogram (1, 128, time_frames)
    Output: 7 emotion classes (Angry, Happy, Sad, Neutral, Fear, Surprise, Disgust)
    """
    
    def __init__(self, num_classes: int = 7):
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
        import torch.nn.functional as F
        
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


class ModelHandler:
    """Handles loading and running the emotion recognition model."""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load the trained PyTorch model.
        
        Args:
            model_path: Path to the .pth model file
            
        Returns:
            bool: True if loaded successfully
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        try:
            # First, try to load as a complete model (saved with torch.save(model, path))
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Check if it's a state_dict or a full model
            if isinstance(checkpoint, dict):
                # It's a state dict or checkpoint dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume the dict itself is the state_dict
                    state_dict = checkpoint
                
                # Create model and load state dict
                self.model = CNN_BiLSTM(num_classes=7)
                self.model.load_state_dict(state_dict)
            else:
                # It's a full model
                self.model = checkpoint
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nThe model architecture might be different. Please provide the model class definition.")
            raise
    
    def predict(self, features: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """
        Run emotion prediction on preprocessed audio features.
        
        Args:
            features: Preprocessed log-mel spectrogram tensor
            
        Returns:
            Tuple of (predicted_emotion, confidence, all_probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Move features to device
        features = features.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
        
        # Get emotion label
        emotion_label = EMOTION_LABELS.get(predicted_idx, "unknown")
        
        # Get all probabilities
        all_probs = {
            EMOTION_LABELS[i]: float(probabilities[0][i])
            for i in range(len(EMOTION_LABELS))
        }
        
        return emotion_label, confidence, all_probs


# Test the model handler
if __name__ == "__main__":
    import sys
    
    # Test with the actual model
    model_path = "../AudioModel/best_model.pth"
    
    if os.path.exists(model_path):
        try:
            handler = ModelHandler(model_path)
            print("\n✅ Model loaded successfully!")
            
            # Test with dummy input
            dummy_input = torch.randn(1, 1, 128, 401)  # (batch, channel, n_mels, time)
            emotion, conf, probs = handler.predict(dummy_input)
            
            print(f"\nTest prediction (dummy input):")
            print(f"  Emotion: {emotion}")
            print(f"  Confidence: {conf:.2%}")
            print(f"  All probabilities: {probs}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nThe model might have a different architecture.")
            print("Please share your model training code so I can match the architecture.")
    else:
        print(f"Model file not found at: {model_path}")
        print("Please check the path.")
