"""
Text Handler for AEMER
Loads the DistilBERT text emotion model and handles predictions.

Emotion Classes:
- 0: angry
- 1: happy
- 2: sad
- 3: neutral
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import os
from typing import Tuple, Dict


# Emotion class mapping
TEXT_EMOTION_LABELS = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "neutral"
}


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


class TextHandler:
    """Handles loading and running the text emotion model."""
    
    MAX_LEN = 128
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load the trained text emotion model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Text model not found: {model_path}")
        
        print(f"Loading text emotion model from: {model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Load model
            self.model = TextEmotionClassifier(num_classes=4)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("Text emotion model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading text model: {e}")
            raise
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict emotion from text.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (predicted_emotion, confidence, all_probabilities)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Text model not loaded.")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1).squeeze()
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
        
        emotion_label = TEXT_EMOTION_LABELS.get(predicted_idx, "unknown")
        all_probs = {
            TEXT_EMOTION_LABELS[i]: float(probabilities[i])
            for i in range(len(TEXT_EMOTION_LABELS))
        }
        
        return emotion_label, confidence, all_probs


# Test
if __name__ == "__main__":
    model_path = "../TextModel/text_model.pth"
    
    if os.path.exists(model_path):
        handler = TextHandler(model_path)
        
        # Test with sample texts
        test_texts = [
            "I'm so happy today!",
            "This is terrible, I'm furious!",
            "I feel so sad and lonely...",
            "The meeting is at 3 PM."
        ]
        
        print("\nTest predictions:")
        for text in test_texts:
            emotion, conf, probs = handler.predict(text)
            print(f"  '{text[:30]}...' → {emotion} ({conf:.2%})")
    else:
        print(f"Model not found: {model_path}")
