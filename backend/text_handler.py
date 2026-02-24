"""
Text Handler for AEMER
Uses pre-trained j-hartmann/emotion-english-distilroberta-base model
for 7-emotion text classification.

Emotion Classes:
- angry
- happy (joy)
- sad (sadness)
- neutral
- fear
- surprise
- disgust

No custom model training needed - uses HuggingFace pre-trained model directly.
"""

import os
from typing import Tuple, Dict
from transformers import pipeline


# Map HuggingFace model labels to our consistent label names
HF_LABEL_MAP = {
    "anger": "angry",
    "joy": "happy",
    "sadness": "sad",
    "neutral": "neutral",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust"
}

# Our emotion labels (consistent ordering across all models)
TEXT_EMOTION_LABELS = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "neutral",
    4: "fear",
    5: "surprise",
    6: "disgust"
}


class TextHandler:
    """Handles text emotion prediction using pre-trained HuggingFace model."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the text handler.
        
        Args:
            model_path: Ignored (kept for backward compatibility).
                        Model loads from HuggingFace directly.
        """
        self.classifier = None
        self.load_model()
    
    def load_model(self, model_path: str = None) -> bool:
        """Load the pre-trained HuggingFace emotion model."""
        try:
            print("Loading pre-trained text emotion model from HuggingFace...")
            self.classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,  # Return all emotion scores
                truncation=True,
                model_kwargs={"use_safetensors": True}
            )
            print("✅ Text emotion model loaded successfully (j-hartmann/emotion-english-distilroberta-base)")
            return True
        except Exception as e:
            print(f"❌ Error loading text model: {e}")
            return False
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict emotion from text.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (predicted_emotion, confidence, all_probabilities)
        """
        if self.classifier is None:
            raise RuntimeError("Text model not loaded.")
        
        # Run prediction (returns list of dicts with label + score)
        results = self.classifier(text)[0]
        
        # Map HF labels to our labels and build probabilities dict
        all_probs = {}
        for item in results:
            our_label = HF_LABEL_MAP.get(item["label"], item["label"])
            all_probs[our_label] = float(item["score"])
        
        # Find the top prediction
        top_label = max(all_probs, key=all_probs.get)
        confidence = all_probs[top_label]
        
        return top_label, confidence, all_probs


# Test
if __name__ == "__main__":
    handler = TextHandler()
    
    test_texts = [
        "I'm so happy today!",
        "This is terrible, I'm furious!",
        "I feel so sad and lonely...",
        "The meeting is at 3 PM.",
        "I'm really scared about the exam tomorrow.",
        "Wow, I can't believe that just happened!",
        "That food was absolutely revolting.",
    ]
    
    print("\nTest predictions:")
    for text in test_texts:
        emotion, conf, probs = handler.predict(text)
        print(f"  '{text[:40]}...' → {emotion} ({conf:.2%})")
