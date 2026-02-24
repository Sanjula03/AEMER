"""
AEMER Backend API Server
FastAPI server that receives audio files and returns emotion predictions.

Endpoints:
- POST /predict: Upload audio file, get emotion prediction
- GET /health: Health check endpoint
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_processor import AudioProcessor
from model_handler import ModelHandler
from accent_handler import AccentHandler
from text_handler import TextHandler
from video_handler import VideoEmotionHandler

# Initialize FastAPI app
app = FastAPI(
    title="AEMER API",
    description="Accent-aware Emotion Recognition API",
    version="1.0.0"
)

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
audio_processor = AudioProcessor()
model_handler = None
accent_handler = None
text_handler = None
video_handler = None

# Path to the trained model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "AudioModel",
    "best_model.pth"
)

# Path to the accent model
ACCENT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "AccentModel",
    "accent_model.pth"
)

# Text emotion model now loads from HuggingFace directly (no .pth file needed)

# Path to the video emotion model
VIDEO_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "VideoModel",
    "video_model.pth"
)


# Response models
class PredictionResponse(BaseModel):
    """Response model for emotion prediction."""
    emotion_label: str
    confidence_score: float
    detected_accent: Optional[str] = None
    audio_weight: float = 1.0
    text_weight: float = 0.0
    visual_weight: float = 0.0
    audio_score: float
    text_score: float = 0.0
    visual_score: float = 0.0
    all_probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    accent_model_loaded: bool
    text_model_loaded: bool
    video_model_loaded: bool
    device: str


class TextPredictionRequest(BaseModel):
    """Request model for text emotion prediction."""
    text: str


class TextPredictionResponse(BaseModel):
    """Response model for text emotion prediction."""
    emotion_label: str
    confidence_score: float
    all_probabilities: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts."""
    global model_handler, accent_handler, text_handler, video_handler
    
    print(f"\n{'='*50}")
    print("AEMER Backend Starting...")
    print(f"{'='*50}")
    
    # Load emotion model
    if os.path.exists(MODEL_PATH):
        try:
            model_handler = ModelHandler(MODEL_PATH)
            print(f"✅ Emotion model loaded from: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load emotion model: {e}")
    else:
        print(f"⚠️ Emotion model not found at: {MODEL_PATH}")
    
    # Load accent model
    if os.path.exists(ACCENT_MODEL_PATH):
        try:
            accent_handler = AccentHandler(ACCENT_MODEL_PATH)
            print(f"✅ Accent model loaded from: {ACCENT_MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load accent model: {e}")
    else:
        print(f"⚠️ Accent model not found at: {ACCENT_MODEL_PATH}")
    
    # Load text emotion model (from HuggingFace pre-trained)
    try:
        text_handler = TextHandler()
        print(f"✅ Text emotion model loaded (HuggingFace pre-trained)")
    except Exception as e:
        print(f"❌ Failed to load text model: {e}")
    
    # Load video emotion model
    if os.path.exists(VIDEO_MODEL_PATH):
        try:
            video_handler = VideoEmotionHandler(VIDEO_MODEL_PATH)
            print(f"✅ Video emotion model loaded from: {VIDEO_MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load video model: {e}")
    else:
        print(f"⚠️ Video model not found at: {VIDEO_MODEL_PATH}")
    
    print(f"\n{'='*50}")
    print("Server ready! API docs at: http://localhost:8000/docs")
    print(f"{'='*50}\n")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "AEMER API",
        "version": "1.0.0",
        "description": "Accent-aware Emotion Recognition API",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check endpoint."""
    import torch
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_handler is not None and model_handler.model is not None,
        accent_model_loaded=accent_handler is not None and accent_handler.model is not None,
        text_model_loaded=text_handler is not None and text_handler.classifier is not None,
        video_model_loaded=video_handler is not None and video_handler.model is not None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_emotion(file: UploadFile = File(...)):
    """
    Analyze audio file and predict emotion.
    
    Args:
        file: Audio file (WAV, MP3, OGG, etc.)
        
    Returns:
        PredictionResponse with emotion label and confidence scores
    """
    # Validate model is loaded
    if model_handler is None or model_handler.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file type
    allowed_types = [
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave",
        "audio/x-wav", "audio/ogg", "audio/flac", "audio/m4a",
        "audio/x-m4a", "audio/mp4"
    ]
    
    # Also allow by extension for browsers that don't send correct MIME types
    allowed_extensions = [".mp3", ".wav", ".ogg", ".flac", ".m4a", ".mp4"]
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    
    if file.content_type not in allowed_types and file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: MP3, WAV, OGG, FLAC, M4A. Got: {file.content_type}"
        )
    
    try:
        # Read file bytes
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        print(f"\n📁 Processing: {file.filename} ({len(audio_bytes)} bytes)")
        
        # Preprocess audio (with noise reduction)
        features, snr_db = audio_processor.process(audio_bytes, file.filename or "audio.wav")
        print(f"   Features shape: {features.shape}, SNR: {snr_db:.1f} dB")
        
        # Run emotion prediction
        emotion_label, confidence, all_probs = model_handler.predict(features)
        print(f"   Emotion: {emotion_label} ({confidence:.2%})")
        
        # Run accent detection
        detected_accent = None
        if accent_handler is not None and accent_handler.model is not None:
            try:
                # Get raw audio for accent detection
                audio_data, sr = audio_processor.load_audio(audio_bytes, file.filename or "audio.wav")
                accent_label, accent_conf, accent_probs = accent_handler.predict_from_audio(audio_data, sr)
                detected_accent = accent_label
                print(f"   Accent: {accent_label} ({accent_conf:.2%})")
            except Exception as e:
                print(f"   Accent detection failed: {e}")
        
        # Return response
        return PredictionResponse(
            emotion_label=emotion_label,
            confidence_score=confidence,
            detected_accent=detected_accent,
            audio_weight=1.0,
            text_weight=0.0,
            visual_weight=0.0,
            audio_score=confidence,
            text_score=0.0,
            visual_score=0.0,
            all_probabilities=all_probs
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )


@app.post("/predict-text", response_model=TextPredictionResponse, tags=["Prediction"])
async def predict_text_emotion(request: TextPredictionRequest):
    """
    Predict emotion from text input.
    
    Args:
        request: JSON with 'text' field
        
    Returns:
        TextPredictionResponse with emotion prediction
    """
    if text_handler is None or text_handler.classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Text emotion model not loaded."
        )
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )
    
    try:
        print(f"\n📝 Processing text: '{request.text[:50]}...'")
        emotion, confidence, all_probs = text_handler.predict(request.text)
        print(f"   Text emotion: {emotion} ({confidence:.2%})")
        
        return TextPredictionResponse(
            emotion_label=emotion,
            confidence_score=confidence,
            all_probabilities=all_probs
        )
        
    except Exception as e:
        print(f"❌ Error processing text: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text: {str(e)}"
        )


@app.post("/predict-video", tags=["Prediction"])
async def predict_video_emotion(file: UploadFile = File(...)):
    """
    Analyze video file and predict facial emotion.
    
    Args:
        file: Video file (MP4, AVI, MOV, etc.) or image file (JPG, PNG)
        
    Returns:
        Emotion prediction with confidence scores
    """
    # Validate model is loaded
    if video_handler is None or video_handler.model is None:
        raise HTTPException(
            status_code=503,
            detail="Video model not loaded. Please check server logs."
        )
    
    # Validate file type
    allowed_video_ext = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    allowed_image_ext = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    
    file_ext = ""
    if file.filename:
        file_ext = "." + file.filename.lower().split(".")[-1]
    
    is_video = file_ext in allowed_video_ext
    is_image = file_ext in allowed_image_ext
    
    if not is_video and not is_image:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_video_ext + allowed_image_ext}"
        )
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        print(f"\n📹 Processing: {file.filename} ({len(file_bytes)} bytes)")
        
        if is_video:
            result = video_handler.process_video(file_bytes)
        else:
            result = video_handler.process_image(file_bytes)
        
        print(f"   Emotion: {result.get('emotion')} ({result.get('confidence', 0):.2%})")
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )


# Run server
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting AEMER Backend Server...")
    print("="*60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
