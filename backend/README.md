# AEMER Backend

This folder contains the Python backend that runs your trained audio emotion recognition model.

## Quick Start

### Windows
Double-click `start_backend.bat` or run:
```cmd
cd backend
start_backend.bat
```

### Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
python main.py
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server - handles API requests |
| `model_handler.py` | Loads PyTorch model, runs predictions |
| `audio_processor.py` | Preprocesses audio (Log-Mel Spectrogram) |
| `requirements.txt` | Python package dependencies |
| `start_backend.bat` | Windows startup script |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Upload audio, get emotion prediction |

## Testing the API

```bash
# Test with curl
curl -X POST -F "file=@your_audio.wav" http://localhost:8000/predict

# Or open http://localhost:8000/docs in your browser for interactive API docs
```

## Model

The backend loads your trained model from:
```
AudioModel/best_model.pth
```

Supported emotions: **Angry**, **Happy**, **Sad**, **Neutral**
