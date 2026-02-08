# 🚀 AEMER Hugging Face Deployment Guide

## Quick Start

### 1. Create Hugging Face Space
1. Go to [huggingface.co](https://huggingface.co) and sign up/login
2. Click profile → **New Space**
3. Name: `aemer`, SDK: **Docker**, then Create

### 2. Upload Files
Upload these files to your Space:

| File | Source Location |
|------|-----------------|
| `app.py` | huggingface/app.py |
| `Dockerfile` | huggingface/Dockerfile |
| `requirements.txt` | huggingface/requirements.txt |
| `README.md` | huggingface/README.md |
| `best_model.pth` | AudioModel/best_model.pth |
| `accent_model.pth` | AccentModel/accent_model.pth |
| `text_model.pth` | TextModel/text_model.pth ✨ **NEW** |

### 3. Wait for Build
- Takes 10-15 minutes (text model adds ~2min)
- Check Logs tab if issues

### 4. Your API URL
```
https://YOUR_USERNAME-aemer.hf.space
```

**Test:** Go to `https://YOUR_USERNAME-aemer.hf.space/docs`

---

## API Endpoints

### Audio Prediction: `POST /predict`
Upload audio file, get emotion + accent

### Text Prediction: `POST /predict-text` (NEW!)
```json
{"text": "I'm so happy today!"}
```
Returns:
```json
{
  "emotion_label": "happy",
  "confidence_score": 0.98,
  "all_probabilities": {...}
}
```

### Health Check: `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "accent_model_loaded": true,
  "text_model_loaded": true
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check Logs tab |
| Model not found | Upload all 3 `.pth` files |
| Text not working | Check `text_model.pth` uploaded |
| Slow first request | Cold start - wait 30s |
