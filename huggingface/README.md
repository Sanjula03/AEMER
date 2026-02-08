---
title: AEMER - Emotion Recognition
emoji: 🎭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# AEMER - Accent-aware Multimodal Emotion Recognition

Upload an audio file to detect emotions (Angry, Happy, Sad, Neutral).

## API Endpoints

- `POST /predict` - Upload audio file for emotion prediction
- `GET /health` - Health check
- `GET /docs` - API documentation

## Model

CNN + Bidirectional LSTM trained on speech emotion datasets.
