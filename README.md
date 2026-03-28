# AEMER - Accent-Enhanced Multimodal Emotion Recognition

This repository contains the source code for my final year project. AEMER is a multimodal emotion recognition framework that combines facial, vocal, and textual inputs to classify human emotions. A key technical feature of this project is the integration of an accent detection model (identifying 9 regional accents) to dynamically adjust the late-fusion weights, reducing bias in speech emotion recognition.

## Tech Stack
*   **Frontend:** React, TypeScript, Vite, Tailwind CSS
*   **Backend:** FastAPI, Python
*   **Machine Learning:** PyTorch, Librosa, OpenCV
*   **Models:** ResNet-18 (Video), CNN-BiLSTM (Audio), DistilRoBERTa (Text)

---

## Local Setup Instructions

To evaluate the application locally, you need to run both the Python backend API and the React frontend server.

### 1. Starting the Backend API
The backend handles media processing and loads all deep learning models into memory.

1. Open a terminal and navigate to the `backend` directory.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server using the batch script:
   ```bash
   start_backend.bat
   ```
   *(Alternatively, run `uvicorn main:app --reload`)*

**Note:** The pre-trained model weights are approximately 300MB. The backend may take 15-30 seconds to map them into RAM on the initial startup.

### 2. Starting the Frontend UI
The frontend provides the dashboard for capturing media and viewing classification metrics.

1. Open a new, separate terminal at the root of the project directory.
2. Install the Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open your browser and navigate to `http://localhost:5173`.

### 3. Environment Configuration
For the Gemini generative AI integration to function, please locate the `.env.example` file in the root directory and rename it to `.env`. You must replace the placeholder text inside with a valid Gemini API key.

---

