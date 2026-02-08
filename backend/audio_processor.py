"""
Audio Processor for AEMER
Converts raw audio files into Log-Mel Spectrograms for the emotion model.

Preprocessing steps:
1. Resample to 16kHz
2. Trim silence
3. Normalize audio
4. Pad/trim to 4 seconds
5. Extract Log-Mel Spectrogram
6. Per-sample normalization (zero mean, unit variance)
"""

import numpy as np
import librosa
import torch


class AudioProcessor:
    """Preprocesses audio files for the emotion recognition model."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 4.0,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int = 400,
        fmax: int = 8000
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        # Mel spectrogram parameters
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmax = fmax
    
    def load_audio_from_path(self, file_path: str) -> np.ndarray:
        """Load and resample audio from file path."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def load_audio(self, audio_bytes: bytes, filename: str) -> tuple:
        """
        Load audio from bytes and return raw audio data with sample rate.
        Used for accent detection which needs raw audio.
        
        Returns:
            tuple: (audio_array, sample_rate)
        """
        import tempfile
        import os
        
        ext = os.path.splitext(filename)[1] if filename else '.wav'
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            audio, sr = librosa.load(tmp_path, sr=self.sample_rate)
        finally:
            os.unlink(tmp_path)
        
        return audio, sr
    
    def load_audio_from_bytes(self, audio_bytes: bytes, original_filename: str) -> np.ndarray:
        """Load audio from bytes (for uploaded files)."""
        import tempfile
        import os
        
        # Get file extension from original filename
        ext = os.path.splitext(original_filename)[1] if original_filename else '.wav'
        
        # Write bytes to temp file and load with librosa
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            audio = self.load_audio_from_path(tmp_path)
        finally:
            os.unlink(tmp_path)
        
        return audio
    
    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading and trailing silence."""
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        return trimmed
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent amplitude differences."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def pad_or_trim(self, audio: np.ndarray) -> np.ndarray:
        """Pad or trim audio to fixed length (4 seconds)."""
        if len(audio) < self.target_length:
            # Pad with zeros
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            # Trim to target length
            audio = audio[:self.target_length]
        return audio
    
    def extract_log_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract Log-Mel Spectrogram features."""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmax=self.fmax
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def normalize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Normalize spectrogram to zero mean and unit variance."""
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        if std > 0:
            spectrogram = (spectrogram - mean) / std
        return spectrogram
    
    def process(self, audio_bytes: bytes, filename: str) -> torch.Tensor:
        """
        Full preprocessing pipeline.
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename (for extension detection)
            
        Returns:
            torch.Tensor: Preprocessed log-mel spectrogram ready for model
        """
        # Load audio from bytes
        audio = self.load_audio_from_bytes(audio_bytes, filename)
        
        # Preprocessing pipeline
        audio = self.trim_silence(audio)
        audio = self.normalize_audio(audio)
        audio = self.pad_or_trim(audio)
        
        # Feature extraction
        log_mel_spec = self.extract_log_mel_spectrogram(audio)
        log_mel_spec = self.normalize_spectrogram(log_mel_spec)
        
        # Convert to tensor and add batch and channel dimensions
        # Shape: (1, 1, n_mels, time_frames)
        tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0)
        
        return tensor


# Test the processor
if __name__ == "__main__":
    processor = AudioProcessor()
    print("AudioProcessor initialized successfully!")
    print(f"Target sample rate: {processor.sample_rate} Hz")
    print(f"Target duration: {processor.duration} seconds")
    print(f"Mel bands: {processor.n_mels}")
