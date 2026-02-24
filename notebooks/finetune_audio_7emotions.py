"""
Audio Emotion Model Fine-Tuning: 4 → 7 Emotions
=================================================

Fine-tunes the existing CNN-BiLSTM audio emotion model to recognize 7 emotions
by replacing the final classification layer and training on RAVDESS.

Old classes (4): angry, happy, sad, neutral
New classes (7): angry, happy, sad, neutral, fear, surprise, disgust

Usage:
    1. Upload this script to Google Colab
    2. Upload your existing 4-class model (best_model.pth)
    3. Mount Google Drive or upload RAVDESS dataset
    4. Run all cells
    5. Download the new 7-class model

Dataset:
    - RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ============================================
# CONFIGURATION
# ============================================

# New 7 emotion labels
LABEL_NAMES = ['angry', 'happy', 'sad', 'neutral', 'fear', 'surprise', 'disgust']
NUM_CLASSES = 7

# Path to your existing 4-class model
OLD_MODEL_PATH = 'best_model.pth'

# Audio preprocessing settings (MUST match your original training)
SAMPLE_RATE = 16000
DURATION = 4.0  # seconds
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 400
FMAX = 8000

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Lower LR for full model fine-tuning
EPOCHS = 50  # More epochs for better convergence
FREEZE_EARLY_LAYERS = False  # Train ALL layers for maximum accuracy

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================
# MODEL ARCHITECTURE (same as original)
# ============================================

class CNN_BiLSTM(nn.Module):
    """
    CNN + Bidirectional LSTM model for audio emotion recognition.
    Input: Log-Mel Spectrogram (1, 128, time_frames)
    """
    def __init__(self, num_classes=7):
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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop(x)

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


# ============================================
# STEP 1: LOAD & MODIFY THE PRE-TRAINED MODEL
# ============================================

def load_and_modify_model(old_model_path, old_classes=4, new_classes=7):
    """
    Load a pre-trained model and modify for new_classes.
    Auto-detects if model is already 7-class (from previous fine-tuning).
    """
    print(f"\n{'='*50}")
    print("STEP 1: Loading pre-trained model")
    print(f"{'='*50}")
    
    # Load checkpoint to detect num_classes
    checkpoint = torch.load(old_model_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Auto-detect num_classes from fc2 layer shape
    if 'fc2.weight' in state_dict:
        detected_classes = state_dict['fc2.weight'].shape[0]
        print(f"  Detected {detected_classes}-class model in checkpoint")
        old_classes = detected_classes
    
    if old_classes == new_classes:
        # Model is already the right size - load directly
        model = CNN_BiLSTM(num_classes=new_classes)
        model.load_state_dict(state_dict)
        print(f"  ✅ Loaded {new_classes}-class model directly (continuing training)")
    else:
        # Need to transfer weights and replace final layer
        old_model = CNN_BiLSTM(num_classes=old_classes)
        old_model.load_state_dict(state_dict)
        print(f"  ✅ Loaded {old_classes}-class model from {old_model_path}")
        
        model = CNN_BiLSTM(num_classes=new_classes)
        old_state = old_model.state_dict()
        new_state = model.state_dict()
        
        transferred = 0
        for name, param in old_state.items():
            if name in new_state and 'fc2' not in name:
                new_state[name] = param
                transferred += 1
        
        model.load_state_dict(new_state)
        print(f"  ✅ Transferred {transferred} layers (CNN + LSTM + fc1)")
        print(f"  ✅ New fc2 layer initialized randomly ({new_classes} outputs)")
    
    # Optionally freeze early layers
    if FREEZE_EARLY_LAYERS:
        for name, param in model.named_parameters():
            if 'fc2' not in name and 'fc1' not in name:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Froze early layers. Training {trainable:,}/{total:,} parameters ({100*trainable/total:.1f}%)")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Training ALL {trainable:,}/{total:,} parameters (100%)")
    
    return model.to(device)


# ============================================
# STEP 2: DATASET - RAVDESS + IEMOCAP
# ============================================

# RAVDESS emotion mapping (from filename encoding)
# RAVDESS format: XX-XX-EMOTION-XX-XX-XX-XX.wav
# Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_EMOTION_MAP = {
    '01': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise',
    # '02': 'calm' - excluded (not in our 7 emotions)
}


def load_ravdess(data_dir):
    """Load RAVDESS dataset with 7 emotion labels.
    
    Expected structure:
        data_dir/
            Actor_01/
                03-01-01-01-01-01-01.wav
                ...
            Actor_02/
                ...
    """
    files = []
    labels = []
    
    for wav_file in glob.glob(os.path.join(data_dir, '**/*.wav'), recursive=True):
        filename = os.path.basename(wav_file)
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in RAVDESS_EMOTION_MAP:
                emotion = RAVDESS_EMOTION_MAP[emotion_code]
                files.append(wav_file)
                labels.append(LABEL_NAMES.index(emotion))
    
    print(f"  RAVDESS: {len(files)} samples loaded")
    return files, labels




class AudioEmotionDataset(Dataset):
    """Dataset that loads audio files and extracts mel spectrograms."""
    
    def __init__(self, file_paths, labels, sample_rate=16000, duration=4.0, n_mels=128):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration)
        self.n_mels = n_mels
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Pad or trim to fixed length
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)))
            else:
                audio = audio[:self.target_length]
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            # Extract log-mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_mels=self.n_mels,
                n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, fmax=FMAX
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize spectrogram
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
            
            tensor = torch.FloatTensor(log_mel).unsqueeze(0)  # (1, n_mels, time)
            return tensor, label
            
        except Exception as e:
            print(f"  Error loading {audio_path}: {e}")
            dummy = torch.zeros(1, self.n_mels, self.target_length // HOP_LENGTH + 1)
            return dummy, label


# ============================================
# STEP 3: TRAINING
# ============================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_labels


# ============================================
# MAIN - RUN FINE-TUNING
# ============================================

if __name__ == '__main__':
    # =====================
    # ⚠️ SET YOUR PATH HERE
    # =====================
    RAVDESS_DIR = '/content/RAVDESS'        # <-- Change to your RAVDESS path
    
    # Step 1: Load pre-trained model
    model = load_and_modify_model(OLD_MODEL_PATH, old_classes=4, new_classes=7)
    
    # Step 2: Load RAVDESS dataset
    print(f"\n{'='*50}")
    print("STEP 2: Loading RAVDESS dataset")
    print(f"{'='*50}")
    
    all_files, all_labels = [], []
    
    if os.path.exists(RAVDESS_DIR):
        files, labels = load_ravdess(RAVDESS_DIR)
        all_files.extend(files)
        all_labels.extend(labels)
    else:
        print(f"  ❌ RAVDESS not found at {RAVDESS_DIR}")
    
    if len(all_files) == 0:
        print("\n❌ No data found! Please set RAVDESS_DIR correctly.")
        exit(1)
    
    # Print distribution
    print(f"\n  Total samples: {len(all_files)}")
    for i, name in enumerate(LABEL_NAMES):
        count = all_labels.count(i)
        print(f"    {name}: {count}")
    
    # Train/Val/Test split
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    
    print(f"\n  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Create data loaders
    train_dataset = AudioEmotionDataset(train_files, train_labels)
    val_dataset = AudioEmotionDataset(val_files, val_labels)
    test_dataset = AudioEmotionDataset(test_files, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Step 3: Set up training
    print(f"\n{'='*50}")
    print("STEP 3: Fine-tuning")
    print(f"{'='*50}")
    
    # Handle class imbalance with weighted loss
    class_counts = np.array([all_labels.count(i) for i in range(NUM_CLASSES)])
    class_weights = 1.0 / (class_counts + 1)  # +1 to avoid division by zero
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    
    # Only optimize trainable parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
              + (" ★ BEST" if val_acc > best_val_acc else ""))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_7class.pth')
    
    # Step 4: Evaluate
    print(f"\n{'='*50}")
    print("STEP 4: Evaluation")
    print(f"{'='*50}")
    
    model.load_state_dict(torch.load('best_model_7class.pth', map_location=device))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    
    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=LABEL_NAMES)}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    import seaborn as sns
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[1])
    axes[1].set_title('Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('audio_finetune_results.png', dpi=150)
    plt.show()
    
    # Step 5: Save final model
    print(f"\n{'='*50}")
    print("STEP 5: Save model")
    print(f"{'='*50}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': NUM_CLASSES,
        'label_names': LABEL_NAMES,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
    }, 'best_model.pth')
    
    print("  ✅ Saved: best_model.pth")
    print("\n  📥 Copy this file to: AudioModel/best_model.pth")
    
    # Colab download
    try:
        from google.colab import files
        files.download('best_model.pth')
        files.download('audio_finetune_results.png')
    except:
        print("  (Not in Colab - manually download the file)")
