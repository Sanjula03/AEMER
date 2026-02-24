"""
Video Emotion Model Fine-Tuning: 4 → 7 Emotions
=================================================

Fine-tunes the existing ResNet-18 facial emotion model to recognize 7 emotions
by replacing the final classification layer and training on FER2013 (all 7 classes).

Old classes (4): angry, happy, sad, neutral
New classes (7): angry, happy, sad, neutral, fear, surprise, disgust

FER2013 already has all 7 classes! You were previously filtering to only 4.
This script uses all 7 classes from FER2013.

Usage:
    1. Upload this script to Google Colab
    2. Upload your existing 4-class model (video_model.pth)
    3. Download FER2013 from Kaggle
    4. Run all cells
    5. Download the new 7-class model

Dataset:
    - FER2013: https://www.kaggle.com/datasets/msambare/fer2013
    - Structure: fer2013/train/{emotion_name}/*.jpg, fer2013/test/{emotion_name}/*.jpg
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ============================================
# CONFIGURATION
# ============================================

# FER2013 emotion folder names → our label mapping
# FER2013 folders: angry, disgust, fear, happy, neutral, sad, surprise
# Our order:       angry, happy, sad, neutral, fear, surprise, disgust
LABEL_NAMES = ['angry', 'happy', 'sad', 'neutral', 'fear', 'surprise', 'disgust']
NUM_CLASSES = 7

# FER2013 folder name → our index mapping
FER_TO_OUR_INDEX = {
    'angry': 0,
    'happy': 1,
    'sad': 2,
    'neutral': 3,
    'fear': 4,
    'surprise': 5,
    'disgust': 6,
}

# Path to existing 4-class model
OLD_MODEL_PATH = 'video_model.pth'

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0001  # Lower LR for full model fine-tuning
EPOCHS = 50  # More epochs for better convergence
FREEZE_BACKBONE = False  # Train ALL layers for maximum accuracy
IMAGE_SIZE = 224

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================
# MODEL ARCHITECTURE (same as original)
# ============================================

class FacialEmotionResNet(nn.Module):
    """ResNet-18 based facial emotion classifier."""
    
    def __init__(self, num_classes=7):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


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
    
    # Auto-detect num_classes from final layer shape
    fc_key = 'resnet.fc.4.weight'  # Last Linear layer in the fc Sequential
    if fc_key in state_dict:
        detected_classes = state_dict[fc_key].shape[0]
        print(f"  Detected {detected_classes}-class model in checkpoint")
        old_classes = detected_classes
    
    if old_classes == new_classes:
        # Model is already the right size - load directly
        model = FacialEmotionResNet(num_classes=new_classes)
        model.load_state_dict(state_dict)
        print(f"  ✅ Loaded {new_classes}-class model directly (continuing training)")
    else:
        # Need to transfer weights and replace final layer
        old_model = FacialEmotionResNet(num_classes=old_classes)
        old_model.load_state_dict(state_dict)
        print(f"  ✅ Loaded {old_classes}-class model from {old_model_path}")
        
        model = FacialEmotionResNet(num_classes=new_classes)
        old_state = old_model.state_dict()
        new_state = model.state_dict()
        
        transferred = 0
        skipped = 0
        for name, param in old_state.items():
            if name in new_state and param.shape == new_state[name].shape:
                new_state[name] = param
                transferred += 1
            else:
                skipped += 1
        
        model.load_state_dict(new_state)
        print(f"  ✅ Transferred {transferred} layers (ResNet backbone + fc1)")
        print(f"  ⏩ Skipped {skipped} layers (final classification layer)")
    
    # Optionally freeze backbone
    if FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            if 'resnet.fc' not in name:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Froze ResNet backbone. Training {trainable:,}/{total:,} parameters ({100*trainable/total:.1f}%)")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Training ALL {trainable:,}/{total:,} parameters (100%)")
    
    return model.to(device)


# ============================================
# STEP 2: DATASET - FER2013 (ALL 7 CLASSES)
# ============================================

class RemappedDataset(torch.utils.data.Dataset):
    """Wraps ImageFolder to remap FER2013 class indices to our label order."""
    
    def __init__(self, image_folder):
        self.dataset = image_folder
        
        # Build remapping: ImageFolder.class_to_idx → our indices
        self.remap = {}
        for folder_name, folder_idx in image_folder.class_to_idx.items():
            if folder_name.lower() in FER_TO_OUR_INDEX:
                self.remap[folder_idx] = FER_TO_OUR_INDEX[folder_name.lower()]
        
        # Filter samples to only include valid emotions
        self.indices = [
            i for i, (_, label) in enumerate(image_folder.samples)
            if label in self.remap
        ]
        
        print(f"    Mapping: {image_folder.class_to_idx}")
        print(f"    Remapped to our order: {self.remap}")
        print(f"    Valid samples: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        return img, self.remap[label]


def load_fer2013(data_dir, split='train'):
    """
    Load FER2013 dataset.
    
    Expected structure:
        data_dir/
            train/
                angry/
                disgust/
                fear/
                happy/
                neutral/
                sad/
                surprise/
            test/
                (same folders)
    """
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"  ❌ Directory not found: {split_dir}")
        return None
    
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # FER2013 images are grayscale 48x48, but ResNet expects 3-channel
    # The Resize + ToTensor handles this via transforms
    image_folder = ImageFolder(split_dir, transform=transform)
    
    print(f"\n  Loading {split} set from {split_dir}:")
    dataset = RemappedDataset(image_folder)
    
    return dataset


# ============================================
# STEP 3: TRAINING
# ============================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
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
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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
    FER2013_DIR = '/content/fer2013'  # <-- Change to your FER2013 path
    
    # Step 1: Load pre-trained model
    model = load_and_modify_model(OLD_MODEL_PATH, old_classes=4, new_classes=7)
    
    # Step 2: Load FER2013 dataset
    print(f"\n{'='*50}")
    print("STEP 2: Loading FER2013 dataset (all 7 classes)")
    print(f"{'='*50}")
    
    train_dataset = load_fer2013(FER2013_DIR, 'train')
    test_dataset = load_fer2013(FER2013_DIR, 'test')
    
    if train_dataset is None or test_dataset is None:
        print("\n❌ FER2013 not found! Download from:")
        print("   https://www.kaggle.com/datasets/msambare/fer2013")
        print(f"   Expected at: {FER2013_DIR}/train/ and {FER2013_DIR}/test/")
        exit(1)
    
    # Print class distribution
    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Use portion of train for validation
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  Train split: {train_size}, Val split: {val_size}")
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Step 3: Setup training
    print(f"\n{'='*50}")
    print("STEP 3: Fine-tuning")
    print(f"{'='*50}")
    
    criterion = nn.CrossEntropyLoss()
    
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
            torch.save(model.state_dict(), 'video_model_7class.pth')
    
    # Step 4: Evaluate
    print(f"\n{'='*50}")
    print("STEP 4: Evaluation")
    print(f"{'='*50}")
    
    model.load_state_dict(torch.load('video_model_7class.pth', map_location=device))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    
    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=LABEL_NAMES)}")
    
    # Plot results
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
    plt.savefig('video_finetune_results.png', dpi=150)
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
    }, 'video_model.pth')
    
    print("  ✅ Saved: video_model.pth")
    print("\n  📥 Copy this file to: VideoModel/video_model.pth")
    
    # Colab download
    try:
        from google.colab import files
        files.download('video_model.pth')
        files.download('video_finetune_results.png')
    except:
        print("  (Not in Colab - manually download the file)")
