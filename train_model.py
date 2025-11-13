"""
train_model.py - Train SRNet Steganalysis Model

This script trains a deep learning model to detect steganography in images.

Usage:
    python train_model.py

Requirements:
    - Dataset of cover images (normal images)
    - Will automatically generate stego images during training
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from AES_LSB import UniversalSteganography
from Crypto.Random import get_random_bytes


# ===============================================
# SRNet Model (Same as in app.py)
# ===============================================
class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer3 = self._make_res_block(16, 16)
        self.layer4 = self._make_res_block(16, 64)
        self.layer5 = self._make_res_block(64, 128)
        self.layer6 = self._make_res_block(128, 256)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1)
        )

    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.fc(x)
        return x


# ===============================================
# Dataset Class
# ===============================================
class StegoDataset(Dataset):
    """Dataset that generates stego images on-the-fly"""
    
    def __init__(self, image_dir, transform=None, stego_ratio=0.5):
        """
        Args:
            image_dir: Directory containing cover images
            transform: Image transformations
            stego_ratio: Ratio of images to convert to stego (0.5 = 50%)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.stego_ratio = stego_ratio
        
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(valid_extensions)
        ]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images")
        
        # Initialize steganography
        self.stego_engine = UniversalSteganography(payload=0.3)
        self.key = get_random_bytes(32)
        
        # Sample messages for embedding
        self.messages = [
            "This is a secret message",
            "Hidden data inside",
            "Steganography test",
            "Confidential information",
            "Secret communication",
            "Encrypted payload",
            "Covert channel data",
            "Private message content"
        ]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Decide if this should be stego or cover
        is_stego = np.random.random() < self.stego_ratio
        
        if is_stego:
            # Create stego image
            image = self._create_stego_image(image)
            label = 1.0  # Stego
        else:
            label = 0.0  # Cover
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor([label], dtype=torch.float32)
    
    def _create_stego_image(self, pil_image):
        """Embed a message into the image"""
        try:
            # Save to temporary file
            temp_cover = "temp_cover.png"
            temp_stego = "temp_stego.png"
            pil_image.save(temp_cover)
            
            # Embed random message
            message = np.random.choice(self.messages)
            success = self.stego_engine.embed_file(
                cover_path=temp_cover,
                output_path=temp_stego,
                data=message,
                key=self.key
            )
            
            if success:
                stego_image = Image.open(temp_stego).convert('RGB')
            else:
                stego_image = pil_image
            
            # Cleanup
            if os.path.exists(temp_cover):
                os.remove(temp_cover)
            if os.path.exists(temp_stego):
                os.remove(temp_stego)
            
            return stego_image
        except:
            return pil_image


# ===============================================
# Training Function
# ===============================================
def train_model(
    image_dir,
    num_epochs=10,
    batch_size=8,
    learning_rate=0.001,
    img_size=256,
    save_path="best_srnet_from_scratch_changed.pth"
):
    """Train the steganalysis model"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training SRNet Steganalysis Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Image Size: {img_size}x{img_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = StegoDataset(image_dir, transform=transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}\n")
    
    # Initialize model
    model = SRNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc="Training")
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model (Val Acc: {val_accuracy:.2f}%)")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}\n")
    
    # Plot training history
    plot_history(history)
    
    return model, history


def plot_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training plot saved to: training_history.png")


# ===============================================
# Main Function
# ===============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SRNet Steganalysis Model")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="training_images",
        help="Directory containing training images"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size (will be resized to img_size x img_size)"
    )
    
    args = parser.parse_args()
    
    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"\n❌ Error: Directory '{args.image_dir}' not found!")
        print("\nPlease create a directory with training images:")
        print(f"  1. Create folder: mkdir {args.image_dir}")
        print(f"  2. Add 100+ images (any normal images)")
        print(f"  3. Run: python train_model.py\n")
        exit(1)
    
    # Train the model
    model, history = train_model(
        image_dir=args.image_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        img_size=args.img_size
    )
    
    print("\n✅ Model training complete!")
    print("You can now use the model in your Flask app.")