import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
import os
import json
import config
import argparse
import shutil
from datetime import datetime
import inspect
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None, class_to_idx_override=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # If a class mapping is provided (e.g., from model), use it instead of the hardcoded one
        if class_to_idx_override is not None:
            print(f"[Dataset] Using provided class-to-index mapping with {len(class_to_idx_override)} classes")
            self.class_to_idx = class_to_idx_override
            self.classes = sorted(list(self.class_to_idx.keys()))
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        else:
            self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
            
        self.samples = self._make_dataset()

        # Cache training dataset in RAM to speed up epochs on slow storage/OneDrive
        self.cache_in_ram = getattr(config.TrainingConfig, 'CACHE_IN_RAM', False) and ('train' in data_dir.lower())
        if self.cache_in_ram:
            print(f"\n[RAM Cache] Loading {len(self.samples)} images from {data_dir} into RAM for faster training...")
            self.cached_samples = []
            for path, target in self.samples:
                try:
                    with Image.open(path) as img:
                        self.cached_samples.append((img.convert('RGB'), target))
                except Exception as e:
                    print(f"Warning: Failed to load {path} during RAM caching: {e}")
                    self.cached_samples.append((None, target))
            print("[RAM Cache] Caching complete.")
        else:
            self.cached_samples = None

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            target_dir = os.path.join(self.data_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target_class])
                    samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cached_samples is not None and self.cached_samples[idx][0] is not None:
            sample, target = self.cached_samples[idx]
        else:
            path, target = self.samples[idx]
            try:
                with Image.open(path) as img:
                    sample = img.convert('RGB')
            except Exception as e:
                # Return empty black image if corrupted/missing
                sample = Image.new('RGB', config.IMAGE_SIZE, color=0)

        if self.transform:
            sample = self.transform(sample)
        return sample, target


def get_model(model_name, num_classes, base_weights_path=None, freeze_until=None):
    """
    Initializes a model, optionally loading custom base weights and freezing layers.

    Args:
        model_name (str): The name of the model architecture.
        num_classes (int): The number of output classes for the final layer.
        base_weights_path (str, optional): Path to a .pth file with custom pre-trained weights.
        freeze_until (str, optional): Name of the last layer to freeze. All subsequent
                                      layers will be trainable.

    Returns:
        torch.nn.Module: The initialized model.
    """

    # 1. Initialize model with modern ImageNet weights
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    elif model_name == 'shufflenet_v2':
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)

    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    elif model_name == 'lite_mdc':
        from lite_mdc import lite_mdc
        model = lite_mdc(num_classes=num_classes)
        # Note: Lite-MDC is trained from scratch (no ImageNet weights)
        # base_weights_path still works if you have domain-pretrained weights
        if base_weights_path:
            print(f"Loading custom base weights from: {base_weights_path}")
            model.load_state_dict(torch.load(base_weights_path), strict=False)
        return model
    
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # 2. Load custom base weights if a path is provided
    if base_weights_path:
        print(f"Loading custom base weights from: {base_weights_path}")
        model.load_state_dict(torch.load(base_weights_path), strict=False)

    # 3. Freeze layers up to the specified layer
    if freeze_until:
        print(f"Freezing layers until (and including): {freeze_until}")
        freezing = True
        for name, param in model.named_parameters():
            if name.startswith(freeze_until):
                freezing = False

            if freezing:
                param.requires_grad = False
            else:
                param.requires_grad = True

    # 4. Replace the final classification layer for the new task
    if model_name in ['resnet18', 'resnet34', 'resnet50']:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif model_name == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'shufflenet_v2':
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif model_name == 'densenet121':
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier.in_features, num_classes)
        )

    elif model_name == 'efficientnet_b0':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


def get_backbone_freeze_point(model_name):
    """
    Returns the layer name to freeze until for Stage 1 (Warmup).
    Freezes the entire backbone, leaving the head trainable.
    """
    if 'resnet' in model_name:
        return 'fc'
    elif 'mobilenet' in model_name:
        return 'classifier'
    elif 'shufflenet' in model_name:
        return 'fc'
    elif 'densenet' in model_name:
        return 'classifier'
    elif 'efficientnet' in model_name:
        return 'classifier'
    elif 'lite_mdc' in model_name:
        return 'classifier'  # freeze everything except final linear layer for warmup
    else:
        return None


def save_config_and_summary(output_dir, model_name, epochs, lr, batch_size, data_dir):
    """
    Save the config.py file and create a training summary document
    with all configuration parameters used.
    """
    # Copy config.py to output directory
    config_src = os.path.join(os.path.dirname(__file__), 'config.py')
    config_dst = os.path.join(output_dir, 'config.py')
    if os.path.exists(config_src):
        shutil.copy2(config_src, config_dst)
        print(f"Config file copied to {config_dst}")
    
    # Create a training summary document
    summary_path = os.path.join(output_dir, 'TRAINING_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING CONFIGURATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Training Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device Used: {DEVICE}\n\n")
        
        f.write("--- DATA CONFIGURATION ---\n")
        f.write(f"Data Directory: {data_dir}\n\n")
        
        f.write("--- MODEL CONFIGURATION ---\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Model Available Options: {config.TrainingConfig.AVAILABLE_MODELS}\n")
        f.write(f"Base Model Path: {config.TrainingConfig.BASE_MODEL_PATH}\n")
        f.write(f"Freeze Layers Until: {config.TrainingConfig.FREEZE_LAYERS_UNTIL}\n\n")
        
        f.write("--- TRAINING HYPERPARAMETERS (STAGE 1: WARMUP) ---\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"LR Scheduler Step Size: {config.TrainingConfig.LR_SCHEDULER_STEP_SIZE}\n")
        f.write(f"LR Scheduler Gamma: {config.TrainingConfig.LR_SCHEDULER_GAMMA}\n\n")
        
        f.write("--- EARLY STOPPING CONFIGURATION ---\n")
        f.write(f"Early Stopping Enabled: {config.TrainingConfig.APPLY_EARLY_STOPPING}\n")
        f.write(f"Early Stopping Patience: {config.TrainingConfig.EARLY_STOPPING_PATIENCE}\n")
        f.write(f"Early Stopping Min Delta: {config.TrainingConfig.EARLY_STOPPING_MIN_DELTA}\n\n")
        
        f.write("--- FINE-TUNING CONFIGURATION (STAGE 2) ---\n")
        f.write(f"Fine-Tuning Enabled: {config.TrainingConfig.FINE_TUNING_CONFIG.get('ENABLED', False)}\n")
        if config.TrainingConfig.FINE_TUNING_CONFIG.get('ENABLED', False):
            f.write(f"Unfreeze Layers: {config.TrainingConfig.FINE_TUNING_CONFIG.get('UNFREEZE_LAYERS', 'layer4')}\n")
            f.write(f"Fine-Tuning Learning Rate: {config.TrainingConfig.FINE_TUNING_CONFIG.get('LEARNING_RATE', 'N/A')}\n")
            f.write(f"Fine-Tuning Epochs: {config.TrainingConfig.FINE_TUNING_CONFIG.get('EPOCHS', 'N/A')}\n\n")
        
        f.write("--- IMAGE PREPROCESSING ---\n")
        f.write(f"Image Size: {config.IMAGE_SIZE}\n")
        f.write(f"Normalization Mean: {config.MEAN}\n")
        f.write(f"Normalization Std: {config.STD}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("All outputs from this training run are saved in this directory.\n")
        f.write("=" * 80 + "\n")
    
    print(f"Training summary saved to {summary_path}")




def train_model_stage(model, train_loader, val_loader, epochs, lr, model_path, model_dir, stage_name):
    """
    Runs the training loop for a single stage.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=getattr(config.TrainingConfig, 'LABEL_SMOOTHING', 0.1))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    # Scheduler disabled
    scheduler = None

    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_loss_history = []
    val_loss_history = []

    best_val_acc = 0.0

    print(f"Starting {stage_name} with lr={lr}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_loss_history.append(val_epoch_loss)
        val_acc = correct / total

        print(f"[{stage_name}] Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {epoch_loss:.3f}.. "
              f"Val loss: {val_epoch_loss:.3f}.. "
              f"Val accuracy: {val_acc:.3f}")

        if scheduler is not None:
            scheduler.step()

        # Save on BOTH val loss improvement AND val accuracy improvement
        # CORRECTED early stopping block (replace the duplicated section)
        if val_epoch_loss < best_val_loss - config.TrainingConfig.EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved (val_loss={val_epoch_loss:.3f}, val_acc={val_acc:.3f})")
        else:
            epochs_no_improve += 1
            print(f"Early stopping counter: {epochs_no_improve}/{config.TrainingConfig.EARLY_STOPPING_PATIENCE}")

        if config.TrainingConfig.APPLY_EARLY_STOPPING:
            if epochs_no_improve >= config.TrainingConfig.EARLY_STOPPING_PATIENCE:
                print(f"Stopping early.")
                break
        
                   

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title(f'{stage_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(model_dir, f'loss_plot_{stage_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.close()


def train(data_dir, model_name, epochs, lr, batch_size, model_path, model_dir):
    # No online augmentation — offline pipeline handles it
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])

    train_dataset = PlantDiseaseDataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = PlantDiseaseDataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    
    # If caching in RAM is enabled, num_workers=0 is faster on Windows to avoid process spawning overhead
    num_workers = 0 if getattr(config.TrainingConfig, 'CACHE_IN_RAM', False) else 4
    print(f"DataLoader setup: num_workers={num_workers}, cache_in_ram={getattr(config.TrainingConfig, 'CACHE_IN_RAM', False)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    num_classes = len(train_dataset.classes)

    # --- STAGE 1: WARMUP ---
    print("\n=== STAGE 1: Warmup Training ===")

    stage1_freeze = get_backbone_freeze_point(model_name)
    if stage1_freeze is None:
        stage1_freeze = config.TrainingConfig.FREEZE_LAYERS_UNTIL
        print(f"Warning: Could not determine backbone freeze point for {model_name}. Using config default: {stage1_freeze}")

    model = get_model(
        model_name,
        num_classes,
        base_weights_path=config.TrainingConfig.BASE_MODEL_PATH,
        freeze_until=stage1_freeze
    )
    model.to(DEVICE)

    stage1_model_path = os.path.join(model_dir, 'best_model_stage1.pth')

    train_model_stage(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        model_path=stage1_model_path,
        model_dir=model_dir,
        stage_name="Stage 1 (Warmup)"
    )

    final_model_path = stage1_model_path

    # --- STAGE 2: FINE-TUNING ---
    if config.TrainingConfig.FINE_TUNING_CONFIG.get('ENABLED', False):
        print("\n=== STAGE 2: Fine-Tuning ===")

        ft_config = config.TrainingConfig.FINE_TUNING_CONFIG

        stage2_freeze = ft_config.get('UNFREEZE_LAYERS', 'layer4')

        model = get_model(
            model_name,
            num_classes,
            base_weights_path=None,
            freeze_until=stage2_freeze
        )

        if os.path.exists(stage1_model_path):
            print(f"Loading Stage 1 weights from {stage1_model_path}...")
            model.load_state_dict(torch.load(stage1_model_path))
        else:
            print(f"Error: Stage 1 model not found at {stage1_model_path}. Cannot proceed to Stage 2.")
            return

        model.to(DEVICE)

        stage2_model_path = os.path.join(model_dir, 'best_model_stage2.pth')

        train_model_stage(
            model,
            train_loader,
            val_loader,
            epochs=ft_config.get('EPOCHS', 10),
            lr=ft_config.get('LEARNING_RATE', lr * 0.1),
            model_path=stage2_model_path,
            model_dir=model_dir,
            stage_name="Stage 2 (Fine-Tuning)"
        )

        final_model_path = stage2_model_path

    if model_path != final_model_path:
        print(f"Copying best model from {final_model_path} to {model_path}")
        shutil.copy(final_model_path, model_path)

    class_to_idx_path = os.path.splitext(model_path)[0] + '_class_to_idx.json'
    with open(class_to_idx_path, 'w') as f:
        json.dump(train_dataset.class_to_idx, f)
    print(f"Class to index mapping saved to {class_to_idx_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data_dir', type=str, required=False, default=None, help='Directory with train and val data')
    parser.add_argument('--processed_dir', type=str, default=None, help='Path to processed dataset containing train/val/test subfolders (overrides --data_dir if provided)')
    parser.add_argument('--model_name', type=str, default=config.TrainingConfig.MODEL_NAME, choices=config.TrainingConfig.AVAILABLE_MODELS, help='Model to train')
    parser.add_argument('--epochs', type=int, default=config.TrainingConfig.EPOCHS, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=config.TrainingConfig.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=config.TrainingConfig.BATCH_SIZE, help='Batch size')
    parser.add_argument('--output_dir', type=str, required=False, default=None, help='Output directory for training results (auto-generated if not provided)')
    args = parser.parse_args()

    if not args.data_dir and not args.processed_dir:
        default_path = r"processed_dataset"
        print(f"No data_dir provided. Using local processed_dataset folder")
        args.data_dir = default_path

    data_root = args.processed_dir if args.processed_dir else args.data_dir

    if not os.path.isdir(data_root):
        parser.error(f"Data directory does not exist: {data_root}")

    train_sub = os.path.join(data_root, 'train')
    val_sub = os.path.join(data_root, 'val')
    if not os.path.isdir(train_sub):
        parser.error(f"Required subdirectory missing: {train_sub}")
    if not os.path.isdir(val_sub):
        parser.error(f"Required subdirectory missing: {val_sub}")

    # Create timestamped output directory if not specified
    if not args.output_dir:
        base_output_dir = os.path.join(os.getcwd(), 'training_outputs')
        os.makedirs(base_output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = os.path.join(base_output_dir, f'{args.model_name}_{timestamp}')
        os.makedirs(model_dir, exist_ok=True)
    else:
        model_dir = args.output_dir
        os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training Output Directory: {model_dir}")
    print(f"{'='*80}\n")
    
    # Save config and create summary
    save_config_and_summary(model_dir, args.model_name, args.epochs, args.lr, args.batch_size, data_root)
    
    # Set model path inside the output directory
    model_path = os.path.join(model_dir, f'{args.model_name}_final.pth')
    
    train(data_root, args.model_name, args.epochs, args.lr, args.batch_size, model_path, model_dir)