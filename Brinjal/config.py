"""
Configuration file for Cotton Image Preprocessing Pipeline
All tunable parameters are defined here
"""

import os
import cv2
import numpy as np

# --- Training Configuration ---
class TrainingConfig:
    """Configuration class for model training parameters"""
    # Model Selection
    MODEL_NAME = 'resnet50'  # The model to be trained
    AVAILABLE_MODELS = ['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2',
                        'efficientnet_b0', 'densenet121', 'lite_mdc', 'shufflenet_v2']  # add lite_mdc
    

    # Fine-tuning with Custom Weights
    # If a path is provided, the script will load these weights as a base
    # before fine-tuning. Set to None to use default ImageNet weights.
    BASE_MODEL_PATH = None  # Example: 'path/to/your/plantvillage_weights.pth'

    # Layer Freezing for Fine-Tuning
    # Specify the name of the last layer to be frozen. All layers before this
    # one will not be trained. Set to None to train all layers.
    # Examples:
    # - ResNet50: 'layer3' (freezes conv1, layer1, layer2, layer3; trains layer4 and fc)
    # - MobileNetV2: 'features.14' (freezes up to the 14th feature block)
    FREEZE_LAYERS_UNTIL = 'None'  # Set to None to freeze no layers (train all)
    
    # Training Hyperparameters
    EPOCHS = 40
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32

    # RAM Caching to speed up training (bypasses OneDrive/disk I/O bottlenecks)
    CACHE_IN_RAM = True

    # Label smoothing in loss function (0.0 makes the model's predictions more confident, helping to beat the 70% confidence evaluation threshold)
    LABEL_SMOOTHING = 0.0



    # Early Stopping Configuration
    APPLY_EARLY_STOPPING = True # Set to False to disable early stopping
    # Stop training if the validation loss doesn't imporove for 'PATIENCE' epochs.
    EARLY_STOPPING_PATIENCE = 5
    # Minimum change in the monitored quantity to qualify as an improvement.
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Learning Rate Scheduler Configuration (DISABLED)
    # LR_SCHEDULER_TYPE = 'none'  # Scheduler is disabled
    LR_SCHEDULER_STEP_SIZE = 7
    LR_SCHEDULER_GAMMA = 0.1

    # Fine-Tuning Configuration (Stage 2)
    FINE_TUNING_CONFIG = {
        'ENABLED': True,
        # Layer to unfreeze from (inclusive). 
        # For ResNet: 'layer4' (unfreezes layer4 and fc), 'layer3' (unfreezes layer3, layer4, fc)
        # For MobileNetV2: 'features.14' (example)
        # If model specific logic is needed, this might need adjustment per model.
        'UNFREEZE_LAYERS': 'layer4',  # Example for MobileNetV2
        'LEARNING_RATE': 0.0001, # Typically lower than initial LR
        'EPOCHS': 25 # Number of fine-tuning epochs
    }

    

# --- Data Parameters (used by train/evaluate after preprocessing) ---
IMAGE_SIZE = (320,320) # Standard input size for many models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PreprocessingConfig:
    """Configuration class for image preprocessing parameters"""

    # ==================== PATH CONFIGURATION ====================
    # Base directories
    INPUT_DIR = r"Brinjal_leaf_cropped"  # Source dataset directory (SAM-cropped images)
    OUTPUT_DIR = r"processed_dataset_sam"  # Output directory for processed dataset
    DUPLICATES_DIR = r"duplicates_sam"  # Directory to move duplicate images

    # ==================== DUPLICATE REMOVAL CONFIGURATION ====================
    # Use MD5 for duplicate detection (exact match only)
    MOVE_DUPLICATES = True  # If False, duplicates will be deleted

    # ==================== RESIZING CONFIGURATION ====================
    # Target image size (width, height)
    TARGET_SIZE = (320, 320)
    # Interpolation method for resizing
    # cv2.INTER_LANCZOS4 is best for downscaling and feature retention
    INTERPOLATION_METHOD = cv2.INTER_LANCZOS4

    # ==================== NORMALIZATION CONFIGURATION ====================
    # Normalize images to [0, 1] range
    APPLY_NORMALIZATION = False
    # Normalization method: 'minmax' (0-1) or 'standardize' (mean=0, std=1)
    NORMALIZATION_METHOD = 'minmax'  # 'minmax' or 'standardize'
    # For standardization (if using 'standardize')
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean (RGB)
    NORMALIZATION_STD = [0.229, 0.224, 0.225]   # ImageNet std (RGB)

    
    # ==================== AUGMENTATION CONFIGURATION ====================
    # Target number of images per class after augmentation
    # Strategy: Check raw data first

    #- If count == 0 -> use max(image count per class) as target count
    #- Else: 
    #    - If count < 800 → augment to 800 (diverse augmentation)
    #    - If count > 800 → downsample intelligently to 800 (keep best images)
    #    - If count = 800 → no change

    TARGET_IMAGES_PER_CLASS = 1200  # middle of your natural class-size range

    CLASS_AUGMENTATION_TARGETS = {
        "Aphid":                   1200,  # 612 -> augment up
        "Early_Blight":            1000,  # 526 -> augment up
        "Healthy":                 1000,  # 758 -> augment up
        "Jassid":                  1000,  # 579 -> augment up
        "Little_Leaf":             1200,  # 889 -> augment up
        "Mealybugs":               1000,  # 685 -> augment up
        "Phomopsis_Blight_Fruit":  800,   # 196 -> augment up (small class)
        "Shoot and fruit borer":   1200,  # 1443 -> downsample
        "Whitefly":                1000,  # 756 -> augment up
    }



    # Augmentation operation pool (diverse set)
    # When augmenting, we will try to distribute required generated images
    # as evenly as possible across the enabled operations below.
    AUGMENTATION_OPERATIONS = [
        'rotation', 'flip', 'translation', 'scale', 'shear',
        'color_jitter', 'gaussian_noise', 'blur', 'random_crop'
    ]

    #Enable randomize augmentation operations
    RANDOMIZE_AUGMENTATION_OPERATIONS = True

    # Toggle additional augmentation ops
    APPLY_COLOR_JITTER = True
    APPLY_GAUSSIAN_NOISE = True
    APPLY_BLUR = True
    APPLY_RANDOM_CROP = False
    # Keep distribution balanced across ops when augmenting
    AUGMENTATION_EQUAL_DISTRIBUTION = True

    # How many times to attempt different augmentations per image when
    # balancing. If set to None, will compute evenly based on needed count.
    AUGMENTATION_MAX_ATTEMPTS_PER_SOURCE = None

    # Maximum number of augmented images generated from a single source image, default = 10
    MAX_AUG_PER_IMAGE = 10

    # Color jitter parameters (brightness, contrast, saturation)
    COLOR_JITTER_BRIGHTNESS = 0.2
    COLOR_JITTER_CONTRAST = 0.2
    COLOR_JITTER_SATURATION = 0.15

    # Gaussian noise parameters
    GAUSSIAN_NOISE_STD = 5.0  # pixel intensity stddev

    # Blur parameters
    BLUR_KERNEL_SIZES = [3, 5]  # odd kernel sizes

    # Random crop parameters (fraction of area to keep)
    RANDOM_CROP_RANGE = (0.8, 1.0)

    # Downsampling/selection policy when class has more than TARGET_IMAGES_PER_CLASS
    # Options: 'random', 'sharpness', 'sharpness_then_diversity'
    DOWNSAMPLE_METHOD = 'sharpness_then_diversity'
    # When using sharpness_then_diversity we first pick a sharpness pool
    # of size TARGET_IMAGES_PER_CLASS * DOWNSAMPLE_POOL_FACTOR, then select a
    # diversified subset from it.
    DOWNSAMPLE_POOL_FACTOR = 2

    # Rotation parameters
    APPLY_ROTATION = True
    ROTATION_ANGLES = [-15, -10, -5, 5, 10, 15]  # Rotation angles in degrees

    # Flip parameters
    APPLY_FLIP = True
    FLIP_HORIZONTAL = True
    FLIP_VERTICAL = True
    FLIP_BOTH = True  # Flip both horizontal and vertical

    # Translation parameters (shift image)
    APPLY_TRANSLATION = True
    TRANSLATION_RANGE = 0.1  # 10% of image size in any direction

    # Scale/Zoom parameters
    APPLY_SCALE = True
    SCALE_RANGE = (0.9, 1.1)  # Scale factor range (0.9 = 90%, 1.1 = 110%)

    # Shear/Affine transformation
    APPLY_SHEAR = True
    SHEAR_RANGE = 10  # Shear angle in degrees
    
    # ==================== TRAIN/VAL/TEST SPLIT CONFIGURATION ====================
    # Split ratios (must sum to 1.0)
    TRAIN_RATIO = 0.8   # 80% for training
    VAL_RATIO = 0.1     # 10% for validation
    TEST_RATIO = 0.1    # 10% for testing

    # Random seed for reproducible splits
    RANDOM_SEED = 42

    # Create split directories
    CREATE_SPLIT_DIRS = True  # Creates train/val/test subdirectories

    # ==================== TESTING CONFIGURATION ====================
    # Test mode - process only a subset of images
    TEST_MODE = False
    TEST_SIZE = 50  # Number of images to process in test mode (per class if possible)

    # ==================== LOGGING CONFIGURATION ====================
    VERBOSE = True  # Print detailed logs to console

    # ==================== VALIDATION ====================
    @classmethod
    def validate(cls):
        """Validate configuration parameters"""
        errors = []

        # Validate paths
        if not os.path.exists(cls.INPUT_DIR):
            errors.append(f"Input directory does not exist: {cls.INPUT_DIR}")

        # Validate target size
        if not (isinstance(cls.TARGET_SIZE, tuple) and len(cls.TARGET_SIZE) == 2):
            errors.append("TARGET_SIZE must be a tuple of (width, height)")

        # Validate target images per class
        if cls.TARGET_IMAGES_PER_CLASS < 0:
            errors.append("TARGET_IMAGES_PER_CLASS must be greater or equal to 0 (highest img count as target)")

        # Validate translation range
        if not (0 <= cls.TRANSLATION_RANGE <= 0.5):
            errors.append("TRANSLATION_RANGE must be between 0 and 0.5")

        # Validate scale range
        if cls.SCALE_RANGE[0] >= cls.SCALE_RANGE[1]:
            errors.append("SCALE_RANGE must be (min, max) where min < max")

        # Validate split ratios
        total_ratio = cls.TRAIN_RATIO + cls.VAL_RATIO + cls.TEST_RATIO
        if abs(total_ratio - 1.0) > 0.001:
            errors.append(f"Train/Val/Test ratios must sum to 1.0 (current: {total_ratio})")

        if cls.TRAIN_RATIO <= 0 or cls.VAL_RATIO <= 0 or cls.TEST_RATIO <= 0:
            errors.append("All split ratios must be greater than 0")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

        return True

    @classmethod
    def compute_augmentation_plan(cls, class_name: str, current_count: int, target: int):
        """Return a dict mapping augmentation operation -> number to generate

        Ensures total generated images = TARGET_IMAGES_PER_CLASS - current_count
        and (when AUGMENTATION_EQUAL_DISTRIBUTION=True) distributes them as
        evenly as possible across the enabled ops.
        """

        #get class specific augemntaions else default to target
        target = cls.CLASS_AUGMENTATION_TARGETS.get(class_name, target)

        need = target - current_count

        # Build enabled ops list from flags and master list
        enabled = []
        for op in cls.AUGMENTATION_OPERATIONS:
            if op == 'rotation' and cls.APPLY_ROTATION:
                enabled.append(op)
            elif op == 'flip' and cls.APPLY_FLIP:
                enabled.append(op)
            elif op == 'translation' and cls.APPLY_TRANSLATION:
                enabled.append(op)
            elif op == 'scale' and cls.APPLY_SCALE:
                enabled.append(op)
            elif op == 'shear' and cls.APPLY_SHEAR:
                enabled.append(op)
            elif op == 'color_jitter' and cls.APPLY_COLOR_JITTER:
                enabled.append(op)
            elif op == 'gaussian_noise' and cls.APPLY_GAUSSIAN_NOISE:
                enabled.append(op)
            elif op == 'blur' and cls.APPLY_BLUR:
                enabled.append(op)
            elif op == 'random_crop' and cls.APPLY_RANDOM_CROP:
                enabled.append(op)

        if not enabled:
            # Fallback to rotation if nothing enabled to avoid empty plan
            enabled = ['rotation']

        plan = {op: 0 for op in enabled}

        if cls.AUGMENTATION_EQUAL_DISTRIBUTION:
            base = need // len(enabled)
            rem = need % len(enabled)
            for i, op in enumerate(enabled):
                plan[op] = base + (1 if i < rem else 0)
        else:
            # If not equal distribution, put more emphasis on geometric transforms
            weights = []
            for op in enabled:
                if op in ('rotation', 'flip', 'translation', 'scale', 'shear'):
                    weights.append(2.0)
                else:
                    weights.append(1.0)
            weights = np.array(weights) / np.sum(weights)
            counts = np.floor(weights * need).astype(int)
            # adjust to match exactly
            diff = need - np.sum(counts)
            idx = 0
            while diff > 0:
                counts[idx % len(counts)] += 1
                idx += 1
                diff -= 1
            for op, c in zip(enabled, counts):
                plan[op] = int(c)

        return plan

    @classmethod
    def select_images_to_keep(cls, file_paths, target=None):
        """Select a subset of images to keep when class has more than target.

        Strategy:
        - If DOWNSAMPLE_METHOD == 'random' -> random sample
        - If 'sharpness' -> sort by variance of Laplacian (sharpness) and pick top
        - If 'sharpness_then_diversity' -> take top (target*pool_factor) by sharpness
          then greedily pick a diversified subset from their HSV histograms.

        Returns list of selected file paths.
        """
        if target is None:
            target = cls.TARGET_IMAGES_PER_CLASS

        n = len(file_paths)
        if n <= target:
            return list(file_paths)

        if cls.DOWNSAMPLE_METHOD == 'random':
            rng = np.random.default_rng(cls.RANDOM_SEED)
            chosen = list(rng.choice(file_paths, size=target, replace=False))
            return chosen

        # compute sharpness scores
        from tqdm import tqdm
        print(f"Computing sharpness scores for {len(file_paths)} images...")
        scores = []
        for p in tqdm(file_paths, desc="Computing sharpness", unit="img"):
            try:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    scores.append((p, -1.0))
                    continue
                scores.append((p, float(cv2.Laplacian(img, cv2.CV_64F).var())))
            except Exception:
                scores.append((p, -1.0))

        # sort by sharpness desc
        scores.sort(key=lambda x: x[1], reverse=True)

        if cls.DOWNSAMPLE_METHOD == 'sharpness':
            selected = [p for p, s in scores[:target]]
            return selected

        # sharpness_then_diversity
        pool_size = min(len(scores), int(target * cls.DOWNSAMPLE_POOL_FACTOR))
        pool = [p for p, s in scores[:pool_size]]

        # compute HSV histograms for pool
        print(f"Computing HSV histograms for diversity selection ({pool_size} images)...")
        hists = []
        for p in tqdm(pool, desc="Computing histograms", unit="img"):
            img = cv2.imread(p)
            if img is None:
                hists.append(None)
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hists.append(hist)

        # greedy diversity selection: start with the sharpest image
        print(f"Selecting {target} diverse images from pool...")
        selected = [pool[0]]
        selected_hists = [hists[0]]

        def hist_distance(a, b):
            if a is None or b is None:
                return 0.0
            # use correlation distance
            return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

        idxs = list(range(1, len(pool)))
        pbar = tqdm(total=target-1, desc="Selecting diverse images", unit="img")
        while len(selected) < target and idxs:
            best_idx = None
            best_score = -1.0
            for i in idxs:
                # compute min distance to existing selected
                dists = [hist_distance(hists[i], sh) for sh in selected_hists]
                min_d = min(dists) if dists else 0.0
                # prefer items with larger min distance (more diverse)
                if min_d > best_score:
                    best_score = min_d
                    best_idx = i
            if best_idx is None:
                break
            selected.append(pool[best_idx])
            selected_hists.append(hists[best_idx])
            idxs.remove(best_idx)
            pbar.update(1)
        pbar.close()

        # If still under target (rare), pad with next sharpest
        if len(selected) < target:
            extras = [p for p, s in scores if p not in selected]
            for e in extras:
                if len(selected) >= target:
                    break
                selected.append(e)

        return selected

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*60)
        print("PREPROCESSING CONFIGURATION")
        print("="*60)
        print(f"Input Directory: {cls.INPUT_DIR}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print(f"Test Mode: {cls.TEST_MODE} (Size: {cls.TEST_SIZE if cls.TEST_MODE else 'N/A'})")

        print("\nDuplicate Removal:")
        print(f"  Method: MD5 (exact match)")

        print("\nResizing:")
        print(f"  Target Size: {cls.TARGET_SIZE}")
        print(f"  Interpolation: cv2.INTER_LANCZOS4")

        print("\nNormalization:")
        print(f"  Apply: {cls.APPLY_NORMALIZATION}")
        print(f"  Method: {cls.NORMALIZATION_METHOD}")

        print("\nAugmentation:")
        print(f"  Target images per class: {cls.TARGET_IMAGES_PER_CLASS}")
        print(f"  Rotation: {cls.APPLY_ROTATION} (Angles: {cls.ROTATION_ANGLES})")
        print(f"  Flip: {cls.APPLY_FLIP} (H:{cls.FLIP_HORIZONTAL}, V:{cls.FLIP_VERTICAL}, Both:{cls.FLIP_BOTH})")
        print(f"  Translation: {cls.APPLY_TRANSLATION} (Range: {cls.TRANSLATION_RANGE})")
        print(f"  Scale: {cls.APPLY_SCALE} (Range: {cls.SCALE_RANGE})")
        print(f"  Shear: {cls.APPLY_SHEAR} (Range: +/-{cls.SHEAR_RANGE} degrees)")

        print("\nTrain/Val/Test Split:")
        print(f"  Train: {cls.TRAIN_RATIO*100:.0f}%")
        print(f"  Val: {cls.VAL_RATIO*100:.0f}%")
        print(f"  Test: {cls.TEST_RATIO*100:.0f}%")
        print(f"  Random Seed: {cls.RANDOM_SEED}")
        print("="*60)
