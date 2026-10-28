from pyexpat import model

import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent GUI windows from opening
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import cv2
import json
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from ppnet_hybrid import PPNetHybrid
from tqdm import tqdm
from train import get_model
from xai import GradCAMPlusPlus, get_target_layer, overlay_heatmap
import config
import warnings
from train_ppnet_ms import PPNetMS
# Silence PyTorch deprecation and user warnings to keep terminal outputs neat and progress bar readable
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_inferred_model_name(model_path):
    """
    Robustly infers the model architecture name from the weight filename.
    Searches config.TrainingConfig.AVAILABLE_MODELS. Falls back to config default.
    """
    base_name = os.path.basename(model_path).lower()
    
    # Try exact match first
    for m in config.TrainingConfig.AVAILABLE_MODELS:
        if m.lower() in base_name:
            return m
            
    # Try matching parts of the name
    for m in config.TrainingConfig.AVAILABLE_MODELS:
        base_name_clean = base_name.replace('_', '').replace('-', '')
        m_clean = m.lower().replace('_', '').replace('-', '')
        if m_clean in base_name_clean:
            return m
            
    # Fallback to the configured model name
    return config.TrainingConfig.MODEL_NAME

def generate_prediction_report_image(image_path, image_rgb, overlay_img, actual_class, raw_predicted_class, predicted_class, confidence, met_threshold, is_correct, probabilities, idx_to_class, output_path):
    """
    Generates a premium 2x2 grid report image containing:
    1. Top-Left: Original input image with filename as title
    2. Top-Right: Grad-CAM attention heatmap overlay
    3. Bottom-Left: Styled textual PREDICTION RESULTS with dynamic styling for Low Confidence
    4. Bottom-Right: Raw Class Probabilities horizontal bar chart
    """
    num_classes = len(idx_to_class)
    
    # Create figure matching the aspect ratio of the target design
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    
    # 1. Top-Left: Input Image
    axs[0, 0].imshow(image_rgb)
    axs[0, 0].set_title(f"Input Image: {os.path.basename(image_path)}", fontsize=12, pad=10)
    axs[0, 0].axis('off')
    
    # 2. Top-Right: Grad-CAM Attention
    axs[0, 1].imshow(overlay_img)
    axs[0, 1].set_title("Grad-CAM Attention", fontsize=12, pad=10)
    axs[0, 1].axis('off')
    
    # 3. Bottom-Left: Prediction Results
    ax_text = axs[1, 0]
    ax_text.axis('off')
    
    # Draw text fields with precise spacing and sizes
    ax_text.text(0.05, 0.85, "PREDICTION RESULTS", fontsize=16, fontweight='bold', ha='left', va='top')
    
    # Ground truth in blue
    ax_text.text(0.05, 0.65, f"Ground Truth: {actual_class}", fontsize=13, color='blue', ha='left', va='top')
    
    if not met_threshold:
        # Style differently for Low Confidence
        ax_text.text(0.05, 0.50, f"Predicted:  Low Confidence\n(Model: {raw_predicted_class})", fontsize=13, fontweight='bold', color='#d35400', ha='left', va='top')
        ax_text.text(0.05, 0.30, f"Confidence:  {confidence:.2f}%", fontsize=12, color='#333333', ha='left', va='top')
        ax_text.text(0.05, 0.15, "✗ LOW CONFIDENCE", fontsize=13, fontweight='bold', color='#d35400', ha='left', va='top')
    else:
        # Style normally for Accepted Predictions
        pred_color = 'green' if is_correct else 'red'
        ax_text.text(0.05, 0.50, f"Predicted:  {predicted_class}", fontsize=14, fontweight='bold', color=pred_color, ha='left', va='top')
        ax_text.text(0.05, 0.35, f"Confidence:  {confidence:.2f}%", fontsize=12, color='#333333', ha='left', va='top')
        
        if is_correct:
            ax_text.text(0.05, 0.20, "✓ CORRECT", fontsize=13, fontweight='bold', color='green', ha='left', va='top')
        else:
            ax_text.text(0.05, 0.20, "✗ INCORRECT", fontsize=13, fontweight='bold', color='red', ha='left', va='top')
        
    # 4. Bottom-Right: Raw Class Probabilities Horizontal Bar plot
    ax_bar = axs[1, 1]
    
    # Fetch top-5 classes for the bar chart
    top_k = min(5, num_classes)
    top_probs_fig, top_indices_fig = torch.topk(probabilities, top_k)
    top_probs_fig = top_probs_fig.cpu().numpy()
    top_indices_fig = top_indices_fig.cpu().numpy()
    
    fig_classes = [idx_to_class[idx] for idx in top_indices_fig]
    
    # Invert lists so highest value is plotted at the top of the horizontal chart
    fig_classes = fig_classes[::-1]
    top_probs_fig = top_probs_fig[::-1]
    
    # Dynamic bar coloring: green for predicted, light blue for others (use orange if low confidence)
    if not met_threshold:
        bar_colors = ['#a2d2ff' if c != raw_predicted_class else '#e67e22' for c in fig_classes]
    else:
        bar_colors = ['#a2d2ff' if c != predicted_class else '#2ecc71' for c in fig_classes]
        
    bars = ax_bar.barh(fig_classes, top_probs_fig, color=bar_colors, edgecolor='black', linewidth=1, height=0.5)
    
    ax_bar.set_title("Raw Class Probabilities", fontsize=12, pad=10)
    ax_bar.set_xlabel("Probability", fontsize=10)
    ax_bar.set_xlim(0, 1.05) # extra limit buffer for percentage label spacing
    ax_bar.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Draw horizontal percentage value label on top of each bar
    for bar, prob in zip(bars, top_probs_fig):
        ax_bar.text(
            prob + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{prob*100:.1f}%",
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )
        
    # Adjust layout to remove overlapping labels and spacing issues
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # Save high-resolution report image
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_batch_predictions(data_dir, model_path, output_dir=None, threshold=0.0):
    """
    Runs batch predictions on images inside data_dir organized by class folders.
    Saves premium visual report images (2x2 grid), multi-sheet Excel reports, and summary text files.
    """
    # 1. Device and Path setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT] Using device: {device}")
    
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"batch_predictions_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, "prediction_images")
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Convert threshold to percentage scale if specified as a decimal < 1.0
    if 0.0 < threshold <= 1.0:
        threshold *= 100.0
    print(f"[INIT] Confidence Safety Threshold: {threshold:.2f}%")
    
    # 2. Load Class Mappings
    class_to_idx_path = os.path.splitext(model_path)[0] + '_class_to_idx.json'
    if not os.path.exists(class_to_idx_path):
        # Fallback to checking same directory as model
        model_dir = os.path.dirname(model_path)
        class_to_idx_path = os.path.join(model_dir, 'resnet50_class_to_idx.json') # fallback to default
        if not os.path.exists(class_to_idx_path):
            raise FileNotFoundError(f"Class mapping file not found at expected paths. Make sure it exists near the model file.")
            
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
    print(f"[INIT] Loaded {num_classes} classes from {class_to_idx_path}")
    
    # 3. Load the Model
    model_name = get_inferred_model_name(model_path)
    print(f"[INIT] Inferred architecture '{model_name}' from model weights path.")
    
    if 'ppnet_hybrid' in model_path.lower():
        model = PPNetHybrid(num_classes=num_classes)
        model_name = 'ppnet_hybrid'
        print("[INIT] Detected PPNetHybrid architecture from model path.")
    elif 'ppnet_ms' in model_path.lower():
        model = PPNetMS(num_classes=num_classes)
        model_name = 'ppnet_ms'
        print("[INIT] Detected PPNetMS architecture from model path.")
    else:
        model = get_model(model_name, num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle mismatch between old fc (Linear) and new fc (Sequential with Dropout)
    if "fc.weight" in state_dict and "fc.1.weight" not in state_dict:
        # Old checkpoint has plain Linear head, remap keys
        state_dict["fc.1.weight"] = state_dict.pop("fc.weight")
        state_dict["fc.1.bias"] = state_dict.pop("fc.bias")
    checkpoint = torch.load(model_path, map_location=device)

    

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # --- State dict key remapping for head architecture mismatches ---

    # ResNet / ShuffleNet: old plain Linear (fc.*) -> new Sequential+Dropout (fc.1.*)
    if "fc.weight" in state_dict and "fc.1.weight" not in state_dict:
        state_dict["fc.1.weight"] = state_dict.pop("fc.weight")
        state_dict["fc.1.bias"]   = state_dict.pop("fc.bias")

    # ResNet / ShuffleNet: new Sequential+Dropout (fc.1.*) -> old plain Linear (fc.*)
    # (shouldn't happen in practice but included for symmetry)
    # Not needed since your current train.py always saves Sequential heads for these.

    # DenseNet: old plain Linear (classifier.*) -> new Sequential+Dropout (classifier.1.*)
    if "classifier.weight" in state_dict and "classifier.1.weight" not in state_dict:
        # Check it's not lite_mdc (lite_mdc uses plain Linear, so we should NOT remap up)
        if model_name != "lite_mdc":
            state_dict["classifier.1.weight"] = state_dict.pop("classifier.weight")
            state_dict["classifier.1.bias"]   = state_dict.pop("classifier.bias")

    # Lite-MDC: checkpoint saved Sequential+Dropout keys (classifier.1.*) 
    # but Lite-MDC model uses plain Linear (classifier.*)
    if model_name == "lite_mdc" and "classifier.1.weight" in state_dict and "classifier.weight" not in state_dict:
        state_dict["classifier.weight"] = state_dict.pop("classifier.1.weight")
        state_dict["classifier.bias"]   = state_dict.pop("classifier.1.bias")

    # MobileNetV2: its built-in head is already classifier.1.* so no remapping needed.
        # EfficientNet-B0: its built-in head is already classifier.1.* so no remapping needed.

    model.load_state_dict(state_dict, strict=False)
   
        
    model.to(device)
    model.eval()
    print("[INIT] Model successfully loaded and set to evaluation mode.")
    
    # 4. Prepare Image Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])
    
    # 5. Initialize Grad-CAM++ Generator
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAMPlusPlus(model, target_layer)
    print("[INIT] Registered Grad-CAM++ Hooks on target layer.")
    
    # 6. Scan directory for images
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_paths = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_paths.append(os.path.join(root, file))
                
    total_images = len(image_paths)
    if total_images == 0:
        print(f"[ERROR] No images with supported extensions {supported_extensions} found in '{data_dir}'")
        return
        
    print(f"[SCAN] Found {total_images} images to classify across subfolders.\n")
    
    # 7. Process Images
    detailed_records = []
    class_stats = {} # { class_name: { 'total': 0, 'correct': 0, 'low_conf': 0 } }
    
    for img_path in tqdm(image_paths, desc="Processing Predictions", unit="img"):
        # Infer actual class from the immediate parent directory
        parent_dir = os.path.basename(os.path.dirname(img_path))
        actual_class = parent_dir
        
        # Initialize stats for this class if not present
        if actual_class not in class_stats:
            class_stats[actual_class] = {'total': 0, 'correct': 0, 'low_conf': 0}
        class_stats[actual_class]['total'] += 1
        
        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] Failed to load image: {img_path}")
            continue
            
        # Get RGB version at model size for transformation and inference
        img_resized = cv2.resize(img_bgr, config.IMAGE_SIZE)
        img_rgb_model = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Get RGB version of the original image
        img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Infer
        img_tensor = transform(img_rgb_model).unsqueeze(0).to(device)
        
        # Calculate logits and class probabilities (automatic target_class)
        cam = grad_cam.generate(img_tensor, target_class=None)
        
        # Re-run forward pass without gradients to grab raw predictions
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        # Extract predictions
        top_probs, top_indices = torch.topk(probs, min(5, num_classes))
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Map top-1 prediction and safety threshold check
        predicted_idx = top_indices[0]
        raw_predicted_class = idx_to_class[predicted_idx]
        predicted_confidence = top_probs[0] * 100 # percentage
        
        met_threshold = predicted_confidence >= threshold
        predicted_class = raw_predicted_class if met_threshold else "Low Confidence"
        is_correct = (predicted_class == actual_class)
        
        if not met_threshold:
            class_stats[actual_class]['low_conf'] += 1
        elif is_correct:
            class_stats[actual_class]['correct'] += 1
            
        # Generate Grad-CAM Heatmap overlay
        overlay_img = overlay_heatmap(img_rgb_original, cam, alpha=0.5)
        
        # Generate and save the beautiful 2x2 grid report image
        class_out_dir = os.path.join(images_output_dir, actual_class)
        os.makedirs(class_out_dir, exist_ok=True)
        report_output_path = os.path.join(class_out_dir, os.path.basename(img_path))
        
        generate_prediction_report_image(
            image_path=img_path,
            image_rgb=img_rgb_original,
            overlay_img=overlay_img,
            actual_class=actual_class,
            raw_predicted_class=raw_predicted_class,
            predicted_class=predicted_class,
            confidence=predicted_confidence,
            met_threshold=met_threshold,
            is_correct=is_correct,
            probabilities=probs,
            idx_to_class=idx_to_class,
            output_path=report_output_path
        )
        
        # Compile top-5 predictions list for Excel sheet
        top_preds = []
        for i in range(5):
            if i < len(top_indices):
                cls_name = idx_to_class[top_indices[i]]
                prob = top_probs[i] * 100
                top_preds.append(f"{cls_name} ({prob:.2f}%)")
            else:
                top_preds.append("N/A")
                
        # Record detailed information
        detailed_records.append({
            "Image Name": os.path.basename(img_path),
            "Actual Class": actual_class,
            "Raw Model Prediction": raw_predicted_class,
            "Confidence (%)": round(predicted_confidence, 2),
            "Met Threshold": "Yes" if met_threshold else "No",
            "Final Prediction": predicted_class,
            "Correct": "Yes" if is_correct else "No",
            "Top 1 Prediction": top_preds[0],
            "Top 2 Prediction": top_preds[1],
            "Top 3 Prediction": top_preds[2],
            "Top 4 Prediction": top_preds[3],
            "Top 5 Prediction": top_preds[4],
            "Relative Path": os.path.relpath(img_path, data_dir)
        })
        
    # 8. Generate Summary Calculations
    total_processed = len(detailed_records)
    total_correct = sum(1 for r in detailed_records if r["Correct"] == "Yes")
    total_low_conf = sum(1 for r in detailed_records if r["Met Threshold"] == "No")
    total_incorrect = total_processed - total_correct - total_low_conf
    overall_accuracy = (total_correct / total_processed) * 100 if total_processed > 0 else 0
    overall_low_conf_pct = (total_low_conf / total_processed) * 100 if total_processed > 0 else 0
    
    # 9. Create Pandas DataFrames
    df_detail = pd.DataFrame(detailed_records)
    
    # Overall summary dataframe
    overall_summary_data = [
        {"Metric": "Total Images Processed", "Value": str(total_processed)},
        {"Metric": "Total Correct Predictions (High Conf)", "Value": str(total_correct)},
        {"Metric": "Total Low Confidence Predictions", "Value": str(total_low_conf)},
        {"Metric": "Total Incorrect Predictions (High Conf)", "Value": str(total_incorrect)},
        {"Metric": "Overall Accuracy (High Conf & Correct)", "Value": f"{overall_accuracy:.2f}%"},
        {"Metric": "Overall Low Confidence Rate", "Value": f"{overall_low_conf_pct:.2f}%"}
    ]
    df_overall = pd.DataFrame(overall_summary_data)
    
    # Per-class summary dataframe
    class_summary_records = []
    for cls_name, stats in sorted(class_stats.items()):
        cls_total = stats['total']
        cls_correct = stats['correct']
        cls_low_conf = stats['low_conf']
        cls_incorrect = cls_total - cls_correct - cls_low_conf
        cls_acc = (cls_correct / cls_total) * 100 if cls_total > 0 else 0
        cls_low_pct = (cls_low_conf / cls_total) * 100 if cls_total > 0 else 0
        
        class_summary_records.append({
            "Class Name": cls_name,
            "Total Images": cls_total,
            "Correct (High Conf)": cls_correct,
            "Low Confidence": cls_low_conf,
            "Incorrect (High Conf)": cls_incorrect,
            "Accuracy (%)": round(cls_acc, 2),
            "Low Conf Rate (%)": round(cls_low_pct, 2)
        })
    df_class = pd.DataFrame(class_summary_records)
    
    # 10. Save to Excel
    excel_path = os.path.join(output_dir, "predictions_report.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_detail.to_excel(writer, sheet_name='Detailed Predictions', index=False)
        
        # Write both summary tables to Summary Report sheet
        df_overall.to_excel(writer, sheet_name='Summary Report', index=False, startrow=0)
        
        # Write per-class breakdown with space
        df_class.to_excel(writer, sheet_name='Summary Report', index=False, startrow=len(df_overall) + 3)
        
        # Styling sheet column widths dynamically
        workbook = writer.book
        
        # Sheet 1 autofit
        ws1 = writer.sheets['Detailed Predictions']
        for col in ws1.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            ws1.column_dimensions[col_letter].width = max(max_len + 3, 12)
            
        # Sheet 2 autofit
        ws2 = writer.sheets['Summary Report']
        for col in ws2.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            ws2.column_dimensions[col_letter].width = max(max_len + 3, 12)
            
    # Save a text version of summary report
    summary_txt_path = os.path.join(output_dir, "summary_report.txt")
    with open(summary_txt_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("                BATCH PREDICTION SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Data Dir: {data_dir}\n")
        f.write(f"Threshold: {threshold:.2f}%\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Images:              {total_processed}\n")
        f.write(f"Correct (High Conf):       {total_correct}\n")
        f.write(f"Low Confidence:            {total_low_conf}\n")
        f.write(f"Incorrect (High Conf):     {total_incorrect}\n")
        f.write(f"Accuracy (High & Correct): {overall_accuracy:.2f}%\n")
        f.write(f"Low Conf Rate:             {overall_low_conf_pct:.2f}%\n")
        f.write("="*60 + "\n\n")
        f.write("PER-CLASS breakdown:\n")
        f.write(df_class.to_string(index=False))
        f.write("\n")
        
    # 11. Print beautiful terminal report
    print("\n" + "="*80)
    print("                      BATCH PREDICTION REPORT")
    print("="*80)
    print(f"Total Processed: {total_processed}")
    print(f"Correct (High):  {total_correct}  ({overall_accuracy:.2f}%)")
    print(f"Low Confidence:  {total_low_conf}  ({overall_low_conf_pct:.2f}%)")
    print(f"Incorrect (High): {total_incorrect}  ({(100 - overall_accuracy - overall_low_conf_pct):.2f}%)")
    print("-"*80)
    print("PER-CLASS breakdown:")
    print(df_class.to_string(index=False))
    print("="*80)
    print(f"\n[SUCCESS] 2x2 Grid Prediction reports saved to: {images_output_dir}")
    print(f"[SUCCESS] Multi-sheet Excel report saved to:     {excel_path}")
    print(f"[SUCCESS] Summary text file report saved to:     {summary_txt_path}")
    print("="*80 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch prediction pipeline on directories containing class folders.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing class subfolders of images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pth model weights file.')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory to save reports and overlay images.')
    parser.add_argument('--threshold', type=float, default=0.0, help='Confidence threshold (0-100 or 0.0-1.0) below which predictions are classified as Low Confidence.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        parser.error(f"Input directory does not exist: {args.data_dir}")
    if not os.path.exists(args.model_path):
        parser.error(f"Model path does not exist: {args.model_path}")
        
    run_batch_predictions(args.data_dir, args.model_path, args.output_dir, args.threshold)
