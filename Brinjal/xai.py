"""
Explainable AI (XAI) Module for Cotton Disease Classification
Provides visualization of what features the model uses to make predictions
Includes: Grad-CAM, Grad-CAM++, and attention visualization
"""
 
from xml.parsers.expat import model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ppnet_hybrid import PPNetHybrid
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
import json
import argparse
from train import get_model
import config
 
 
class GradCAM:
    """
    Generates Grad-CAM (Gradient-weighted Class Activation Map) for visualization
    Shows which regions of the image contributed most to the model's prediction
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
 
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
 
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
 
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
 
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward()
 
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_np = cam.squeeze().cpu().numpy()
        cam_min = np.percentile(cam_np, 2)
        cam_max = np.percentile(cam_np, 98)
 
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            cam_np = np.clip(cam_np, 0, 1)
 
        cam_np = np.power(cam_np, 0.8)
        return cam_np
 
 
class GradCAMPlusPlus:
    """
    Generates Grad-CAM++ (improved version of Grad-CAM)
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
 
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
 
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward(retain_graph=True)
 
        gradients = self.gradients
        activations = self.activations
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        spatial_sum = (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        alpha_denom = 2 * grad_2 + spatial_sum
        alpha = grad_2 / (alpha_denom + 1e-10)
        relu_grad = F.relu(gradients)
        weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_np = cam.squeeze().cpu().numpy()
        cam_min = np.percentile(cam_np, 2)
        cam_max = np.percentile(cam_np, 98)
 
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            cam_np = np.clip(cam_np, 0, 1)
 
        cam_np = np.power(cam_np, 0.8)
        return cam_np
 
 
def get_target_layer(model, model_name):

    if model_name in ['resnet18', 'resnet34', 'resnet50']:
        if hasattr(model, 'cbam'):
            return model.cbam
        return model.layer4[-1]
    elif model_name == 'ppnet_ms':
        return model.se
    elif model_name == 'ppnet_hybrid':
        # PPNetHybrid keeps layer4 fully intact as the CNN's spatial feature
        # map — Grad-CAM works normally here even though the classifier also
        # fuses in the transformer's global [CLS] feature downstream.
        return model.layer4[-1]
    elif model_name == 'mobilenet_v2':
        return model.features[-1]
    elif model_name == 'efficientnet_b0':
        return model.features[-1]
    elif model_name == 'shufflenet_v2':
        return model.conv5
    elif model_name == 'densenet121':
        return model.features.denseblock4
    elif model_name == 'lite_mdc':
        return model.features[-1]
    else:
        raise ValueError(f"Model {model_name} not supported")
    
def overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay
 
 
def explain_single_prediction(image_path, model_path, model_name=None, output_dir=None, true_label=None):
    """
    Generate complete explainability report for a single image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
   
    class_to_idx_path = os.path.splitext(model_path)[0] + '_class_to_idx.json'
    if not os.path.exists(class_to_idx_path):
        print(f"Error: Class mapping not found at {class_to_idx_path}")
        return
    
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
   
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
   
    if model_name is None:
        model_name = os.path.basename(os.path.splitext(model_path)[0]).split('_')[0]
        if model_name not in config.TrainingConfig.AVAILABLE_MODELS:
            model_name = config.TrainingConfig.MODEL_NAME
   
    model = get_model(model_name, num_classes)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # fc.weight -> fc.1.weight (old plain Linear -> new Sequential+Dropout)
    if "fc.weight" in state_dict and "fc.1.weight" not in state_dict:
        state_dict["fc.1.weight"] = state_dict.pop("fc.weight")
        state_dict["fc.1.bias"]   = state_dict.pop("fc.bias")

    # fc.1.weight -> fc.weight (new Sequential+Dropout -> old plain Linear)
    if "fc.1.weight" in state_dict and "fc.weight" not in state_dict:
        state_dict["fc.weight"] = state_dict["fc.1.weight"]
        state_dict["fc.bias"]   = state_dict["fc.1.bias"]

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])
   
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Error: Could not read image at {image_path}")
        return
   
    image_cv = cv2.resize(image_cv, config.IMAGE_SIZE)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)
   
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = outputs.argmax(dim=1).item()
        predicted_class = idx_to_class[predicted_idx]
        confidence = probabilities[0, predicted_idx].item()
   
    print(f"\n{'='*70}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.3f})")
    print(f"{'='*70}\n")
   
    print("All class probabilities:")
    for idx in sorted(idx_to_class.keys()):
        print(f"  {idx_to_class[idx]:20s}: {probabilities[0, idx].item():.4f}")
   
    print("\nGenerating Grad-CAM++ heatmap...")
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAMPlusPlus(model, target_layer)
    cam = grad_cam.generate(image_tensor, target_class=predicted_idx)
   
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.1)
    fig.suptitle(f'Explainability Report: {predicted_class} (Confidence: {confidence:.1%})',
                 fontsize=18, fontweight='bold', y=0.95)
   
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.imshow(image_rgb)
    actual_label_text = f"Original Image = {true_label if true_label else 'Unknown'}"
    ax1.set_title(actual_label_text, fontsize=13, fontweight='bold', pad=10)
    ax1.axis('off')
   
    ax2 = fig.add_subplot(gs[0:2, 1])
    im = ax2.imshow(cam, cmap='jet')
    ax2.set_title('Activation Heatmap', fontsize=13, fontweight='bold', color='red', pad=10)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
   
    ax3 = fig.add_subplot(gs[0:2, 2])
    overlay_img = overlay_heatmap(image_rgb, cam, alpha=0.5)
    ax3.imshow(overlay_img)
    pred_label_text = f"Pred: {predicted_class}"
    ax3.set_title(pred_label_text, fontsize=13, fontweight='bold', color='darkred', pad=15)
    ax3.axis('off')
   
    ax4 = fig.add_subplot(gs[2, 0:2])
    classes = sorted([idx_to_class[i] for i in range(num_classes)])
    confs = [probabilities[0, class_to_idx[c]].item() for c in classes]
    colors_list = ['#2ecc71' if c == predicted_class else '#3498db' for c in classes]

    bars = ax4.barh(classes, confs, color=colors_list, edgecolor='black', linewidth=2, height=0.6)
    ax4.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax4.set_title('Model Confidence per Class', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 1.1)
    ax4.grid(axis='x', alpha=0.2, linestyle='--')

    for bar, conf in zip(bars, confs):
        ax4.text(conf + 0.01, bar.get_y() + bar.get_height()/2, f'{conf:.1%}', va='center', fontweight='bold')
   
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
   
    cam_std = np.std(cam)
    high_importance_region = np.sum(cam > 0.7) # Using a fixed threshold for analysis
    importance_pct = (high_importance_region / cam.size) * 100
   
    stats_text = (
        f"ANALYSIS SUMMARY\n"
        f"{'-'*20}\n"
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"Focus Metrics:\n"
        f"• Feature Spread (Std): {cam_std:.3f}\n"
        f"• Heat Coverage: {importance_pct:.1f}%\n\n"
        f"Top Alternative:\n"
        f"• {classes[np.argsort(confs)[-2]]}: {confs[np.argsort(confs)[-2]]*100:.1f}%"
    )

    ax5.text(0.0, 0.8, stats_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    
    plt.subplots_adjust(top=0.88, bottom=0.1, left=0.05, right=0.95)
    
    if output_dir is None: output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'xai_{os.path.basename(image_path)}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved report: {os.path.basename(image_path)}")
   
    # Generate detailed attention map
    print("\nGenerating detailed attention map...")
    fig_attn = plt.figure(figsize=(20, 12))
    gs = fig_attn.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    fig_attn.suptitle(f'Detailed Attention Analysis for {predicted_class}',
                     fontsize=16, fontweight='bold')
   
    thresholds = [0.25, 0.5, 0.75]
    for idx, threshold in enumerate(thresholds):
        ax1 = fig_attn.add_subplot(gs[0, idx])
        cam_thresholded = np.where(cam > threshold, cam, 0)
        im1 = ax1.imshow(cam_thresholded, cmap='hot')
        ax1.set_title(f'Attention (Threshold > {threshold})', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
       
        ax2 = fig_attn.add_subplot(gs[1, idx])
        overlay = overlay_heatmap(image_rgb, cam_thresholded, alpha=0.6)
        ax2.imshow(overlay)
        ax2.set_title(f'Overlay (Threshold > {threshold})', fontsize=12, fontweight='bold')
        ax2.axis('off')
   
    attn_output_path = os.path.join(output_dir, f'attention_map_{os.path.basename(image_path)}.png')
    fig_attn.savefig(attn_output_path, dpi=150, bbox_inches='tight')
    plt.close()
 
 
def batch_explain(test_dir, model_path, model_name=None, output_dir=None, num_samples_per_class=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx_path = os.path.splitext(model_path)[0] + '_class_to_idx.json'
    if not os.path.exists(class_to_idx_path):
        print(f"Error: Class mapping not found at {class_to_idx_path}")
        return
    
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
   
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
   
    if model_name is None:
        model_name = os.path.basename(os.path.splitext(model_path)[0]).split('_')[0]
        if model_name not in config.TrainingConfig.AVAILABLE_MODELS:
            model_name = config.TrainingConfig.MODEL_NAME
   
    model = get_model(model_name, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    # fc key remapping (old plain Linear -> new Sequential+Dropout)
    if "fc.weight" in state_dict and "fc.1.weight" not in state_dict:
        state_dict["fc.1.weight"] = state_dict.pop("fc.weight")
        state_dict["fc.1.bias"]   = state_dict.pop("fc.bias")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
   
    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(model_path))
   
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating batch explainability reports from: {test_dir}")
   
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
       
        print(f"Processing class: {class_name}")
        image_files = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
       
        image_files = image_files[:num_samples_per_class]
       
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            if "Actual_" in image_file:
                extracted_label = image_file.split('Actual_')[1].split('_Pred')[0]
            else:
                extracted_label = class_name
            
            try:
                explain_single_prediction(image_path, model_path, model_name, output_dir, true_label=extracted_label)
            except Exception as e:
                print(f"  ✗ {image_file}: {str(e)}")
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainable AI (XAI) visualization')
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None, choices=config.TrainingConfig.AVAILABLE_MODELS)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=10)
   
    args = parser.parse_args()
    if not args.model_path:
        args.model_path = os.path.join(os.getcwd(), 'models', 'trained_model.pth')
   
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)
   
    if args.image_path:
        inferred_label = os.path.basename(os.path.dirname(args.image_path))
        explain_single_prediction(args.image_path, args.model_path, args.model_name, args.output_dir, true_label=inferred_label)
    elif args.test_dir:
        batch_explain(args.test_dir, args.model_path, args.model_name, args.output_dir, args.num_samples)
    else:
        print("Error: Provide either --image_path or --test_dir")
        exit(1)