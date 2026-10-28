# TIH Brinjal Project: Trained Models Catalog & Technical Specifications

This catalog provides an authoritative reference for all trained model weights, checkpoints, prototypes, and representations developed under the PoCRA / TIH CHANAKYA fellowship.

---

## 1. Quick Summary Table

| Category | Model File | Backbone / Architecture | Target Classes | Key Performance Metric | Recommended Script |
|---|---|---|---|---|---|
| **Dual-Branch (Best)** | `dual_branch_head.pth` | DINOv2 ViT-B/14 + Prototype Branch | 10 Brinjal classes | **84.43%** Unseen Domain Transfer | `eval_dual_branch.py` |
| **Dual-Branch Baseline** | `dual_branch_head_normal.pth` | DINOv2 ViT-B/14 without domain adapter | 10 Brinjal classes | 81.20% Domain Transfer | `eval_dual_branch.py` |
| **Baseline CNN** | `resnet50.pth` | ResNet-50 (ImageNet Pretrained + 2-Stage FT) | 10 Brinjal classes | 95.8% Source / 50.3% Target | `predict.py` / `predict_batch.py` |
| **Domain-Invariant** | `domain_invariant_head.pth` | DINOv2 ViT-B/14 + Domain Adversarial Loss | 10 Brinjal classes | 83.15% Domain Transfer | `eval_dual_branch.py` |
| **Pest Disambiguator** | `disambig_pests.pth` | DINOv2 ViT Specialist Head | Aphid, Jassid, Whitefly | 91.2% within-pest accuracy | `eval_cascade.py` |
| **Fruit Disambiguator** | `disambig_fruit.pth` | DINOv2 ViT Specialist Head | Phomopsis Fruit, Fruit Borer | 89.6% within-fruit accuracy | `eval_cascade.py` |
| **Modern ConvNeXt** | `modern_head_convnext.pth` | ConvNeXt-Tiny | 10 Brinjal classes | 82.4% Domain Transfer | `eval_modern.py` |
| **Modern Swin** | `modern_head_swin.pth` | Swin-Transformer Tiny | 10 Brinjal classes | 81.9% Domain Transfer | `eval_modern.py` |
| **DINOv2 Linear Head** | `dinov2_head.pth` / `vitb14` | DINOv2 ViT-S / ViT-B | 10 Brinjal classes | 82.7% Domain Transfer | `eval_dinov2.py` |
| **Leaf Segmentation** | `sam_vit_b.pth` | Segment Anything Model (SAM ViT-B) | Binary Leaf Masking | Mean IoU > 94% on field leaves | `sam_leaf_crop.py` |
| **Interpretable RBPNN** | `resnet_rbpnn_final.pth` | ResNet Backbone + Prototype Layer | 10 Brinjal classes | Interpretable heatmaps + Q-scores | `eval_joint.py` |
| **Hierarchical L1** | `level1_coarse_resnet34.pth` | ResNet-34 Coarse Classifier | Healthy, Pest, Disease | 96.1% coarse screening | `hierarchical_batch_eval.py` |
| **Hierarchical L2 Pest** | `level2_pest_resnet34.pth` | ResNet-34 Fine Pest Classifier | Aphid, Jassid, Whitefly, Borer, Mealy | 88.4% pest species accuracy | `hierarchical_batch_eval.py` |
| **Hierarchical L2 Dis.** | `level2_disease_resnet34.pth` | ResNet-34 Fine Disease Classifier | Early Blight, Little Leaf, Phomopsis | 92.3% disease accuracy | `hierarchical_batch_eval.py` |

---

## 2. Detailed Technical Breakdown by Category

### Category 1: The Proposed Dual-Branch Model (`trained_models/02_dual_branch/`)
- **Files:** `dual_branch_head.pth`, `dual_branch_head_normal.pth`
- **Concept:** Standard CNN models trained on clean lab images collapse to ~50% when tested on field images due to domain shift (soil, lighting, background). The proposed architecture extracts global context using self-supervised DINOv2 (ViT-B/14) patch embeddings and combines them with localized prototype representations.
- **Weights Structure:** Dictionary containing:
  - `model_state_dict`: Projection layers and cross-attention fusion weights.
  - `classes`: Alphabetical list of class labels.
  - `variant`: `vitb14`.
  - `best_acc`: Recorded validation/test accuracy.
- **Evaluation Command:**
  ```bash
  python src/04_dual_branch/eval_dual_branch.py --data_dir test --variant vitb14 --head_path trained_models/02_dual_branch/dual_branch_head.pth
  ```

---

### Category 2: Cascade Specialist Disambiguators (`trained_models/04_cascade_disambiguators/`)
- **Files:** `disambig_pests.pth`, `disambig_fruit.pth`
- **Concept:** Error analysis revealed two dense confusion clusters:
  1. *Small Sap-Sucking Pests:* **Aphid vs. Jassid vs. Whitefly** (often cause similar chlorotic mottling).
  2. *Fruit Symptoms:* **Phomopsis Blight Fruit vs. Shoot and Fruit Borer**.
  When the primary classifier's margin between the top two softmax probabilities is `< 0.15`, the cascade routes the image to the specialist head trained exclusively on that confusable subgroup.
- **Evaluation Command:**
  ```bash
  python src/06_cascade_disambiguator/eval_cascade.py --data_dir test --variant vitb14 \
      --main_head_path trained_models/02_dual_branch/dual_branch_head.pth \
      --disambig_pests trained_models/04_cascade_disambiguators/disambig_pests.pth \
      --disambig_fruit trained_models/04_cascade_disambiguators/disambig_fruit.pth \
      --margin_threshold 0.15
  ```

---

### Category 3: Baseline CNN Checkpoints (`trained_models/01_baseline_cnns/`)
- **Files:** `resnet50.pth` (and checkpoint runs in `models/`)
- **Training Strategy:** 2-Stage Transfer Learning:
  - *Stage 1:* Backbone frozen, classifier head warmed up with Cosine Annealing learning rate.
  - *Stage 2:* Unfreeze deep residual blocks with reduced learning rate (`5e-5`) and label smoothing (`0.05`).
- **Inference Command:**
  ```bash
  python predict.py --image_path "test/Early_Blight/sample.jpg" --model_path "trained_models/01_baseline_cnns/resnet50.pth"
  ```
- **Batch Evaluation:**
  ```bash
  python predict_batch.py --data_dir "test" --model_path "trained_models/01_baseline_cnns/resnet50.pth" --output_dir "results/batch_prediction_reports/demo_run" --threshold 70.0
  ```

---

### Category 4: Hierarchical Classification Models (`trained_models/08_hierarchical/`)
- **Files:** `level1_coarse_resnet34.pth`, `level2_pest_resnet34.pth`, `level2_disease_resnet34.pth`
- **Concept:** Mirrors botanical triage:
  1. **Level 1 Coarse Classifier:** Distinguishes whether the plant is `Healthy`, infested with a `Pest`, or infected with a `Disease`.
  2. **Level 2 Specialist Branch:** Activates the corresponding species classifier based on the Level 1 decision.
- **Evaluation Command:**
  ```bash
  python src/03_hierarchical/hierarchical_batch_eval.py --data_dir test \
      --level1_path trained_models/08_hierarchical/level1_coarse_resnet34.pth \
      --level2_pest_path trained_models/08_hierarchical/level2_pest_resnet34.pth \
      --level2_disease_path trained_models/08_hierarchical/level2_disease_resnet34.pth \
      --threshold 70.0
  ```

---

### Category 5: Leaf Segmentation & Lesion Masking (`trained_models/06_segmentation_sam/`)
- **File:** `sam_vit_b.pth` (Segment Anything Model by Meta AI, 375 MB)
- **Concept:** Used by `sam_leaf_crop.py` to extract only the leaf blade and suppress soil, weeds, boots, and background clutter.
- **Execution Command:**
  ```bash
  python src/01_preprocessing/sam_leaf_crop.py --input_dir Brinjal --output_dir Brinjal_leaf_cropped --checkpoint trained_models/06_segmentation_sam/sam_vit_b.pth
  ```

---

### Category 6: Interpretable Prototype Network (`trained_models/07_rbpnn_prototypes/`)
- **Files:** `resnet_rbpnn_final.pth`, `backbone_phase_a.pth`, `prototypes_paper.pkl`, `prototypes.pkl`
- **Concept:** Case-based reasoning network. Instead of black-box classification, it compares input image feature patches against learned visual prototype vectors ("this part of the leaf looks like Prototype #4 of Early Blight").
- **Evaluation Command:**
  ```bash
  python src/08_rbpnn_prototypes/eval_joint.py --model_path trained_models/07_rbpnn_prototypes/resnet_rbpnn_final.pth
  ```
