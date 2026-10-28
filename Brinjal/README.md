# PoCRA-TIH: Brinjal Disease & Pest Classification System
### Robust Domain-Generalized Vision Framework with Explainability & VLM

**Project Title:** Development of an Image-Based Disease and Pest Detection System for Crops (Brinjal / Eggplant)  
**Program:** TIH CHANAKYA Fellowship / PoCRA (Project on Climate Resilient Agriculture)  
**Student Researcher:** Sakshi Patil  
**Collaborator (Chickpea Work):** Devanshi  
**Primary Focus:** Brinjal (*Solanum melongena*) leaf disease and insect pest identification under real-world domain shifts, explainable AI (XAI), and symptom captioning using Vision-Language Models (VLM).

---

## 1. Executive Summary & Handover Overview

This repository houses the complete engineering deliverables and research codebase developed for automated detection, fine-grained classification, open-set rejection, and multimodal visual-linguistic analysis of **Brinjal** foliar diseases and insect pests.

### Key Research Contributions & Handover Deliverables:
1. **Clean Known-Class & OOD Pipeline:** Standardized 10-class taxonomy (pests, foliar fungal/bacterial diseases, healthy leaves, and unknown outlier rejection).
2. **Leaf-Lesion Guided Preprocessing:** Background isolation using Segment Anything Model (SAM ViT-B) to prevent models from learning field soil/sky background biases.
3. **Dual-Branch Architecture (DINOv2 + Prototype Head):** Solves severe domain transfer drop (from ~50% baseline up to **84.43%** cross-dataset accuracy) combining self-supervised ViT representations with localized texture cues.
4. **Specialist Cascade Disambiguators:** Specialized sub-classifiers resolving high-confusion clusters (e.g., Aphid vs. Jassid vs. Whitefly and Phomopsis Fruit Blight vs. Fruit Borer).
5. **Hierarchical 2-Level Decision Tree:** Level 1 coarse screening (Healthy vs. Pest vs. Disease) and Level 2 fine-grained species determination.
6. **Interpretability & XAI Suite:** Grad-CAM, Grad-CAM++, LIME, and Occlusion Sensitivity spatial heatmaps.
7. **Vision-Language Model (VLM) Symptom Captioning:** Agronomic prose captioning pipeline using **Gemma 3 4B-it** (4-bit quantized) with expert system prompts (`vlm/`).

---

## 2. Supported Disease & Pest Classes

The model is trained to recognize 10 distinct classes mapped in [`resnet50_class_to_idx.json`](file:///c:/Users/saksh/OneDrive/Desktop/tih/pocra_image_dev-main/resnet50_class_to_idx.json):

| Index | Class Name | Category | Symptoms & Visual Characteristics |
|---|---|---|---|
| **0** | **Aphid** | Insect Pest | Clusters of tiny green/black insects on lower leaf surface, honey-dew sheen, leaf curling. |
| **1** | **Early_Blight** | Fungal Disease | Concentric target-board brown spots (*Alternaria solani*), yellow chlorotic halos. |
| **2** | **Healthy** | Normal Foliage | Clean green canopy, uniform leaf venation, devoid of lesions or pests. |
| **3** | **Jassid** | Insect Pest | Hopper burn, leaf margin yellowing, curling downward, nymph feeding marks. |
| **4** | **Little_Leaf** | Phytoplasma | Severe leaf reduction, bushy stunted rosette appearance, loss of leaf lamina. |
| **5** | **Mealybugs** | Insect Pest | White cottony waxy clusters along veins, shoot junctions, and leaf undersides. |
| **6** | **Phomopsis_Blight_Fruit** | Fungal Disease | Dark circular sunken spots on fruit and leaves, pycnidia formation. |
| **7** | **Shoot and fruit borer** | Insect Pest | Larval bore holes, wilting terminal shoots, excreta near bore entry points. |
| **8** | **Unknown** | Out-of-Distribution | Non-crop imagery, weed leaves, severely damaged unidentifiable artifacts. |
| **9** | **Whitefly** | Insect Pest | Minute powdery-white insects, chlorotic mottling, sooty mold progression. |

---

## 3. Repository Directory Structure

```
pocra_image_dev-main/
│
├── README.md                              # This master document
├── requirements.txt                       # Python dependencies (PyTorch, Torchvision, etc.)
├── resnet50_class_to_idx.json             # 10-class dictionary mapping
│
├── [Root Quickstart Scripts]
│   ├── main.py                            # End-to-end pipeline launcher
│   ├── preprocessing.py                   # Data cleaning, augmentation & train/val/test split
│   ├── train.py                           # Baseline CNN model trainer (ResNet, EfficientNet, etc.)
│   ├── evaluate.py                        # Model evaluation & metrics calculator
│   ├── predict.py                         # Single-image inference CLI
│   ├── predict_batch.py                   # Batch prediction with Grad-CAM++ & Excel analytics
│   └── xai.py                             # Explainability suite (Grad-CAM & saliency)
│
├── src/                                   # Organized Source Code Modules
│   ├── 01_preprocessing/                  # Dataset preparation, SAM segmentation, domain splitting
│   ├── 02_baselines/                      # Standard CNN implementations & cross-validation
│   ├── 03_hierarchical/                   # 2-Level hierarchical classifier scripts
│   ├── 04_dual_branch/                    # Proposed Dual-Branch (DINOv2 + Prototype) model
│   ├── 05_domain_invariant/               # Domain-invariant feature alignment
│   ├── 06_cascade_disambiguator/          # Specialist cascade heads for confusable groups
│   ├── 07_unified_model/                  # Unified multi-task head
│   ├── 08_rbpnn_prototypes/               # Prototype networks (RBPNN / ProtoPNet)
│   ├── 09_foundation_backbones/           # DINOv2, ConvNeXt, Swin Transformer backbones
│   └── 10_explainability_xai/             # Grad-CAM, LIME, Occlusion Sensitivity map scripts
│
├── trained_models/                        # Model Checkpoints Catalog
│   ├── MODELS_CATALOG.md                  # Comprehensive technical catalog of all models
│   ├── 01_baseline_cnns/                  # resnet50.pth, resnet34, resnet18
│   ├── 02_dual_branch/                    # dual_branch_head.pth (Best model: 84.43% transfer)
│   ├── 03_domain_invariant/               # domain_invariant_head.pth
│   ├── 04_cascade_disambiguators/         # disambig_pests.pth, disambig_fruit.pth
│   ├── 05_foundation_backbones/           # dinov2_head.pth, modern_head_convnext.pth, etc.
│   ├── 06_segmentation_sam/               # sam_vit_b.pth (Segment Anything checkpoint)
│   ├── 07_rbpnn_prototypes/               # resnet_rbpnn_final.pth, prototypes_paper.pkl
│   └── 08_hierarchical/                   # level1_run1, level2_pest_run1, level2_disease_run1
│
├── results/                               # Experimental Evaluations & Reports
│   ├── RESULTS_OVERVIEW.md                # Summary tables and accuracy logs
│   ├── batch_prediction_reports/          # 17 historical batch runs with Excel & summary logs
│   ├── xai_visualizations/                # Grad-CAM overlays, Occlusion maps, and Attention maps
│   └── training_logs/                     # Training curves & checkpoint logs
│
├── docs/                                  # Project Documentation & Research Papers
│   ├── TIH_Methodology_Submission_Sakshi_Patil_08_July_2026.docx
│   ├── QUICK_IMPLEMENTATION_GUIDE.md
│   └── TIH_HANDOVER_GUIDE.md              # Demonstration checklist for TIH evaluators
│
└── vlm/                                   # Vision-Language Model Module
    ├── VLM_DOCUMENTATION.md               # Guide to Gemma 3 4B-it symptom captioning
    ├── main.py                            # 4-bit quantized VLM caption generation script
    └── brinjal_captions.jsonl             # Generated symptom captions database
```

---

## 4. Environment Setup & Installation

### Prerequisites
- Python 3.10+ (Tested with CUDA 12.1 / 12.2 on NVIDIA GPUs)
- Virtual Environment recommended:

```bash
# Create and activate environment
python -m venv .venv
.venv\Scripts\activate   # On Windows

# Install core requirements
pip install -r requirements.txt
```

---

## 5. Quickstart: How to Run the Pipeline

### 5.1 Single Image Inference (`predict.py`)
Predict the disease/pest for an individual field photo:
```bash
python predict.py --image_path "test/Healthy/sample.jpg" --model_path "resnet50.pth"
```

### 5.2 Comprehensive Batch Prediction & Excel Report (`predict_batch.py`)
Runs inference across a whole test directory, builds **Grad-CAM++ heatmaps**, flags low-confidence images, and exports a multi-sheet Excel spreadsheet (`predictions_report.xlsx`):
```bash
python predict_batch.py --data_dir "test" --model_path "resnet50.pth" --threshold 70.0
```

### 5.3 Explainable AI Heatmaps (`xai.py`)
Generates visual attention maps highlighting where the neural network is looking:
```bash
python xai.py --image_path "test/Early_Blight/sample.jpg" --model_path "resnet50.pth" --output_dir "results/xai_visualizations"
```

---

## 6. How to Run Advanced Research Architectures

### 6.1 Dual-Branch Architecture (DINOv2 + CNN) — *Top Performing Model*
The dual-branch model fuses self-supervised global ViT tokens from DINOv2 with local texture features, giving **84.43%** generalization on unseen datasets:
```bash
python src/04_dual_branch/eval_dual_branch.py --data_dir test --variant vitb14 --head_path trained_models/02_dual_branch/dual_branch_head.pth
```

### 6.2 Cascade Disambiguation System
Whenever top-1 prediction confidence margin is narrow between confusable pest/fruit classes, the cascade delegates to specialized sub-models:
```bash
python src/06_cascade_disambiguator/eval_cascade.py --data_dir test --variant vitb14 \
    --main_head_path trained_models/02_dual_branch/dual_branch_head.pth \
    --disambig_pests trained_models/04_cascade_disambiguators/disambig_pests.pth \
    --disambig_fruit trained_models/04_cascade_disambiguators/disambig_fruit.pth \
    --margin_threshold 0.15
```

### 6.3 Hierarchical 2-Level Classification
Evaluates coarse Level 1 (Healthy vs. Pest vs. Disease) before dispatching to Level 2 fine-grained species classifiers:
```bash
python src/03_hierarchical/hierarchical_batch_eval.py --data_dir test \
    --level1_path trained_models/08_hierarchical/level1_coarse_resnet34.pth \
    --level2_pest_path trained_models/08_hierarchical/level2_pest_resnet34.pth \
    --level2_disease_path trained_models/08_hierarchical/level2_disease_resnet34.pth \
    --threshold 70.0
```

### 6.4 SAM Leaf Background Segmentation (`sam_leaf_crop.py`)
Isolates green leaf lamina to remove background soil/tools/weed artifacts:
```bash
python src/01_preprocessing/sam_leaf_crop.py --input_dir Brinjal --output_dir Brinjal_leaf_cropped --checkpoint trained_models/06_segmentation_sam/sam_vit_b.pth
```

### 6.5 Vision-Language Model (VLM) Symptom Captioning (`vlm/`)
Generates natural language symptom descriptions using local quantized **Gemma 3 4B-it**:
```bash
cd vlm
python main.py
```

---

## 7. Datasets Used

1. **Eggplant Leaf Disease Dataset (Source Domain):**
   - Public benchmark (Kaggle/Mendeley) containing clean laboratory and greenhouse leaf disease images.
   - Used for primary training and held-out validation (`processed_dataset/`).
2. **Field Evaluation Dataset (`test/` and `testing/`):**
   - Unseen real-world field images gathered under variable sunlight, complex soil backgrounds, and different camera sensors.
   - Evaluates real-world domain transfer robustness.
3. **SAM Cropped Dataset (`Brinjal_leaf_cropped/`):**
   - Preprocessed using Segment Anything Model ViT-B promptable segmentation to test leaf-isolated vs. raw image domain robustness.
4. **Hierarchical Dataset Split (`hierarchical_dataset/`):**
   - Reorganized into `level1` (Coarse) and `level2` (Specialist Disease / Pest) folders.

---

## 8. TIH Handover Checklist for Mentors & Evaluators

| Requirement | Artifact / Path | Status |
|---|---|---|
| **1. Working Deliverables & Demo** | `predict.py`, `predict_batch.py`, `eval_dual_branch.py` | Ready for interactive CLI demo |
| **2. Trained Models Handover** | `trained_models/` (8 distinct categories + catalog) | All checkpoints cataloged |
| **3. Repository Structure** | Clean modular layout (`src/`, `trained_models/`, `results/`, `docs/`) | Standardized |
| **4. Commented Source Code** | All modules in `src/` contain docstrings & argument descriptions | Verified |
| **5. Technical Documentation** | `README.md`, `MODELS_CATALOG.md`, `TIH_HANDOVER_GUIDE.md` | Complete |
| **6. Testing Datasets** | `test/`, `testing/`, `domain_transfer_10pct/` | Accessible |
| **7. Research Paper & Conference Copy** | `docs/TIH_Methodology_Submission_Sakshi_Patil_08_July_2026.docx` | Included in `docs/` |
| **8. VLM Implementation** | `vlm/main.py` & `vlm/brinjal_captions.jsonl` | Ready with Gemma 3 pipeline |
| **9. Model Usage Instructions** | Section 5 & 6 above (1-2 pages concise steps) | Detailed |
| **10. TIH / VDI Demonstration** | Fully reproducible offline or on VDI | Tested |

---

## 9. Contact & Attribution
- **Researcher:** Sakshi Patil (TIH Chanakya Scholar)
- **Institution:** Technology Innovation Hub (TIH), NMICPS
- **Repository Maintenance:** For future extensions, see `src/` modules. Each module contains decoupled training and evaluation loops.