# TIH On-Site Demonstration & Repository Handover Guide
### Prepared for TIH Mentors & Evaluators (Presentation Dates: 27th & 28th)

**Project:** Leaf-Lesion Guided Domain-Generalized Vision Model with Explainability & VLM for Brinjal Disease Classification  
**Researcher:** Sakshi Patil  
**Fellowship:** CHANAKYA Graduate Internship / PoCRA Initiative  
**Venue:** TIH Lab / VDI Environment  

---

## 1. Quick Demonstration Workflow (10-Minute Walkthrough)

When presenting to evaluators on the TIH machine or VDI, execute the following 4-step live demonstration:

### Step 1: Single Image Field Inference with Safety Rejection
Demonstrate that the model predicts disease category and alerts if an image is ambiguous:
```bash
python predict.py --image_path "test/Early_Blight/sample.jpg" --model_path "resnet50.pth"
```
*Evaluator Talking Point:* Explain that our pipeline outputs both top-1 class and calibrated confidence. If confidence is below threshold (e.g., 70%), the system flags it as "Low Confidence" rather than giving a false diagnosis to a farmer.

---

### Step 2: Batch Analysis with Automated Excel & Grad-CAM++ Report
Show real-time batch processing across an entire folder with analytical spreadsheets:
```bash
python predict_batch.py --data_dir "test" --model_path "resnet50.pth" --threshold 70.0
```
*What to show evaluators:*
1. Open the generated `predictions_report.xlsx` inside `results/batch_prediction_reports/`:
   - Point out the per-class accuracy sheet, raw top-1 vs. accepted high-confidence accuracy, and low-confidence rate.
2. Open the 2x2 prediction image grid:
   - Shows original image, Grad-CAM++ attention heatmap, prediction breakdown, and top-5 probability distribution.

---

### Step 3: The Best Proposed Architecture (Dual-Branch 84.43% Cross-Dataset Accuracy)
Demonstrate the research contribution that solves the domain shift problem:
```bash
python src/04_dual_branch/eval_dual_branch.py --data_dir test --variant vitb14 --head_path trained_models/02_dual_branch/dual_branch_head.pth
```
*Evaluator Talking Point:* Explain that standard ResNet models drop to ~50% on unseen field images (`test/`). Our Dual-Branch model bridges this gap, achieving **84.43%** by fusing self-supervised DINOv2 global context with local prototype features.

---

### Step 4: Cascade Specialist Disambiguation
Demonstrate how hard-to-distinguish insect pests (Aphid vs. Jassid vs. Whitefly) are resolved:
```bash
python src/06_cascade_disambiguator/eval_cascade.py --data_dir test --variant vitb14 \
    --main_head_path trained_models/02_dual_branch/dual_branch_head.pth \
    --disambig_pests trained_models/04_cascade_disambiguators/disambig_pests.pth \
    --disambig_fruit trained_models/04_cascade_disambiguators/disambig_fruit.pth \
    --margin_threshold 0.15
```

---

## 2. Research Paper & Conference Submission Details

- **Research Working Title:**  
  *Leaf-Lesion Guided Domain-Generalized Vision Model with Explainability & VLM for Brinjal Disease Classification*
- **Primary Methodological Innovation:**  
  Overcoming severe lab-to-field domain degradation in agricultural computer vision using:
  1. DINOv2 self-supervised ViT representations + CNN prototype fusion.
  2. Segment Anything Model (SAM) promptable leaf isolation.
  3. Narrow-margin cascade specialist routing for confusing pest morphology.
  4. 4-bit quantized Vision-Language Model (Gemma 3 4B-it) for symptom text generation.
- **Target Conferences / Publication Venues:**
  1. *IEEE International Conference on Image Processing (ICIP)*
  2. *Computers and Electronics in Agriculture (Elsevier)*
  3. *IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPR - AgriVision)*
- **Documentation File:**  
  Complete scientific methodology, problem definition, experimental matrix, and citations are located in:  
  [`docs/TIH_Methodology_Submission_Sakshi_Patil_08_July_2026.docx`](file:///c:/Users/saksh/OneDrive/Desktop/tih/pocra_image_dev-main/docs/TIH_Methodology_Submission_Sakshi_Patil_08_July_2026.docx)

---

## 3. Vision-Language Model (VLM) Deliverable

- **Architecture:** Local Google **Gemma 3 4B-it** (4-bit quantized with `bitsandbytes` for GPU memory efficiency).
- **Domain Specialization:** Guided by an agronomist expert prompt requiring affected plant part, color transformation, lesion geometry, distribution pattern, and severity estimation.
- **Artifacts Handover:**
  - Script: `vlm/main.py`
  - Generated Output: `vlm/brinjal_captions.jsonl` (contains multi-paragraph, expert-verified descriptions for field images).
  - Guide: `vlm/VLM_DOCUMENTATION.md`

---

## 4. How to Edit & Enhance Source Code for Future Work

| Task for Next Developer | File to Edit | Key Function / Variable |
|---|---|---|
| **Add a new disease class** | `resnet50_class_to_idx.json` & `config.py` | Add key-value pair; update `NUM_CLASSES` |
| **Change CNN Backbone** | `config.py` | Change `MODEL_NAME = 'efficientnet_b0'` or `'resnet50'` |
| **Adjust Rejection Threshold** | `predict_batch.py` | Pass `--threshold <val>` (e.g. 80.0) |
| **Retrain Dual-Branch Head** | `src/04_dual_branch/train_dual_branch.py` | `--max_epochs 50 --lr 1e-4` |
| **Modify Cascade Confusable Groups** | `src/06_cascade_disambiguator/eval_cascade.py` | Edit `PEST_GROUP` or `FRUIT_GROUP` lists |
| **Tune VLM Expert Prompt** | `vlm/main.py` | Edit `SYSTEM_PROMPT` string |

---

## 5. Summary of Deliverables Checklist

- [x] Working deliverables demonstrated (Single, Batch, XAI, Dual-Branch, Cascade).
- [x] All 14 model weights cataloged in `trained_models/` with `MODELS_CATALOG.md`.
- [x] Repository structured into clean modular folders (`src/`, `trained_models/`, `results/`, `docs/`, `vlm/`).
- [x] All source code commented with usage examples.
- [x] Datasets documented (`processed_dataset/`, `test/`, `Brinjal_leaf_cropped/`).
- [x] Conference methodology document placed in `docs/`.
- [x] VLM symptom captioning pipeline and `.jsonl` included in `vlm/`.
- [x] 1-to-2 page operational guide complete (`TIH_HANDOVER_GUIDE.md`).
