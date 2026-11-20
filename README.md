# deteksi-gambar-ai
AI-generated vs. real image classification web app powered by fine-tuned EfficientNet-B0 (93.33% test accuracy), built with PyTorch and Streamlit.
# 🔍 AI-Generated vs. Real Image Detector

Web application for classifying whether an uploaded image is **AI-generated** or a **real photograph**, powered by a fine-tuned **EfficientNet-B0** model achieving **93.33% test accuracy** and **0.9333 Macro F1-Score**.

## Model Benchmark Summary
Trained and evaluated on a balanced dataset of **2,997 images** (`Train: 2,100` | `Validation: 447` | `Test: 450`):

| Model | Val Accuracy | Test Accuracy | Test Loss | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **EfficientNet-B0 (Fine-Tuned)** | **96.64%** | **93.33%** | **0.1764** | **0.9333** |
| EfficientNet-B0 (Baseline) | 96.20% | 87.33% | 0.3460 | 0.8732 |
| ViT-B/16 (Fine-Tuned) | 91.95% | 87.33% | 0.2937 | 0.8733 |
| VGG16 (Pretrained) | 89.26% | 87.33% | 0.3328 | 0.8730 |
| ViT-B/16 (Baseline) | 91.72% | 86.00% | 0.3221 | 0.8599 |
| Custom CNN (Scratch) | 88.59% | 82.22% | 0.4146 | 0.8217 |

## Tech Stack
- **Framework:** PyTorch & Torchvision
- **Deployment:** Streamlit Community Cloud
- **Selected Model:** EfficientNet-B0 (`best_effnet_ft.pth`)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
