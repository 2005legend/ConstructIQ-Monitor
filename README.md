# SiteSpectra: Construction Reality Intelligence Pipeline

SiteSpectra is an end-to-end computer vision pipeline designed for modern construction monitoring. It combines deep learning-based object detection (YOLOv11) with classical computer vision frame-differencing to automate site safety compliance and quantify structural progress across time.

This project was built to simulate a lightweight construction monitoring pipeline, mirroring the reality-capture workflows used by industry leaders like Track3D.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange)](https://huggingface.co/sidaarth005/ConstructIQ-Monitor)

## 🏗️ Core Features
- **Site Entity & Safety Detection:** Detects workers, machinery, and PPE compliance in real-time. Identifies specific hazard classes like `NO-Hardhat` or `NO-Safety Vest`.
- **Structural Progress Tracking:** Quantifies visual site changes by computing the structural difference between timestamped before/after images.
- **Interactive Dashboard:** A Streamlit-based UI for real-time inference, batch image processing, and result exportation.

## ⚙️ Architecture & Pipeline Flow

The project is structured as a multi-stage vision pipeline rather than a single notebook demo. 

```mermaid
graph TD
    A[Raw Site Images] --> B{Streamlit Dashboard}
    
    subgraph Object Detection Module
    B -->|Batch Inference| C[YOLOv11 PyTorch Model]
    C --> D[PPE & Machinery Counts]
    end
    
    subgraph Progress Tracking Module
    B -->|Timestamped Image Pairs| E[OpenCV Preprocessing]
    E -->|Gaussian Blur & Grayscale| F[Absolute Diff & Thresholding]
    F --> G[Morphological Masking]
    G --> H[Change Contour Localization]
    end
    
    D --> I[Result Visualization & ZIP Export]
    H --> I
```

## 📸 Sample Outputs

*(Note: Replace these placeholders with the best results from the 17 images you downloaded!)*

### Object & Safety Detection
![Detection Sample 1](assets/detection_sample_1.png)
*Figure 1: YOLOv11 successfully identifying workers and PPE compliance.*

### Progress Comparison (Before / After)
![Progress Sample](assets/progress_sample_1.png)
*Figure 2: OpenCV pipeline highlighting new structural additions in red.*

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Pipeline
```bash
streamlit run app.py
```

### 3. Model Training & Validation
The model was fine-tuned on the Construction Site Safety Dataset (2,800+ images) for 20 epochs. It achieved an overall **mAP50 of 80.6%** on the validation set, with high accuracy for critical classes like Hardhats (88.9%) and Machinery (92.0%).

To run training on a custom dataset:
```bash
python train.py
```
*(A detailed Google Colab training guide is available in the project documentation).*

## 🔮 Future Work
- Integration with 3D BIM models for spatial progress mapping.
- Extend pipeline with an optional COLMAP module for multi-view 3D scene reconstruction.
- Implement instance segmentation for specific structural elements (e.g., scaffolding, rebar).
