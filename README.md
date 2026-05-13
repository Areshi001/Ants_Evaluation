
# Drone Human Detection & Counting System

## Overview
This project builds a computer vision pipeline to detect humans and cars in drone/aerial images using YOLOv8n trained on the VisDrone dataset. The system includes human counting, FPS benchmarking, and ByteTrack-based object tracking as a bonus feature.

## Requirements
- Python 3.10+
- CUDA-capable GPU (T4 recommended)
- Kaggle account (for dataset download)
- Groq API key (optional for LLM summary)

## Installation

pip install ultralytics kaggle opencv-python groq

## Dataset
VisDrone Dataset (2019 DET track)
Download automatically via Kaggle API or manually from:
https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset

## Project Structure

├── visdrone_yolo/          # Preprocessed dataset (human + car only)
│   ├── images/train/       # Training images
│   ├── images/val/         # Validation images
│   ├── labels/train/       # YOLO format labels
│   └── labels/val/         # YOLO format labels
├── counted_outputs/        # Images with human count overlay
├── display_outputs/        # Images with human + car counts
├── runs/detect/train/      # Trained model and plots
├── runs/track/             # ByteTrack output videos
├── class_distribution.png  # EDA visualization
├── dataset_eda_summary.txt # Dataset analysis
└── visdrone.yaml           # YOLO dataset config

## Usage

1. Set Kaggle API credentials in the notebook
2. Run all cells in order
3. Training takes approximately 40 minutes on T4 GPU
4. Final outputs saved in display_outputs/ and track_output/

## Results

| Metric | Value |
|--------|-------|
| Overall mAP50 | 0.557 |
| Human mAP50 | 0.390 |
| Car mAP50 | 0.723 |
| Human Recall | 0.347 |
| Car Recall | 0.690 |
| Inference FPS | ~35 |

## Features
- Dataset EDA with class distribution and sample visualizations
- Class filtering (pedestrian, people, car) remapped to human(0) and car(1)
- YOLOv8n training (25 epochs, imgsz=640, batch=16)
- Human and car detection with colored bounding boxes
- Per-image human counting
- FPS benchmarking
- ByteTrack tracking with unique human ID counting
- LLM-generated project summary (Groq Llama 3.3)

## Limitations
- Human detection weak (mAP50 only 0.39)
- Low recall (0.347) - 65% of humans missed
- Small objects below 30 pixels rarely detected
- Occlusion causes missed detections

## Demo Video
A 3-5 minute demonstration video covering:
- Dataset EDA
- Training loss curves
- Sample detection outputs
- ByteTrack tracking visualization
- Metrics and limitations discussion

## License
Educational use only for Antlings Internship Assessment.

## Author
Submitted for Antlings Internship Program - Technical Assessment AI/ML
