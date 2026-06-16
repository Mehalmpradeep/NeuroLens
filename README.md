# NeuroLens – Real-Time Fatigue Detection System

NeuroLens is an AI-powered fatigue detection system designed to identify signs of fatigue in real time using computer vision and deep learning techniques.

## Overview

The system processes video input, extracts facial and behavioral features, and performs inference using a VideoMAE-based model to detect fatigue-related patterns. The application provides real-time monitoring and visualization through an interactive dashboard.

## Features

* Real-time fatigue detection from video streams
* Deep learning-based fatigue analysis using VideoMAE
* FastAPI backend for model inference
* Interactive dashboard for monitoring results
* Explainable AI (XAI) support for model interpretation
* Automated preprocessing and evaluation workflows

## Tech Stack

### Backend

* Python
* FastAPI

### Machine Learning

* VideoMAE
* PyTorch

### Dashboard

* Streamlit

### Data Processing

* OpenCV
* NumPy
* Pandas

## Project Structure

* `dashboard/` – Visualization and monitoring interface
* `backend_api.py` – API endpoints for inference
* `fatigue_preprocessor.py` – Data preprocessing pipeline
* `inference_newf.py` – Model inference workflow
* `evaluate.py` – Model evaluation
* `xai_analysis.py` – Explainability analysis

## My Contributions

- Assisted in dataset collection, preparation, and cleaning
- Participated in training and fine-tuning the VideoMAE model
- Tested system functionality across different usage scenarios
- Validated model outputs and system behavior
- Collaborated with team members during development and evaluation

## Future Improvements

* Cloud deployment
* Mobile support
* Improved model accuracy
* Multi-user monitoring dashboard
