# Hand Gesture Recognition Project Summary

## Project Overview
This project implements a hand gesture recognition system using background elimination techniques and a Convolutional Neural Network (CNN). The system can recognize three types of hand gestures: Palm, Fist, and Swing.

## Key Components

### 1. Hand Tracking and Segmentation (`PalmTracker.py`)
- Uses OpenCV for image processing
- Implements background subtraction techniques
- Segments hand region from video frames
- Creates dataset images for training

### 2. Image Preprocessing (`ResizeImages.py`)
- Uses PIL (Pillow) for image manipulation
- Resizes images to consistent dimensions for the CNN
- Prepares images for the training process

### 3. CNN Model Training (`ModelTrainer.ipynb`)
- Implements a CNN using TensorFlow/TFLearn
- Processes training images from the Dataset directory
- Trains the model to recognize three gesture classes
- Saves the trained model to the TrainedModel directory

### 4. Real-time Gesture Recognition (`ContinuousGesturePredictor.py`)
- Captures video from webcam
- Processes frames in real-time
- Segments hand region
- Uses the trained CNN model for prediction
- Displays recognition results

## Dataset Structure
- **Training Data**: 3,000 images (1,000 per gesture)
  - FistImages: 1,000 files
  - PalmImages: 1,000 files
  - SwingImages: 1,000 files
- **Testing Data**: 300 images (100 per gesture)
  - FistTest: 100 files
  - PalmTest: 100 files
  - SwingTest: 100 files

## System Requirements

### Recommended Environment
- Python 3.7
- TensorFlow 1.15.0
- TFLearn 0.3.2
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- imutils
- scikit-learn

### Compatibility Notes
- The project was originally designed for TensorFlow 1.x and TFLearn
- Running on newer Python versions (3.8+) requires code modifications
- TensorFlow 2.x compatibility requires enabling v1 compatibility mode

## Installation Instructions

### Option 1: Using Python 3.7 (Recommended)
```
# Create virtual environment
python -m venv venv_py37
# Activate virtual environment
venv_py37\Scripts\activate
# Install dependencies
pip install tensorflow==1.15.0 tflearn==0.3.2 numpy==1.19.5 opencv-python==4.5.5.64 imutils==0.5.4 scikit-learn==0.24.2 pillow==8.4.0
```

### Option 2: Using Python 3.12 with TensorFlow 2.x (Requires Code Modifications)
```
# Install dependencies
pip install tensorflow==2.16.1 ml-dtypes~=0.3.1 numpy>=1.22.0 opencv-python>=4.5.5.64 imutils>=0.5.4 scikit-learn>=1.0.2 pillow>=9.0.0
```

## System Workflow
1. Hand segmentation using background elimination
2. Image preprocessing and resizing
3. CNN model training with gesture images
4. Real-time gesture recognition

## Current Status
The project requires Python 3.7 and TensorFlow 1.15.0 for optimal compatibility. Running on newer Python versions requires code modifications to handle API changes in TensorFlow and related libraries.