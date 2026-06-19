# Driver-Drowsiness-Detection

An AI-based driver monitoring system developed using Computer Vision and Machine Learning techniques to detect driver drowsiness and improve road safety.

The project analyzes facial and eye movements in real time to determine whether the driver is drowsy or alert.

## Project Idea

The main objective of the project is to reduce road accidents caused by driver fatigue by building an intelligent monitoring system capable of detecting drowsiness automatically.

The system processes driver images, detects facial features and eyes, extracts important image features, and classifies the driver state using Machine Learning algorithms.

## Project Workflow

Input Image → Face Detection → Eye Detection → Image Enhancement → Feature Extraction → SVM Classification → Driver State Prediction

## Methodology

### Dataset Preparation

* Drowsy
* Non-Drowsy

The dataset was divided into two classes for training and testing.

### Face Detection

The project uses Haar Cascade Classifiers from OpenCV to detect the driver's face.

### Eye Detection

The eye region is extracted and resized for further processing.

### Image Enhancement (CLAHE)

CLAHE was applied to improve image visibility under different lighting conditions.

### Noise Reduction

Gaussian Blur was applied to reduce image noise before feature extraction.

### Edge Detection

Canny Edge Detection was used to extract important eye features.

### Machine Learning Classification

A Linear SVM classifier was trained to distinguish between:

* Drowsy Eyes
* Non-Drowsy Eyes

### Testing & Prediction

The trained model predicts the driver's state and displays the result in real time.

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-Learn
* Haar Cascade
* CLAHE
* Linear SVM

## Features

* Real-Time Driver Monitoring
* Face Detection
* Eye Detection
* Image Enhancement
* Edge Detection
* Drowsiness Classification
* Machine Learning Prediction

## Applications

* Smart Driver Safety Systems
* Automotive AI Systems
* Driver Monitoring Systems
* Accident Prevention Systems
* Intelligent Transportation Systems

## Author

Salama Ahmed Salama
