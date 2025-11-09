# 🩺 Cardiovascular Disease Diagnosis using 4D Cine MRI via Deep Learning


This project presents a Deep Learning approach for the automated diagnosis of
Cardiovascular Diseases (CVDs) using 4D Cardiac Magnetic Resonance (CMR) imaging. The
primary goal was to build a robust automated system capable of accurately segmenting heart
structures and classifying patients into disease categories.

This project also contains a web-based application developed with Streamlit,
allowing users to upload NIFTI files (CMR scans) for segmentation, feature analysis, and
automated diagnosis.

## 🎯 The Challenge

Cardiovascular diseases are a leading cause of mortality worldwide. While CMR imaging is a
powerful, non-invasive tool for assessing cardiac structure and function, its manual analysis
faces several challenges:

- Time-consuming & complex manual segmentation
- Inter-operator variability and susceptibility to human error
- Hand-crafted features may miss subtle patterns that deep learning can detect

## 💡 The Solution: A Hybrid Multi-stage Pipeline

The system combines state-of-the-art deep learning with domain knowledge in three stages:

1. Advanced segmentation using a Fully Convolutional Transformer (FCT) to segment LV, RV and
   myocardium in End-Diastolic (ED) and End-Systolic (ES) phases.
2. Feature extraction that produces (a) deep features via a Variational Autoencoder (VAE)
   and (b) 20 manual clinical features (ventricular volumes, EF, myocardial wall thickness,
   etc.), normalized with RobustScaler.
3. Classification via an ensemble (Voting Classifier) that aggregates MLP, Random Forest and
   SVM predictions on the combined feature vector (32 deep + 20 manual features).

### Dataset

The pipeline was trained and evaluated using the ACDC challenge dataset (150 patients,
5 classes: NOR, HCM, DCM, MINF, RV).

### Key results

- The FCT segmentation model achieves high Dice scores (example: ~93% for LV in ED).
- The hybrid classification pipeline reached ~92.0% accuracy on the test dataset.

## 💻 Streamlit Web Application

A Streamlit app enables users to:

- Upload NIFTI CMR scans
- View automated FCT segmentation results
- Inspect extracted manual features and compare to normal ranges
- Receive an automated diagnosis (NOR, HCM, DCM, MINF, RV)

Run the app (after installing dependencies):


## Pretrained models & results

Pretrained weights are available in `models/` (e.g. `UNet.pt`, `acdc_myunet_weights_224x224_92.84.h5`, `fct.model`, `vae-1.pth`).

The models are available 
[here](https://drive.google.com/drive/folders/19ZuWAJ0J3EvW7ZAJi49ZiUBShtBsEraE?usp=sharing)


