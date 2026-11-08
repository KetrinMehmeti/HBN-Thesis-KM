# EEG-Based Prediction of Dimensional Psychopathology

This project investigates how EEG representation strategies and spatial sampling influence the prediction of dimensional psychopathology in children and adolescents.

Using the Healthy Brain Network EEG (HBN-EEG) dataset, the study benchmarks machine learning and deep learning approaches for predicting four continuous psychopathological dimensions: Attention, Internalizing, Externalizing, and the P-Factor.

A particular focus is placed on evaluating whether predictive performance can be preserved when moving from high-density 128-channel EEG recordings to wearable-compatible 4-channel configurations.

## Methodology

The research explores four predictive approaches:

- **Hand-Crafted Features:** Extraction of spectral, temporal, and complexity-based EEG features followed by classical machine learning.
- **Pretrained EEG Representations:** Evaluation of embeddings extracted from pretrained EEG foundation models.
- **Model Fine-Tuning:** Adaptation of pretrained EEG models to downstream regression tasks.
- **End-to-End Deep Learning:** Training neural networks directly on preprocessed EEG recordings.

The spatial sampling analysis further examines the effects of electrode reduction and montage selection on predictive performance.

Repository Structure

- 01_preprocessing/ — EEG preprocessing scripts, including ICA.

- 02_classical_ml/ — Scripts for feature-based analysis and classical machine learning.

- 03_neural_models/ — Scripts for pretrained EEG representations and neural modeling.

## Tools

Python · scikit-learn · PyTorch · MNE-Python · AWS 

## Repository Contents

This repository is intended to showcase selected implementations of the data processing, feature extraction, predictive modeling, and evaluation methods developed during the research.

## Author

**Ketrin Mehmeti**
MSc Information Bioengineering | Politecnico di Milano
