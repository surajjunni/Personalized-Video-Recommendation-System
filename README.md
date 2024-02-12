# Personalized Video Recommendation System

## Overview

This repository contains code for a personalized video recommendation system. The system is designed to predict user engagement with videos and capture item-item similarities using a combination of binary cross-entropy loss for prediction and cosine similarity loss for capturing relationships between items.

## Files and Structure

- `preprocess_data.py`: Script for loading and preprocessing the data.
- `train_model.py`: Script for training the recommendation model.
- `evaluate_model.py`: Script for evaluating the trained model on test data.
- `video_recommendation_model.h5`: Trained model file.
- `train_data.csv` and `test_data.csv`: Preprocessed training and testing data.

## Instructions

1. **Preprocess Data**: Run `preprocess_data.py` to load and preprocess your data. This script will save the preprocessed data in `train_data.csv` and `test_data.csv`.

```bash
python preprocess_data.py


## Usage

### Train Model

Run `train_model.py` to train the recommendation model using the preprocessed data. The trained model will be saved as `video_recommendation_model.h5`.

```bash
python train_model.py

###Evaluate Model

Run `evaluate_model.py` to evaluate the trained model on the test data. This script calculates binary cross-entropy loss, cosine similarity loss, and binary accuracy.

```bash
python evaluate_model.py

###Customization

- Adjust hyperparameters, model architecture, or loss weights in the scripts based on your specific requirements.
- Ensure your data is in the correct format and adjust file paths accordingly.

##Dependencies

- Python 3.x
- TensorFlow 2.x
- NumPy
- pandas
- scikit-learn

##License

This project is licensed under the MIT License.