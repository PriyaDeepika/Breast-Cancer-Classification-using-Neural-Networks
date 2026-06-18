# Breast Cancer Classification using Neural Networks

## Overview

This project implements a Deep Learning model using TensorFlow/Keras to classify breast cancer tumors as **Malignant** or **Benign** using the Breast Cancer Wisconsin dataset.

The project demonstrates the complete Deep Learning workflow, including data preprocessing, feature scaling, neural network training, model evaluation, and prediction on unseen patient data.

---

## Dataset

The Breast Cancer Wisconsin dataset contains:

* **569 samples**
* **30 numerical features**
* **2 target classes**

### Target Labels

* **0 → Malignant**
* **1 → Benign**

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* TensorFlow / Keras
* Matplotlib

---

## Model Architecture

```text
Input Layer (30 Features)
        ↓
Dense Layer (20 Neurons, ReLU)
        ↓
Dense Layer (10 Neurons, ReLU)
        ↓
Output Layer (1 Neuron, Sigmoid)
```

### Compilation

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Metric: Accuracy

---

## Workflow

1. Load Breast Cancer dataset
2. Perform Train-Test Split
3. Apply StandardScaler for feature scaling
4. Build Neural Network using TensorFlow/Keras
5. Train the model using the training dataset
6. Evaluate performance on unseen test data
7. Create a predictive system for new patient records

---

## Results

### Test Performance

| Metric        | Value      |
| ------------- | ---------- |
| Test Accuracy | **97.37%** |
| Test Loss     | **0.0832** |

### Classification Report

| Class     | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| Malignant | 0.95      | 0.98   | 0.97     |
| Benign    | 0.99      | 0.97   | 0.98     |

Overall model accuracy on the test dataset: **97.37%**

---

## Predictive System

The trained model can classify new patient records by:

1. Accepting 30 diagnostic features.
2. Applying the same feature scaling used during training.
3. Generating a prediction using the trained neural network.
4. Returning either:

   * **Malignant**
   * **Benign**

Example output:

```python
[[3.2170722e-06]]
```

Since the value is less than **0.5**, the model predicts:

```text
Malignant
```

## Trained Model

The repository includes a pre-trained Keras model:

- `breast_cancer_model.keras`

## Loading the Model

```python
from tensorflow import keras
import joblib

# Load the trained neural network
model = keras.models.load_model("breast_cancer_model.keras")

# Load the scaler used during training
scaler = joblib.load("scaler.pkl")
```

---

## Key Concepts Learned

* Train-Test Split
* Feature Scaling using StandardScaler
* Neural Network Architecture
* ReLU Activation Function
* Sigmoid Activation Function
* Binary Crossentropy Loss
* Adam Optimizer
* Model Training with Keras
* Model Evaluation
* Building a Predictive System
