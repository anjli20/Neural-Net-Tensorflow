# Neural Networks with TensorFlow

Deep learning projects built with TensorFlow covering binary classification on tabular medical data
and natural language processing on wine review text. Two notebooks explore different neural network
architectures and techniques.

## Projects Overview

### Project 1 — Diabetes Prediction

Binary classification to predict whether a patient has diabetes based on medical diagnostic features.

- Type: Binary Classification (Tabular Data)
- Dataset: diabetes.csv
- Notebook: diabetes.ipynb

#### Pipeline

1. Exploratory data analysis — overlapping histograms for each feature split by class
2. Feature scaling using StandardScaler
3. Class imbalance handled using RandomOverSampler (268 diabetes vs 500 non-diabetes before oversampling)
4. Train / Validation / Test split: 60% / 20% / 20%
5. Dense neural network trained with Adam optimiser and Binary Cross-Entropy loss

#### Model Architecture

```
Input (8 features)
      |
Dense (16 units, ReLU)
      |
Dense (16 units, ReLU)
      |
Dense (1 unit, Sigmoid)
```

### Project 2 — Wine Review Sentiment Classification

Binary text classification to predict whether a wine review scores 90 points or above,
using two different NLP architectures.

- Type: Binary Text Classification (NLP)
- Dataset: wine-reviews.csv (columns used: description, points)
- Notebook: wine-review.ipynb
- Label: 1 if points >= 90, else 0

#### Pipeline

1. Load and clean dataset — drop rows with missing description or points
2. Create binary label from points threshold (>= 90)
3. Build TensorFlow data pipeline with batching and prefetching
4. Train and compare two model architectures

#### Model A — TF-Hub Pretrained Embedding + Dense

Uses a pretrained Google NNLM 50-dimensional word embedding from TensorFlow Hub.

```
Input (raw text)
      |
TF-Hub NNLM Embedding (50-dim, trainable)
      |
Dense (16 units, ReLU)
      |
Dropout (0.4)
      |
Dense (16 units, ReLU)
      |
Dropout (0.4)
      |
Dense (1 unit, Sigmoid)
```

#### Model B — TextVectorization + Embedding + LSTM

Trains a vocabulary from scratch using TextVectorization (top 2000 tokens),
followed by a learned embedding and LSTM layer.

```
Input (raw text)
      |
TextVectorization (vocab size 2000)
      |
Embedding (32-dim, mask_zero=True)
      |
LSTM (32 units)
      |
Dense (32 units, ReLU)
      |
Dropout (0.4)
      |
Dense (1 unit, Sigmoid)
```

## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| TensorFlow Hub | Pretrained NNLM text embedding |
| NumPy / pandas | Data loading and manipulation |
| matplotlib | Training curves and histograms |
| scikit-learn | Preprocessing and train/test split |
| imbalanced-learn | RandomOverSampler for class imbalance |
