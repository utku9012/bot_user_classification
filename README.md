# Bot vs User Classification using Neural Networks

This project implements a neural network model to classify between bots and users based on various features. The model uses a deep learning approach with proper regularization techniques to prevent overfitting.

## Dataset

The project uses the `https://www.kaggle.com/datasets/juice0lover/users-vs-bots-classification` dataset which contains various features to distinguish between bot and human user behaviors. The dataset includes multiple behavioral and interaction patterns that help in identifying automated bot activities versus genuine user actions.

### Target Variable
- Binary classification (0: Human User, 1: Bot)
- The model aims to accurately identify automated bot activities while minimizing false positives for human users

### Features
The dataset includes various features such as:
- User interaction patterns
- Activity timestamps
- Behavioral metrics
- Interaction frequencies
- Session characteristics
- And other relevant features that help distinguish between bots and human users

## Features

- Neural Network implementation using TensorFlow/Keras
- Data preprocessing and feature engineering
- Model evaluation with multiple metrics
- Visualization of results
- Feature importance analysis

## Project Structure

```
├── data/                  # Data directory
│   └── bots_vs_users.csv  # Dataset
├── src/                   # Source code
│   ├── model.py          # Neural network model
│   ├── preprocessing.py  # Data preprocessing
│   └── utils.py          # Utility functions
├── notebooks/            # Jupyter notebooks
│   └── analysis.ipynb    # Data analysis notebook
├── results/              # Results and visualizations
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/utku9012/bot_user_classification.git
cd bot-user-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python src/model.py
```

2. For interactive analysis, open the Jupyter notebook:
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Model Architecture

The neural network consists of:
- Input layer
- 3 hidden layers (64 → 32 → 16 neurons)
- Output layer (1 neuron)
- Dropout layers for regularization
- Early stopping implementation

## Results

The model performance is evaluated using:
- Accuracy metrics
- Confusion matrix
- Feature importance analysis
- Training history visualization
    