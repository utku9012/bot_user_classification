import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

def create_model(input_shape):
    """
    Create the neural network model.
    
    Args:
        input_shape (tuple): Shape of input features
        
    Returns:
        tensorflow.keras.models.Sequential: Compiled model
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, validation_split=0.2, epochs=100, batch_size=32):
    """
    Train the model with early stopping.
    
    Args:
        model: Compiled model
        X_train: Training features
        y_train: Training labels
        validation_split: Proportion of validation set
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        
    Returns:
        tuple: Training history and trained model
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history, model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and generate performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: Predictions and performance metrics
    """
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return y_pred, metrics

def plot_training_history(history, save_path='results/training_history.png'):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_test, y_pred, save_path='results/confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, save_path='results/feature_importance.png'):
    """
    Plot feature importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    weights = model.layers[0].get_weights()[0]
    importance = np.mean(np.abs(weights), axis=1)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return feature_importance

def save_metrics(metrics, save_path='results/metrics.txt'):
    """
    Save model metrics to a text file.
    
    Args:
        metrics (dict): Dictionary containing model metrics
        save_path (str): Path to save the metrics file
    """
    with open(save_path, 'w') as f:
        f.write("Model Performance Metrics\n")
        f.write("=======================\n\n")
        
        f.write(f"Accuracy Score: {metrics['accuracy']:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'])
        
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix'])) 