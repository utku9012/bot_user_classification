import os
from preprocessing import load_data, preprocess_data, split_and_scale_data
from model import (
    create_model, train_model, evaluate_model,
    plot_training_history, plot_confusion_matrix,
    plot_feature_importance, save_metrics
)

def main():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data('data/bots_vs_users.csv')
    X, y = preprocess_data(df)
    
    # Split and scale data
    print("Splitting and scaling data...")
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X, y)
    
    # Create and train model
    print("Creating and training model...")
    model = create_model(input_shape=(X_train_scaled.shape[1],))
    history, model = train_model(model, X_train_scaled, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred, metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Print metrics
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Generate plots and save metrics
    print("Generating plots and saving metrics...")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    feature_importance = plot_feature_importance(model, X.columns)
    save_metrics(metrics)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10)[['Feature', 'Importance']].to_string(index=False))
    
    print("\nResults have been saved to the 'results' directory.")

if __name__ == "__main__":
    main() 