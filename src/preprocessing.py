import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data including handling categorical variables and missing values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: Processed features (X) and target (y)
    """
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Handle categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        le = LabelEncoder()
        for col in categorical_columns:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into train and test sets and scale the features.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: Scaled training and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test 