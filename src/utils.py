"""
Utility functions for diabetes risk prediction ML pipeline
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib


def load_and_preprocess_data(filepath):
    """
    Load CSV and perform initial preprocessing
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        X: Features dataframe
        y: Target series
        feature_names: List of feature names
    """
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names


def handle_missing_values(X_train, X_test, strategy='mean'):
    """
    Impute missing values using specified strategy
    
    Args:
        X_train: Training features
        X_test: Test features
        strategy: Imputation strategy ('mean', 'median', 'most_frequent')
        
    Returns:
        X_train_imputed: Imputed training features
        X_test_imputed: Imputed test features
        imputer: Fitted imputer object
    """
    imputer = SimpleImputer(strategy=strategy)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    return X_train_imputed, X_test_imputed, imputer


def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        scaler: Fitted scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def save_model_artifacts(model, scaler, imputer, feature_names, model_name):
    """
    Save model and preprocessing artifacts
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        imputer: Fitted imputer
        feature_names: List of feature names
        model_name: Name for the model files
    """
    joblib.dump(model, f'models/{model_name}_model.pkl')
    joblib.dump(scaler, f'models/{model_name}_scaler.pkl')
    joblib.dump(imputer, f'models/{model_name}_imputer.pkl')
    joblib.dump(feature_names, f'models/{model_name}_features.pkl')
    
    print(f"✓ Saved {model_name} artifacts to models/ directory")


def load_model_artifacts(model_name):
    """
    Load saved model and preprocessing artifacts
    
    Args:
        model_name: Name of the model files
        
    Returns:
        model: Loaded model
        scaler: Loaded scaler
        imputer: Loaded imputer
        feature_names: List of feature names
    """
    model = joblib.load(f'models/{model_name}_model.pkl')
    scaler = joblib.load(f'models/{model_name}_scaler.pkl')
    imputer = joblib.load(f'models/{model_name}_imputer.pkl')
    feature_names = joblib.load(f'models/{model_name}_features.pkl')
    
    return model, scaler, imputer, feature_names


def predict_single_patient(model, scaler, imputer, patient_data, feature_names):
    """
    Make prediction for a single patient
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        imputer: Fitted imputer
        patient_data: Dictionary with patient features
        feature_names: List of expected feature names
        
    Returns:
        prediction: 0 or 1
        probability: Probability of positive class
    """
    # Convert to dataframe
    df = pd.DataFrame([patient_data], columns=feature_names)
    
    # Preprocess
    df_imputed = imputer.transform(df)
    df_scaled = scaler.transform(df_imputed)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]
    
    return prediction, probability