# Data loading and preprocessing module

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def Load_data(file_path=None):
    
    if file_path:
        df = pd.read_csv(file_path)
        X = df.drop(['id','diagnosis'], axis=1, errors='ignore')

        # Drop columns with all NaN values
        X = X.dropna(axis=1, how='all')
        
        # keep numeric columns only
        X = X.select_dtypes(include=[np.number])
        y = df['diagnosis'].map({'M':1, 'B':0})
        
    else:
        # If no file path is provided, load a sample dataset
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
    
    # Missing value imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)