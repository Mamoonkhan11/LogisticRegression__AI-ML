# Data loading and preprocessing module

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def Load_data(file_path=None):
    if file_path:
        df = pd.read_csv("../Data/data.csv")
        X = df.drop(['id','diagnosis'], axis=1) 
        y = df['diagnosis'].map({'M':1,'B':0}) 
    else:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
