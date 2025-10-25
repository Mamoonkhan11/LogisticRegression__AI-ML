# Main script to run the logistic regression pipeline

from src.data_loader import Load_data
from src.model import Train_model, Save_model
from src.evaluate import Evaluate_model
from src.utils import Plot_precision_recall_curve

# 1. Load data
X_train, X_test, y_train, y_test = Load_data(file_path="Data/data.csv")

# 2. Train model
model = Train_model(X_train, y_train)
Save_model(model)

# 3. Evaluate model
Evaluate_model(model, X_test, y_test, threshold=0.5)

# 4. Optional: Precision-Recall curve for threshold tuning
y_probs = model.predict_proba(X_test)[:,1]
Plot_precision_recall_curve(y_test, y_probs)