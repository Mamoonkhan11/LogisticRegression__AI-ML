# Main script to run the logistic regression pipeline

from src.data_loader import load_data
from src.model import train_model, save_model
from src.evaluate import evaluate_model
from src.utils import plot_precision_recall_curve

# 1. Load data
X_train, X_test, y_train, y_test = load_data(file_path="Data/data.csv")

# 2. Train model
model = train_model(X_train, y_train)
save_model(model)

# 3. Evaluate model
evaluate_model(model, X_test, y_test, threshold=0.5)

# 4. Optional: Precision-Recall curve for threshold tuning
y_probs = model.predict_proba(X_test)[:,1]
plot_precision_recall_curve(y_test, y_probs)