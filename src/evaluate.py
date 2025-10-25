# src/evaluate.py
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def Evaluate_model(model, X_test, y_test, threshold=0.5):
    y_probs = model.predict_proba(X_test)[:,1]
    y_pred = (y_probs >= threshold).astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    
    # Classification Report
    report = classification_report(y_test, y_pred)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)
    
    # ROC Curve
    roc_auc = roc_auc_score(y_test, y_probs)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("outputs/roc_curve.png")
    plt.close()
    
    print("\n Evaluation complete! Check outputs folder. \n")