# Utility functions for model evaluation and plotting

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def Plot_precision_recall_curve(y_test, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall vs Threshold")
    plt.legend()
    plt.show()