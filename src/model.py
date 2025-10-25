# Model training module

from sklearn.linear_model import LogisticRegression
import pickle

def Train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def Save_model(model, filename="logistic_model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)