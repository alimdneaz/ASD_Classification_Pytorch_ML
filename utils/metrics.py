# utils/metrics.py
def accuracy(predictions, labels):
    return (predictions == labels).float().mean().item()


# main.py
from data.preprocess import load_asd_data
from models.svm import SVM
from training.train_model import train_model
from training.evaluate_model import evaluate_model

X_train, y_train = load_asd_data("data/processed/train.csv")
model = SVM(n_features=X_train.shape[1])
train_model(model, X_train, y_train)
evaluate_model(model, X_train, y_train)
