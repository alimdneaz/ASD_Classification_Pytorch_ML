from sklearn.metrics import accuracy_score


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.sign(outputs)
        accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
        print(f"Test Accuracy: {accuracy:.2f}")
