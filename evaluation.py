import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy_test)
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=[str(c) for c in le.classes_]
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=[str(c) for c in le.classes_]
    )
    plt.title("Confusion Matrix")
    plt.show()
