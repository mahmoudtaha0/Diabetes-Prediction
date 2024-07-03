import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def main():
    # Display frame attributes
    df = pd.read_csv('diabetes.csv')

    # Extract features and labels
    features = np.array(df.iloc[:, :-1].values.tolist())
    labels = np.array(df.iloc[:, -1].values.tolist())

    # Split training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.4)

    # Train a support vector machine model
    model = LogisticRegression(max_iter=100000)
    model.fit(x_train, y_train)

    # Evaluate predicted labels
    predictions = model.predict(x_test)

    print(f"Results for model {type(model).__name__}")
    print(f"Accuracy: {accuracy_score(y_test, predictions)*100} %")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()