import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from statsmodels.api import Logit
import matplotlib.pyplot as plt
import seaborn as sns


def confusion_matrix_statsmodels(model, predictors, target, threshold=0.5):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """
    y_pred = model.predict(predictors) > threshold
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


class TestConfusionMatrixStatsmodels(unittest.TestCase):
    def setUp(self):
        # Generate a binary classification dataset
        X, y = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=42)

        # Convert to DataFrame for compatibility with statsmodels
        predictors = pd.DataFrame(X, columns=[f'var{i}' for i in range(1, 21)])
        target = pd.Series(y)

        # Split the dataset for training and testing
        X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.25, random_state=42)

        # Fit a logistic regression model using statsmodels
        self.model = Logit(y_train, X_train).fit(disp=0)  # disp=0 suppresses fitting output
        self.X_test = X_test
        self.y_test = y_test

    def test_confusion_matrix_statsmodels(self):
        # Test the confusion_matrix_statsmodels function
        try:
            confusion_matrix_statsmodels(self.model, self.X_test, self.y_test)
            plt.close('all')  # Close the plot to avoid interference with other tests
        except Exception as e:
            self.fail(f"Execution of confusion_matrix_statsmodels raised an unexpected exception: {e}")


if __name__ == '__main__':
    unittest.main()
