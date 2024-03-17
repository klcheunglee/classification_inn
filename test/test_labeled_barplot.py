import unittest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from unittest.mock import patch


# Include the function to be tested
def labeled_barplot(data, feature, perc=False, n=None):
    total = len(data[feature])
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc:
            label = "{:.1f}%".format(100 * p.get_height() / total)
        else:
            label = p.get_height()

        x = p.get_x() + p.get_width() / 2
        y = p.get_height()

        ax.annotate(
            label,
            (x, y),
            ha='center',
            va='center',
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.show()


# Unit tests for the function
class TestLabeledBarplot(unittest.TestCase):
    @patch('matplotlib.pyplot.show')  # Mock plt.show() to not display the plot
    def test_with_default_parameters(self, mock_show):
        # Creating a sample dataframe
        test_data = pd.DataFrame({
            'feature': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'C', 'C']
        })
        # Testing function execution with default parameters
        labeled_barplot(test_data, 'feature')
        # If there's no error, the test passes

    @patch('matplotlib.pyplot.show')  # Mock plt.show() to not display the plot
    def test_with_percentage(self, mock_show):
        # Creating another sample dataframe
        test_data = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B', 'C', 'A', 'B', 'C', 'D', 'E']
        })
        # Testing function execution with percentage
        labeled_barplot(test_data, 'feature', perc=True)
        # If there's no error, the test passes

    @patch('matplotlib.pyplot.show')  # Mock plt.show() to not display the plot
    def test_with_n_parameter(self, mock_show):
        # Creating another sample dataframe
        test_data = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B', 'C', 'A', 'B', 'C', 'D', 'E']
        })
        # Testing function execution with n parameter
        labeled_barplot(test_data, 'feature', n=3)
        # If there's no error, the test passes


# Main block to execute the tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
