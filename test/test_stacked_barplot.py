import unittest
import pandas as pd
import matplotlib.pyplot as plt


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


class TestStackedBarplot(unittest.TestCase):
    def setUp(self):
        # Set up mock data for testing
        self.test_data = pd.DataFrame({
            'predictor': ['Cat1', 'Cat2', 'Cat1', 'Cat3', 'Cat2', 'Cat1'],
            'target': [1, 0, 1, 1, 0, 0]
        })

    def test_stacked_barplot_execution(self):
        # Test the stacked_barplot function to ensure it executes without error
        try:
            stacked_barplot(self.test_data, 'predictor', 'target')
        except Exception as e:
            self.fail(f"Execution of stacked_barplot raised an unexpected exception: {e}")


if __name__ == '__main__':
    unittest.main()
