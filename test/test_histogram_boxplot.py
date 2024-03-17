import unittest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Function to be tested
def histogram_boxplot(data, feature, figsize=(15, 10), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="pink")
    if bins:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, color="violet")
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, color="violet")
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")


# Unit test for the function
class TestHistogramBoxplot(unittest.TestCase):
    def test_execution(self):
        # Test data
        test_data = pd.DataFrame({
            'test_feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        # This should run without errors
        histogram_boxplot(test_data, 'test_feature')
        # If there's no error, the test passes
        print("Test passed: The function executed without errors.")


# Main block to execute unit test
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
