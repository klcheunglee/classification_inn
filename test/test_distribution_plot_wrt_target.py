import unittest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def distribution_plot_wrt_target(data, predictor, target):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()

    # Calculate the statistical summary
    summary = data.groupby(target)[predictor].describe()
    print(summary)


# Unit tests for the functions
class TestVisualizationFunctions(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        self.test_data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'C'],
            'value': [10, 20, 15, 10, 25, 30, 5, 20],
            'target': [1, 0, 1, 0, 1, 0, 1, 0]
        })

    def test_distribution_plot_wrt_target(self):
        # Test distribution_plot_wrt_target function
        try:
            distribution_plot_wrt_target(self.test_data, 'value', 'target')
        except Exception as e:
            self.fail(f"distribution_plot_wrt_target raised an exception unexpectedly: {e}")


# Main block to execute unit tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
