import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns

class MultivariateStats:
    def __init__(self, asset_of_interest):
        self.asset_of_interest = asset_of_interest
        self.data = pd.read_csv(f'excess_returns/{asset_of_interest}.csv')
        self.asset_names = self.data.columns.drop('date')
        self.asset_returns = self.data.drop(columns=['date'])
        # check if there are any abnormal values
        abnormal_values = self.asset_returns[(abs(self.asset_returns) > 1).any(axis=1)]
        if abnormal_values.empty:
            print("There are no abnormal values in the data")
        else:
            print("There are abnormal values in the data")
            print(abnormal_values)
    
    def plot_acf(self, lags=20):
        # plot acf for each column in the asset_returns
        for column in self.asset_names:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_acf(self.asset_returns[column], lags=lags, ax=ax)
            plt.show()

    def plot_pacf(self, lags=20):
        # plot pacf for each column in the asset_returns
        for column in self.asset_names:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_pacf(self.asset_returns[column], lags=lags, ax=ax)
            plt.show()

    def plot_return_distribution(self):
        # plot the distribution of the return for each asset
        for column in self.asset_names:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(self.asset_returns[column], ax=ax)
            plt.show()
    
    def plot_returns(self):
        # do NOT use date
        # plot all the returns on the same plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for column in self.asset_names:
            ax.plot(self.asset_returns[column], label=column)
        ax.legend()
        plt.show()
    
    def plot_cumulative_returns(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        for column in self.asset_names:
            cum_returns = (1 + self.asset_returns[column]).cumprod() - 1
            ax.plot(cum_returns, label=column)
        ax.legend()
        plt.show()

if __name__ == '__main__':
    asset_of_interest = ['300015', '600428', '000878', '600096']
    multivariate_stats = MultivariateStats(asset_of_interest)
    # multivariate_stats.plot_acf()
    # multivariate_stats.plot_pacf()
    # multivariate_stats.plot_return_distribution()
    # multivariate_stats.plot_returns()
    multivariate_stats.plot_cumulative_returns()
