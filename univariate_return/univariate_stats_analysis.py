import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
class UnivariateStatsAnalysis:
    ## conduct analysis on log return
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.data = pd.read_csv(f'asset_daily_return/{stock_code}.csv')

    def get_log_return(self):
        return self.data['log return']
    
    def acf(self, lags=20):
        series = self.get_log_return().dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(series, lags=lags, ax=ax, zero=True)  
        ax.set_title(f"ACF of log returns for {self.stock_code}", fontsize=14)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        plt.tight_layout()
        plt.show()
    
    def pacf(self, lags=20):
        series = self.get_log_return().dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_pacf(series, lags=lags, ax=ax, zero=True)
        ax.set_title(f"PACF of log returns for {self.stock_code}", fontsize=14)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Partial Autocorrelation")
        plt.tight_layout()
        plt.show()

    def absolute_return_acf(self, lags=20):
        series = self.get_log_return().dropna()
        absolute_return = series.abs()
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(absolute_return, lags=lags, ax=ax, zero=True)
        ax.set_title(f"ACF of absolute returns for {self.stock_code}", fontsize=14)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        plt.tight_layout()
        plt.show()
        
    def square_return_acf(self, lags=20):
        series = self.get_log_return().dropna()
        square_return = series ** 2
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(square_return, lags=lags, ax=ax, zero=True)
        ax.set_title(f"ACF of square returns for {self.stock_code}", fontsize=14)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        plt.tight_layout()
        plt.show()
    
    def distribution(self):
        series = self.get_log_return().dropna()
        # calculate the kurtosis and skewness
        kurtosis = series.kurtosis()
        skewness = series.skew()
        print(f"Kurtosis of {self.stock_code}: {kurtosis}")
        print(f"Skewness of {self.stock_code}: {skewness}")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(series, ax=ax)
        ax.set_title(f"Distribution of log returns for {self.stock_code}, kurtosis: {kurtosis}, skewness: {skewness}")
        ax.set_xlabel("Log return")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    
    def plot_series_seperately(self, features_of_interest = ['return', 'daily risk free rate', 'daily excess return', 'cumulative excess return', 'log return', 'volatility_GARCH', 'volatility_EWMA']):
        # plot all features in self.data
        for column in features_of_interest:
            fig, ax = plt.subplots(figsize=(8, 4))
            current_data = self.data[column]
            ax.plot(current_data)
            ax.set_title(f"{column} for {self.stock_code}")
            ax.set_xlabel("Date")
            ax.set_ylabel(column)
            plt.tight_layout()
            plt.show()
    

    def plot_series_together(self, features_of_interest = ['volatility_GARCH', 'volatility_EWMA', 'volatility_Kalman']):
        # plot all features in features_of_interest on the same plot
        fig, ax = plt.subplots(figsize=(8, 4))
        for column in features_of_interest:
            current_data = self.data[column]
            ax.plot(current_data)
        ax.set_title(f"All features for {self.stock_code}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        plt.tight_layout()
        plt.legend(features_of_interest)
        plt.show()
            

if __name__ == '__main__':
    stock_code = '600428'
    univariate_stats_analysis = UnivariateStatsAnalysis(stock_code)

    # univariate_stats_analysis.plot_series_seperately()
    univariate_stats_analysis.plot_series_together()
