import akshare as ak
import pandas as pd
import warnings
import numpy as np
import os
from arch import arch_model
from pykalman import KalmanFilter

class Univarite_Volatility:
    def __init__(self, df):
        self.df = df
    def estimate_volatility_GARCH(self, p=1, o=0, q=1):
        log_return = (self.df['log return'].dropna()*100).astype(float)
        am   = arch_model(log_return, mean="Constant", vol="GARCH",
                        p=p, o=o, q=q)
        res  = am.fit(update_freq=0, disp="off")
        sigma = res.conditional_volatility       
        return sigma / 100.0
    
    def estimate_volatility_EWMA(self, lam=0.94):
        alpha = 1 - lam
        r = self.df['log return'].dropna() * 100.0
        # compute EWMA of squared returns
        var = (r**2).ewm(alpha=alpha, adjust=False).mean()
        # take square root to get volatility
        return np.sqrt(var) / 100.0
    

    def estimate_volatility_Kalman(self,
                                    initial_vol=None,
                                    trans_cov=1e-6,
                                    obs_cov=1e-2):
        """
        Estimate volatility using a Kalman Filter on squared log returns.

        Parameters:
        - df: DataFrame with 'log return' column
        - initial_vol: initial estimate of variance (defaults to mean of squared returns)
        - trans_cov: process (transition) covariance
        - obs_cov: observation covariance

        Returns:
        - Series of estimated volatility (standard deviation)
        """
        # Prepare data
        r = self.df['log return'].dropna() * 100.0
        y = (r ** 2).values  # squared returns as observations

        # Set initial state
        if initial_vol is None:
            initial_state_mean = y.mean()
        else:
            initial_state_mean = initial_vol

        # Build Kalman filter: state = variance_t, observation = variance_t + noise
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=initial_state_mean,
            initial_state_covariance=1.0,
            transition_covariance=trans_cov,
            observation_covariance=obs_cov,
        )

        # Run filtering
        state_means, state_covs = kf.filter(y)

        # Convert variance estimates to volatility
        sigma = np.sqrt(state_means.flatten())
        return sigma / 100.0

class UnivariateReturn:
    def __init__(self, stock_code, start_date='20000101', end_date='22220523', period='daily', adjust='qfq'):
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.adjust = adjust

    def get_risk_free_rate(self):
        ##将隔夜拆借利率视为无风险利率
        rate_interbank_df = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="隔夜")
        rate_interbank_df['daily risk free rate'] = rate_interbank_df['利率'] / 100.0 / 365
        # rename '报告日' to '日期'
        rate_interbank_df.rename(columns={'报告日': '日期'}, inplace=True)
        rate_interbank_df = rate_interbank_df[['日期', 'daily risk free rate']]
        return rate_interbank_df

        
    def get_stock_data(self):
        ## 默认特征
        ## 日期    股票代码    开盘    收盘    最高    最低     成交量           成交额     振幅    涨跌幅   涨跌额    换手率
        result = ak.stock_zh_a_hist(symbol=self.stock_code, start_date=self.start_date, end_date=self.end_date, period=self.period, adjust=self.adjust)
        
        
        ## check if there's 0 in 收盘
        if result['收盘'].isin([0]).any():
            print("There are 0 in the the close column")
            ## ffill 0 with the previous value
            result.replace(0, method='ffill', inplace=True) 
        else:
            print("There are no 0 in the the close column")
        

        ## generate the return column
        if '涨跌幅' in result.columns:
            result['return'] = result['涨跌幅'] / 100.0
        else:
            result['return'] = result['收盘'].pct_change()
        return result
    
    def generate_excess_return(self):
        risk_free_rate = self.get_risk_free_rate()
        stock_data = self.get_stock_data()
        risk_free_rate['日期'] = pd.to_datetime(risk_free_rate['日期'])
        stock_data['日期'] = pd.to_datetime(stock_data['日期'])
        ## merge the risk free rate and the stock data
        merged_data = pd.merge(stock_data, risk_free_rate, how='left')
        merged_data['daily excess return'] = merged_data['return'] - merged_data['daily risk free rate']
        merged_data['cumulative excess return'] = (1 + merged_data['daily excess return']).cumprod() - 1
        merged_data['log return'] = np.log(1 + merged_data['daily excess return'])
        # merged_data['cumulative log return'] = merged_data['log return'].cumsum() 
        # note that the cumulative log daily excess return is an approximation of the cumulative excess return


        ## check if there are nan values ##
        if merged_data.isnull().any().any():
            print("There are nan values in the data")
            merged_data = merged_data.dropna()
        else:
            print("There are no nan values in the data")
        ## check if there are place holder values ##
        placeholders = ['', 'null', 'NULL', 'None', 'nan', 'NaN']
        if merged_data.isin(placeholders).any().any():
            print("There are place holder values in the data")
            mask = merged_data.astype(str).apply(
            lambda s: s.str.strip().isin(placeholders))
            merged_data = merged_data[~mask.any(axis=1)].copy()
        else:
            print("There are no place holder values in the data")

        return merged_data
    
    

    def add_features(self):
        df = self.generate_excess_return()
        volatility = Univarite_Volatility(df)
        df['volatility_GARCH'] = volatility.estimate_volatility_GARCH()
        df['volatility_EWMA'] = volatility.estimate_volatility_EWMA()
        df['volatility_Kalman'] = volatility.estimate_volatility_Kalman()
        
        return df

if __name__ == '__main__':
    stock_code = '600428'
    univariate_return = UnivariateReturn(stock_code=stock_code)
    
    ## save the result to a csv file
    # create a folder called 'asset_daily_return'
    os.makedirs('asset_daily_return', exist_ok=True)
    res = univariate_return.add_features()
    res.to_csv(f'asset_daily_return/{stock_code}.csv', index=False)

