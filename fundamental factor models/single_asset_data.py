import akshare as ak
import pandas as pd
import warnings
import numpy as np
import os
from get_rf import get_risk_free_rate
from volatility_estimation import Univarite_Volatility
from categorical_features import CategoricalFeatures


class UnivariateReturn:
    '''
    arithmetic (simple) excess returns is used! 
    '''
    def __init__(self, stock_code, start_date, end_date, period='daily', adjust='hfq', use_log_return_for_volatility=True):
        ## the setting of end date is intentional
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.adjust = adjust
        self.use_log_return_for_volatility = use_log_return_for_volatility

    def get_daily_rf(self):
        # if already exists, return the existing file
        if os.path.exists(f'risk_free_rate/daily_risk_free_rate.csv'):
            return pd.read_csv(f'risk_free_rate/daily_risk_free_rate.csv')
        else:
            return get_risk_free_rate()
    
    def winsorize_series(self, s, lower=0.01, upper=0.99):
        lo, hi = s.quantile([lower, upper])
        return s.clip(lower=lo, upper=hi)
    
    def get_market_data(self):
        szzs = ak.stock_zh_a_hist(symbol='000001', start_date=self.start_date, end_date=self.end_date, period=self.period, adjust=self.adjust)
        szzs['return'] = szzs['涨跌幅'] / 100.0
        szzs.drop(columns=['涨跌幅'], inplace=True)
        szzs.drop(columns=['开盘'], inplace=True)
        szzs.drop(columns=['收盘'], inplace=True)
        szzs.drop(columns=['最高'], inplace=True)
        szzs.drop(columns=['最低'], inplace=True)
        szzs.drop(columns=['换手率'], inplace=True)
        # add prefix to the column names, except for the date column
        szzs.rename(columns=lambda x: f'szzs_{x}' if x != '日期' else x, inplace=True)
        # sczs = ak.stock_zh_a_hist(symbol='399006', start_date=self.start_date, end_date=self.end_date, period=self.period, adjust=self.adjust)
        # sczs['return'] = sczs['涨跌幅'] / 100.0
        # sczs.drop(columns=['涨跌幅'], inplace=True)
        # sczs.drop(columns=['开盘'], inplace=True)
        # sczs.drop(columns=['收盘'], inplace=True)
        # sczs.drop(columns=['最高'], inplace=True)
        # sczs.drop(columns=['最低'], inplace=True)
        # sczs.drop(columns=['换手率'], inplace=True)
        # # add prefix to the column names, except for the date column
        # sczs.rename(columns=lambda x: f'sczs_{x}' if x != '日期' else x, inplace=True)
        return szzs
        
    def get_stock_data(self):
        '''
        for categorical features, we need to use one-hot encoding
        '''
        ## 默认特征
        ## 日期    股票代码    开盘    收盘    最高    最低     成交量           成交额     振幅    涨跌幅   涨跌额    换手率
        result = ak.stock_zh_a_hist(symbol=self.stock_code, start_date=self.start_date, end_date=self.end_date, period=self.period, adjust=self.adjust)
        ## check if there's 0 in 收盘
        if self.adjust != 'qfq':
            # 前复权可能出现0 甚至负值
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
        result['return'] = self.winsorize_series(result['return'])
        # remove the “股票代码” column
        result.drop(columns=['股票代码'], inplace=True)
        # remove others that are not needed
        result.drop(columns=['涨跌幅'], inplace=True)
        result.drop(columns=['开盘'], inplace=True)
        result.drop(columns=['收盘'], inplace=True)
        result.drop(columns=['最高'], inplace=True)
        result.drop(columns=['最低'], inplace=True)
        result.drop(columns=['换手率'], inplace=True)
        
        return result
    
    def generate_excess_return(self):
        risk_free_rate = self.get_daily_rf()
        stock_data = self.get_stock_data()
        risk_free_rate['日期'] = pd.to_datetime(risk_free_rate['日期'])
        stock_data['日期'] = pd.to_datetime(stock_data['日期'])
        ## merge the risk free rate and the stock data
        merged_data = pd.merge(stock_data, risk_free_rate, how='left')
        merged_data['daily excess return'] = merged_data['return'] - merged_data['daily risk free rate']
        merged_data['cumulative excess return'] = (1 + merged_data['daily excess return']).cumprod() - 1
        merged_data['log excess return'] = np.log(1 + merged_data['daily excess return'])
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
    
    def add_market_features(self, df):
        szzs = self.get_market_data()
        df['日期'] = pd.to_datetime(df['日期'])
        szzs['日期'] = pd.to_datetime(szzs['日期'])
        # merge the market data with the stock data
        df = pd.merge(df, szzs, how='left', on='日期')
        return df

    def add_volatility_features(self, df):
        
        volatility = Univarite_Volatility(df, use_log_return=self.use_log_return_for_volatility)
        df['volatility_GARCH'] = volatility.estimate_volatility_GARCH()
        df['volatility_EWMA'] = volatility.estimate_volatility_EWMA()
        df['volatility_Kalman'] = volatility.estimate_volatility_Kalman()
        return df
    
    def add_categorical_features(self, df):
        categorical_features = CategoricalFeatures(self.stock_code)
        industry_one_hot_encoding = categorical_features.one_hot_encoding_industry() # it's a list of length len(self.industry_list)

        num_of_industries = len(industry_one_hot_encoding)
        industry_one_hot_encoding_array = np.tile(industry_one_hot_encoding, (len(df), 1))
        industry_enc_names = [f"industry_onehot_{i}" for i in range(num_of_industries)]
        df[industry_enc_names] = industry_one_hot_encoding_array
        return df
    
    def process_data(self):
        df = self.generate_excess_return()
        df = self.add_volatility_features(df)
        df = self.add_market_features(df)
        df = self.add_categorical_features(df)
        # rename the "日期" column to "date"
        df.rename(columns={'日期': 'date'}, inplace=True)
        return df
    

if __name__ == '__main__':
    stock_code = '600096'
    start_date = '20190101'
    end_date = '22220523'
    
    univariate_return = UnivariateReturn(stock_code=stock_code, start_date='20190101', end_date='22220523')
    ## save the result to a csv file
    # create a folder called 'asset_daily_return'
    os.makedirs('single_asset_data', exist_ok=True)
    res = univariate_return.process_data()
    res.to_csv(f'single_asset_data/{stock_code}_{start_date}_{end_date}.csv', index=False)

