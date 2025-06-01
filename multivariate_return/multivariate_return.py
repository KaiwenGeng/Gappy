import akshare as ak
import pandas as pd
import warnings
import numpy as np
import os
class AssetsDataFactory:
    def __init__(self, asset_of_interest, start_date='20190101', end_date='22220523', period='daily', adjust='hfq'):
        self.asset_of_interest = asset_of_interest
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

        
    def get_stock_data(self, asset_name):
        '''
        for categorical features, we need to use one-hot encoding
        '''
        ## 默认特征
        ## 日期    股票代码    开盘    收盘    最高    最低     成交量           成交额     振幅    涨跌幅   涨跌额    换手率
        result = ak.stock_zh_a_hist(symbol=asset_name, start_date=self.start_date, end_date=self.end_date, period=self.period, adjust=self.adjust)
        ## check if there's 0 in 收盘
        if self.adjust != 'qfq':
            if result['收盘'].isin([0]).any():
                ## ffill 0 with the previous value
                result.replace(0, method='ffill', inplace=True) 
            else:
                print(f"There are no 0 in the the close column for {asset_name}")
        if '涨跌幅' in result.columns:
            result['return'] = result['涨跌幅'] / 100.0
        else:
            result['return'] = result['收盘'].pct_change()
        return result
    
    def generate_raw_returns(self):
        series_list = []
        for asset in self.asset_of_interest:
            df = self.get_stock_data(asset)[['日期', 'return']].copy()
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.rename(columns={'return': asset})
            df = df.set_index('日期')
            series_list.append(df[asset])
        returns_df = pd.concat(series_list, axis=1, join='outer')
        returns_df = returns_df.sort_index()
        ## check if there are nan values ##
        if returns_df.isnull().any().any():
            print("after merging, there are nan values in the data")
            returns_df = returns_df.dropna()
        else:
            print("after merging, there are no nan values in the data")
        ## check if there are place holder values ##
        placeholders = ['', 'null', 'NULL', 'None', 'nan', 'NaN']
        if returns_df.isin(placeholders).any().any():
            print("after merging, there are place holder values in the data")
            mask = returns_df.astype(str).apply(
            lambda s: s.str.strip().isin(placeholders))
            returns_df = returns_df[~mask.any(axis=1)].copy()
        else:
            print("after merging, there are no place holder values in the data")
        # check the start and end date of the returns_df
        print(f"The start date of the returns_df is {returns_df.index.min()}")
        print(f"The end date of the returns_df is {returns_df.index.max()}")
        print(f"the given start date is {self.start_date}")
        print(f"the given end date is {self.end_date}")
        # get the date col back
        returns_df = returns_df.reset_index().rename(columns={'index':'日期'})
    
        
        return returns_df
    
    def generate_excess_returns(self):
        raw_returns = self.generate_raw_returns()
        risk_free_rate = self.get_risk_free_rate()
        # get the asset names, ie all columns except the date column
        asset_names = raw_returns.columns.drop('日期')
        risk_free_rate['日期'] = pd.to_datetime(risk_free_rate['日期'])
        raw_returns['日期'] = pd.to_datetime(raw_returns['日期'])
        merged_df = pd.merge(raw_returns, risk_free_rate, how='left', on='日期')
        # for each column in raw_returns, subtract the daily risk free rate
        excess_cols = []
        for asset_name in asset_names:
            merged_df[asset_name + ' excess return'] = merged_df[asset_name] - merged_df['daily risk free rate']
            excess_cols.append(asset_name + ' excess return')
        res = merged_df[['日期'] + excess_cols]
        # rename the "日期" column to "date"
        res.rename(columns={'日期': 'date'}, inplace=True)
        return res

if __name__ == '__main__':
    asset_of_interest = ['300015', '600428', '000878', '600096']
    multi_asset_data_factory = AssetsDataFactory(asset_of_interest=asset_of_interest)
    excess_returns_df = multi_asset_data_factory.generate_excess_returns()
    # save the excess_returns_df to a csv file
    os.makedirs('excess_returns', exist_ok=True)
    excess_returns_df.to_csv(f'excess_returns/{asset_of_interest}.csv', index=False)

