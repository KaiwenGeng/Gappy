import akshare as ak
import pandas as pd
import warnings
import numpy as np
class UnivariateReturn:
    def __init__(self, stock_code, start_date='20000101', end_date='22220523', period='daily', adjust='qfq'):
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.adjust = adjust

    def get_risk_free_rate(self):
        ##将隔夜拆借利率视为无风险利率
        rate_interbank_df = ak.rate_interbank(market="中国银行同业拆借市场", symbol="Chibor人民币", indicator="隔夜")
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
        ## merge the risk free rate and the stock data
        merged_data = pd.merge(stock_data, risk_free_rate, how='left')
        merged_data['daily excess return'] = merged_data['return'] - merged_data['daily risk free rate']
        merged_data['cumulative excess return'] = (1 + merged_data['daily excess return']).cumprod() - 1
        merged_data['log daily excess return'] = np.log(1 + merged_data['daily excess return'])
        merged_data['cumulative log daily excess return'] = merged_data['log daily excess return'].cumsum() 
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

if __name__ == '__main__':
    univariate_return = UnivariateReturn(stock_code='600428')
    res = univariate_return.generate_excess_return()
    print(res)

