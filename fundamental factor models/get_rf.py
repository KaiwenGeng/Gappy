import akshare as ak
import os
import pandas as pd
def get_risk_free_rate():
    """
    年化利率为True时，将利率除以365
    output:一个包含日期和无风险利率的DataFrame, 小数利率（例如 1% 表示为 0.01)
    从 2006-10-08 开始
    """
    ##将隔夜拆借利率视为无风险利率
    rate_interbank_df = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="隔夜")

    rate_interbank_df['daily risk free rate'] = rate_interbank_df['利率'] / 100.0 
    # 精确复利
    rate_interbank_df['daily risk free rate'] = (1 + rate_interbank_df['daily risk free rate']) ** (1/365) - 1

    # rename '报告日' to '日期'
    rate_interbank_df.rename(columns={'报告日': '日期'}, inplace=True)
    rate_interbank_df = rate_interbank_df[['日期', 'daily risk free rate']]
    return rate_interbank_df


# if __name__ == "__main__":
#     rf_df = get_risk_free_rate()
#     # save the result to csv
#     os.makedirs('risk_free_rate', exist_ok=True)
#     rf_df.to_csv('risk_free_rate/daily_risk_free_rate.csv', index=False)

