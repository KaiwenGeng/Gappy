import akshare as ak
import os
import pandas as pd
"""
WARNING:
SHOULD NEVER APPLY STANDARD SCALER TO ONE-HOT ENCODING FEATURES
"""
class CategoricalFeatures:
    def __init__(self, symbol):
        self.symbol = symbol

        self.industry_list = ak.stock_board_industry_name_em()['板块名称'].tolist()
    def get_industry_classification(self):
        df = ak.stock_individual_info_em(symbol=self.symbol)
        industry = df.loc[df['item'] == '行业', 'value'].iloc[0]
        return industry
    
    def one_hot_encoding_industry(self):
        # return an array of length len(self.industry_list)
        industry_classification = self.get_industry_classification()
        return [1 if industry_classification == industry else 0 for industry in self.industry_list]


# if __name__ == '__main__':
#     symbol = '600096'
#     categorical_features = CategoricalFeatures(symbol)
#     one_hot_encoding_industry = categorical_features.one_hot_encoding_industry()
#     print(one_hot_encoding_industry)
