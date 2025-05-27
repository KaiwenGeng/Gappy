from single_asset_data import UnivariateReturn:
import pandas as pd
import os

class DataFactory:
    def __init__(self, assets_of_interest, start_date, end_date):
        self.assets_of_interest = assets_of_interest
        self.start_date = start_date
        self.end_date = end_date
    
    def get_data(self):
        df_lists = []
        for asset in self.assets_of_interest:
            if os.path.exists(f'single_asset_data/{asset}_{self.start_date}_{self.end_date}.csv'):
                df_lists.append(pd.read_csv(f'single_asset_data/{asset}_{self.start_date}_{self.end_date}.csv'))

            else:
                univariate_return = UnivariateReturn(asset, self.start_date, self.end_date)
                df_lists.append(univariate_return.add_features())
        return df_lists


if __name__ == '__main__':
    assets_of_interest = ['600096', '600428', '000878']
    start_date = '20190101'
    end_date = '22220523'
    data_factory = DataFactory(assets_of_interest, start_date, end_date)
    data = data_factory.get_data()
    

