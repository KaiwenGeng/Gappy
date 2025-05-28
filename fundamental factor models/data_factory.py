from single_asset_data import UnivariateReturn
import pandas as pd
import os
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
class DataFactory:
    def __init__(self, assets_of_interest, start_date, end_date, target_col):
        self.assets_of_interest = assets_of_interest
        self.start_date = start_date
        self.end_date = end_date
        self.target_col = target_col

    def align_dates(self, dfs, how: str = "inner", fill_value=None):
        """
        Align k DataFrames on a common DatetimeIndex *and* keep a `date` column.

        Parameters
        ----------
        dfs : list[pd.DataFrame]
            One DataFrame per asset, with either
            • a DatetimeIndex already, or
            • a plain `"date"` column that can be parsed.
        how : {"inner", "outer"}, default "inner"
            "inner" → intersection of dates (keeps only fully-overlapping days)  
            "outer" → union of dates    (includes all days, pads holes)
        fill_value : scalar or None
            Value used to fill gaps when `how="outer"`.  Defaults to NaN.

        Returns
        -------
        aligned : list[pd.DataFrame]
            Same order as input, identical index, *plus* an explicit `"date"` column.
        """
        dfs = [df.copy() for df in dfs]
        for df in dfs:
            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" not in df.columns:
                    raise ValueError("Frame has neither DatetimeIndex nor a 'date' column")
                df.index = pd.to_datetime(df["date"])
            df.sort_index(inplace=True)          # chronological order
        idx = dfs[0].index
        for df in dfs[1:]:
            idx = idx.intersection(df.index) if how == "inner" else idx.union(df.index)
        aligned = []
        for df in dfs:
            tmp = df.reindex(idx, fill_value=fill_value)
            tmp["date"] = tmp.index        
            aligned.append(tmp)
        assert all(idx.equals(df.index) for df in aligned)

        return aligned
    def split_onehot_numeric(self,df,
                         keyword = "onehot"):
        """
        get all the cols that contain the keyword, which are believed to be one-hot encoded
        """
        onehot_cols = [c for c in df.columns if keyword in c]
        num_cols = [c for c in df.columns if c not in onehot_cols and c != "date"]
        return onehot_cols, num_cols
        

    def get_data(self):
        df_lists = []
        for asset in self.assets_of_interest:
            if os.path.exists(f'single_asset_data/{asset}_{self.start_date}_{self.end_date}.csv'):
                df = (pd.read_csv(f'single_asset_data/{asset}_{self.start_date}_{self.end_date}.csv'))
                # always make the target list be the last column
                df = df[[c for c in df.columns if c != self.target_col] + [self.target_col]]
                df_lists.append(df)
            else:
                print(f"the data for {asset} does not exist, generating data")
                univariate_return = UnivariateReturn(asset, self.start_date, self.end_date)
                df = univariate_return.process_data()
                os.makedirs('single_asset_data', exist_ok=True)
                df.to_csv(f'single_asset_data/{asset}_{self.start_date}_{self.end_date}.csv', index=False)
                # always make the target list be the last column
                df = df[[c for c in df.columns if c != self.target_col] + [self.target_col]]
                df_lists.append(df)
        df_lists = self.align_dates(df_lists)
        return df_lists
    
    def df_to_np(self, df_list_aligned):

        onehot_cols, num_cols = self.split_onehot_numeric(df_list_aligned[0]) # the col names are the same for all dfs
        # stack the numerics
        numerics_vars = np.stack([df[num_cols].values for df in df_list_aligned], axis=0)
        # stack the onehots
        onehots_vars = np.stack([df[onehot_cols].values for df in df_list_aligned], axis=0)
        return numerics_vars, onehots_vars
    
    def temporal_split(self,
                   numerics_vars,
                   onehots_vars,
                   train_ratio = 0.7,
                   val_ratio   = 0.1,
                   scaler = None):
        """
        Chronological split → train / val / test  + StandardScaler on numerics.

        Parameters
        ----------
        numerics_vars : (K, T, n_num)
        onehots_vars  : (K, T, n_cat)
        train_ratio   : fraction of earliest days for training
        val_ratio     : fraction (of total) for validation
                        (test gets the remainder)
        scaler        : optionally pass a pre-fitted StandardScaler

        Returns
        -------
        num_train, num_val, num_test : scaled float32 arrays
        cat_train, cat_val, cat_test : 0/1 float32 arrays
        scaler                       : the fitted StandardScaler
        """
        num_of_assets, num_of_days, _ = numerics_vars.shape

        # 1 — compute slice indices (earliest → latest)
        train_end = int(num_of_days * train_ratio)
        val_end   = int(num_of_days * (train_ratio + val_ratio))

        # 2 — slice numerics & one-hots
        num_train = numerics_vars[:, :train_end]
        num_val   = numerics_vars[:, train_end:val_end] if val_ratio > 0 else None
        num_test  = numerics_vars[:, val_end:]

        cat_train = onehots_vars[:, :train_end]
        cat_val   = onehots_vars[:, train_end:val_end] if val_ratio > 0 else None
        cat_test  = onehots_vars[:, val_end:]

        # 3 — fit scaler on *training* numerics only
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(num_train.reshape(-1, num_train.shape[-1]))

        # 4 — transform each split with the same scaler
        def scale(arr):
            return scaler.transform(arr.reshape(-1, arr.shape[-1]))\
                        .reshape(arr.shape).astype(np.float32)

        num_train = scale(num_train)
        num_test  = scale(num_test)
        if num_val is not None:
            num_val = scale(num_val)

        # 5 — cast one-hots to float32 (leave values unchanged)
        cat_train = cat_train.astype(np.float32)
        cat_test  = cat_test.astype(np.float32)
        if cat_val is not None:
            cat_val = cat_val.astype(np.float32)
        

        continuous_vars = {"train": num_train, "val": num_val, "test": num_test}
        categorical_vars = {"train": cat_train, "val": cat_val, "test": cat_test}
        return continuous_vars, categorical_vars, scaler


        
class FactorCSDataset(Dataset):
    def __init__(self,
                 cont_arr,   # (K, T, n_num)
                 cat_arr):  # (K, T, n_cat)

        self.cont = torch.as_tensor(cont_arr, dtype=torch.float32)
        self.cat  = torch.as_tensor(cat_arr,  dtype=torch.float32)

        self.K, self.T, self.n_num = self.cont.shape
        self.n_cat = self.cat.shape[-1]
        assert self.cat.shape[:2] == (self.K, self.T)

    def __len__(self) -> int:
        return self.T - 1         

    def __getitem__(self, t: int):   
        # exposures at t-1
        B_num = self.cont[:, t, :]     # use all numerical features
        B_cat = self.cat[:,  t, :]     # use all categorical features
        B     = torch.cat([B_num, B_cat], dim=-1)   # (K, F_num+F_cat)
        r = self.cont[:, t + 1, -1]      # (K,)
        return B, r 

# if __name__ == '__main__':
#     assets_of_interest = ['600096', '600428', '300015']
#     start_date = '20000101'
#     end_date = '22220523'
#     data_factory = DataFactory(assets_of_interest, start_date, end_date, target_col = "daily excess return")
#     df_list_aligned = data_factory.get_data()
#     numerics, onehots = data_factory.df_to_np(df_list_aligned)
#     continuous_vars, categorical_vars, scaler = data_factory.temporal_split(numerics, onehots) # need the scaler for inverse transform
#     BATCH_SIZE = 1
#     NUM_WORKERS = 4
#     train_loader = DataLoader(FactorCSDataset(continuous_vars["train"], categorical_vars["train"]), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
#     val_loader = DataLoader(FactorCSDataset(continuous_vars["val"], categorical_vars["val"]), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
#     test_loader = DataLoader(FactorCSDataset(continuous_vars["test"], categorical_vars["test"]), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

