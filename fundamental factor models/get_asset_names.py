
import akshare as ak
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time 
# 1. 定义指数代码
# 上证50：000016，沪深300：000300，中证500：000905
index_codes = ["000016", "000300", "000905"]
# index_codes = ["000016"]
# 2. 拉取所有成分股，合并去重
all_codes = set()
for idx in index_codes:
    df = ak.index_stock_cons(symbol=idx)
    # 不同版本列名可能略有差异，常见“品种代码”或“con_code”
    code_col = "品种代码" if "品种代码" in df.columns else "con_code"
    all_codes.update(df[code_col].tolist())

all_codes = sorted(all_codes)
print(f"三大指数合并后共有 {len(all_codes)} 只股票")

# 3. 逐只获取上市日期，并过滤
records = []
for i, code in enumerate(tqdm(all_codes)):
    if code == "000001" or code == "399001":
        continue
    try:
        info = ak.stock_individual_info_em(symbol=code)
    except:
        wait = 1
        print(f"{code}: retry in {wait:.1f}s")
        time.sleep(wait)
    ipo_date = info.loc[info["item"] == "上市时间", "value"].iloc[0]
    # ipo date is an int like 20150101, convert to datetime
    ipo_date = datetime.strptime(str(ipo_date), "%Y%m%d")
    records.append({"code": code, "ipo_date": ipo_date})
    # sleep for 1 second

df_listing = pd.DataFrame(records)

# 4. 过滤出 2015-01-01 之前上市的
cutoff = datetime(2015, 1, 1)
df_before2015 = df_listing[df_listing["ipo_date"] < cutoff]

# 5. 输出最终股票代码列表
result_codes = df_before2015["code"].tolist()
print("2015 年以前上市的股票列表：")
print(result_codes)
# write to a txt file, put every code on 1 line, only 1 line in the file
with open("asset_names.txt", "w") as f:
    for code in result_codes:
        f.write(code + " ")