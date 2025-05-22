# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:47:07 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
'''
fr sk: 这个表的数据可能是2022-05-26开通的，这之后的数据都是月底当天的晚上八点半左右更新
 -> 隔日0点更新应该没问题
'''
# %%
import sys
import pymysql
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self_defined
from utils.dateutils import get_cffex_trading_days_by_date_range, get_next_trading_day, get_previous_n_trading_day


# %%
def fetch_database(sql, database="TonglianData"):
    conn = pymysql.connect(host='172.16.30.20',
                           user='reader',
                           password='Qingchun123.com',
                           database=database)

    cur = conn.cursor()
    cur.execute(sql)
    output = cur.fetchall()
    # print(output)
    # conn.close()
    data = pd.read_sql(sql, conn) # , index_col='brand_id'
    return data # output


def get_index_weight(idxname="932000", table_list=None):
    """
    从多个可能的表中获取指数成分股权重数据
    
    Args:
        idxname (str): 指数代码
        table_list (list): 尝试查询的表名列表，如果为None则使用默认列表
        
    Returns:
        DataFrame: 查询结果数据
    """
    print(f"开始查询{idxname}")
    # 如果未提供表名列表，使用默认列表
    if table_list is None:
        table_list = ["idx_cons_csi", "idx_cons_citic", "idx_cons_cni", "idx_cons_sw"]
    
    # 逐一尝试每个表
    for table_name in table_list:
        sql = """
        SELECT
            a.SECURITY_ID,
            b.TICKER_SYMBOL AS INDEX_SYMBOL,
            b.SEC_SHORT_NAME AS INDEX_NAME,
            a.CONS_ID,
            c.TICKER_SYMBOL AS STOCK_SYMBOL,
            c.SEC_SHORT_NAME AS STOCK_NAME,
            c.EXCHANGE_CD,
            a.EFF_DATE,
            a.WEIGHT,
            a.UPDATE_TIME
        FROM {table} a 
            LEFT JOIN md_security b ON a.SECURITY_ID=b.SECURITY_ID
            LEFT JOIN md_security c ON a.CONS_ID=c.SECURITY_ID
        WHERE b.TICKER_SYMBOL='{idxname}'
        """.format(table=table_name, idxname=idxname)
        
        try:
            data = fetch_database(sql)
            
            # 检查是否获取到数据
            if data is not None and len(data) > 0:
                print(f"成功从表 {table_name} 获取到 {len(data)} 条数据")
                return data
                
        except Exception as e:
            print(f"从表 {table_name} 查询时出错: {str(e)}")
    
    # 所有表都尝试过但没有数据
    print(f"在所有表中都未找到指数 {idxname} 的数据")
    return None


def index_weight_data(idxcode, target_date, save_dir='', table_list=[]):
    newdata = get_index_weight(idxcode, table_list)  
    newdata.to_parquet(save_dir / f'{idxcode}_sql_org.parquet')
    
    newdata['con_code'] = newdata['STOCK_SYMBOL'] + "." + newdata["EXCHANGE_CD"]
    newdata = newdata.rename(columns={
        'EFF_DATE': 'trade_date',
        'WEIGHT': 'weight',
        })[['con_code', 'trade_date', 'weight']]
    newdata.to_parquet(save_dir / f'{idxcode}_index_weight.parquet')
    
    newdata['weight'] = newdata['weight'].astype(np.float64)
    data = newdata.pivot(index='trade_date', columns='con_code',values='weight').fillna(0)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
    # daily_index = pd.date_range(start=data.index.min(), end=target_date, freq='D') # data.index.max()
    # breakpoint()
    daily_index = get_cffex_trading_days_by_date_range(start_date=data.index.min().date(), end_date=target_date)
    daily_index = pd.to_datetime([get_next_trading_day(dt) for dt in daily_index])
    daily_weights = data.reindex(daily_index, method='ffill').fillna(0)
    # daily_weights.index = daily_weights.index.map(lambda dt: get_next_trading_day(dt))
    daily_weights.to_parquet(save_dir / f'{idxcode}.parquet')
    
    return daily_weights
    

# %%
if __name__ == "__main__":
    delay = 0
    table_list = [
        "idx_cons",
        "idx_pre_cons",
        "idx_weight",
        "hsi_idxm_weight",
        "cni_idxm_weight",
        "sw_idxm_weight",
        "csi_idxm_wt_ashare",
        "csi_idxm_wt_bond",
        "csi_idxm_wt_other",
        "cb_idxm_weight_0",
        "cb_idxm_weight_1",
        "idx_cons_core",
        "idx_wt_aa_hs",
        "idx_opn_wt_aa_hs",
        "idx_wt_aa_hsj",
        "idx_opn_wt_aa_hsj",
        "idx_cons_csi",
        "idx_cons_citic",
        "idx_cons_cni",
        "idx_cons_sw",
        "idx_cons_hsi",
        "idx_cons_cbond",
        "idx_cons_dy"
    ]

    # index_code = ["000016", "000300", "000905", "000852", "932000", "000985"]
    # index_code = [
    #     "000510", "000300", "000852", "000905",
    #     "000688", "000698", "H30590", "H30184", "000685",
    #     "931071", "980017", "931152", "930901", "399986"
    # ]
    # index_code = [
    #     "399673"
    # ]
    index_code = [
        "399006",
        "932000",
        "000852",
        "000905",
        "000300",
        "000001",
        "399001",
        "000903",
        "000688",
        "399371",
        "931052",
        "399370",
        "399346",
        "000015",
        "399321",
        "399662",
        "000998",
        "881155",
        "399986",
        "000006",
        "399241",
        "399393",
        "000819",
        "399395",
        "000820",
        "399998",
        "399809",
        "399235",
        "980048",
        "399440",
        "930606",
        "000807",
        "399959",
        "399967",
        "399417",
        "000941",
        "399997",
        "980031",
        "1B0056",
        "399337"
    ]
    index_code = ["000015"]
    # save_dir = Path(r'D:/CNIndexFutures/timeseries/factor_factory/sample_data/weights_matrix_fr_sql')
    save_dir = Path(r'/mnt/data1/stockweights')
    flag_dir = save_dir / 'flag'
    save_dir.mkdir(parents=True, exist_ok=True)
    flag_dir.mkdir(parents=True, exist_ok=True)
    
    
    date_today = datetime.today().date()
    target_date = get_previous_n_trading_day(date_today, delay)
    
    wgt_dict = {}
    for idxcode in index_code: #["000016", "000300", "000905", "000852", "932000", "000985"]:  
        try:
            wgt_dict[idxcode] = index_weight_data(idxcode, target_date, save_dir, table_list)
        except:
            print(f'{idxcode} failed')
    with open(flag_dir / f'{target_date}.json', 'w') as f:
        json.dump({}, f)
    
      
