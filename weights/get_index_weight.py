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
import time


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self_defined
from utils.dateutils import (get_cffex_trading_days_by_date_range, get_next_trading_day, 
                             get_previous_n_trading_day, is_last_trading_day_of_month)


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


# =============================================================================
# def get_index_weight(idxname="000852"):
#     sql = """
#     SELECT
#         a.SECURITY_ID,
#         b.TICKER_SYMBOL AS INDEX_SYMBOL,
#         b.SEC_SHORT_NAME AS INDEX_NAME,
#         a.CONS_ID,
#         c.TICKER_SYMBOL AS STOCK_SYMBOL,
#         c.SEC_SHORT_NAME AS STOCK_NAME,
#         c.EXCHANGE_CD,
#         a.EFF_DATE,
#         a.WEIGHT,
#         a.UPDATE_TIME
#     FROM idx_weight a 
#         LEFT JOIN md_security b ON a.SECURITY_ID=b.SECURITY_ID
#         LEFT JOIN md_security c ON a.CONS_ID=c.SECURITY_ID
#     WHERE b.TICKER_SYMBOL='{idxname}' /*输入需查询的指数代码*/
#     """.format(idxname=idxname)
#     data = fetch_database(sql)
#     return data
# =============================================================================

def get_index_weight(idxname="000852"):
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
        a.HERMESFIRSTTIME,
        a.HERMESUPTIME,
        a.UPDATE_TIME
    FROM idx_weight a 
        LEFT JOIN md_security b ON a.SECURITY_ID=b.SECURITY_ID
        LEFT JOIN md_security c ON a.CONS_ID=c.SECURITY_ID
    WHERE b.TICKER_SYMBOL='{idxname}' /*输入需查询的指数代码*/
    """.format(idxname=idxname)
    data = fetch_database(sql)
    return data


def get_index_weight2(idxname="932000"):
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
    FROM csi_idxm_wt_ashare a 
        LEFT JOIN md_security b ON a.SECURITY_ID=b.SECURITY_ID
        LEFT JOIN md_security c ON a.CONS_ID=c.SECURITY_ID
    WHERE b.TICKER_SYMBOL='{idxname}' /*输入需查询的指数代码*/
    """.format(idxname=idxname)
    data = fetch_database(sql)
    return data

def get_index_weight3(idxname="932000"):
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
    FROM idx_cons_csi a 
        LEFT JOIN md_security b ON a.SECURITY_ID=b.SECURITY_ID
        LEFT JOIN md_security c ON a.CONS_ID=c.SECURITY_ID
    WHERE b.TICKER_SYMBOL='{idxname}' /*输入需查询的指数代码*/
    """.format(idxname=idxname)
    data = fetch_database(sql)
    return data


# =============================================================================
# def index_weight_data(idxcode, target_date, save_dir=''):
#     newdata = get_index_weight(idxcode)  
#     # breakpoint()
#     if len(newdata) == 0: # zz2000 获取渠道不一样
#         newdata = get_index_weight2(idxcode)
# 
#     newdata.to_parquet(save_dir / f'{idxcode}_sql_org.parquet')
#     
#     newdata['con_code'] = newdata['STOCK_SYMBOL'] + "." + newdata["EXCHANGE_CD"]
#     newdata = newdata.rename(columns={
#         'EFF_DATE': 'trade_date',
#         'WEIGHT': 'weight',
#         })[['con_code', 'trade_date', 'weight']]
#     newdata.to_parquet(save_dir / f'{idxcode}_index_weight.parquet')
#     
#     newdata['weight'] = newdata['weight'].astype(np.float64)
#     data = newdata.pivot(index='trade_date', columns='con_code',values='weight').fillna(0)
#     data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
#     # daily_index = pd.date_range(start=data.index.min(), end=target_date, freq='D') # data.index.max()
#     # breakpoint()
#     daily_index = get_cffex_trading_days_by_date_range(start_date=data.index.min().date(), end_date=target_date)
#     daily_index = pd.to_datetime([get_next_trading_day(dt) for dt in daily_index])
#     daily_weights = data.reindex(daily_index, method='ffill').fillna(0)
#     # daily_weights.index = daily_weights.index.map(lambda dt: get_next_trading_day(dt))
#     daily_weights.to_parquet(save_dir / f'{idxcode}.parquet')
#     
#     return daily_weights
# =============================================================================


def index_weight_data(idxcode, target_date, save_dir='', check_if_inc=False):
    # 获取新数据
    newdata = get_index_weight(idxcode)  
    # breakpoint()
    if len(newdata) == 0:  # zz2000 获取渠道不一样
        newdata = get_index_weight2(idxcode)
    newdata.to_parquet(save_dir / f'{idxcode}_sql_org.parquet')
    
    # 处理新数据
    newdata['con_code'] = newdata['STOCK_SYMBOL'] + "." + newdata["EXCHANGE_CD"]
    newdata = newdata.rename(columns={
        'EFF_DATE': 'trade_date',
        'WEIGHT': 'weight',
        })[['con_code', 'trade_date', 'weight']]
    
    # 检查是否存在旧的权重数据文件
    old_weight_file = save_dir / f'{idxcode}_index_weight.parquet'
    if old_weight_file.exists():
        # 读取旧数据
        olddata = pd.read_parquet(old_weight_file)
        
        # 找出旧数据中存在的日期
        old_dates = set(olddata['trade_date'].unique())
        
        # 筛选出旧数据中不存在的新数据日期
        new_incremental = newdata[~newdata['trade_date'].isin(old_dates)]
        
        if check_if_inc and len(new_incremental) == 0:
            return 'NOINC', None
        
        # 合并旧数据和增量新数据
        mergeddata = pd.concat([olddata, new_incremental], ignore_index=True)
    else:
        # 如果没有旧数据，直接使用新数据
        mergeddata = newdata
    
    # 保存更新后的权重数据
    mergeddata.to_parquet(save_dir / f'{idxcode}_index_weight.parquet')
    
    # 继续处理成宽格式数据
    mergeddata['weight'] = mergeddata['weight'].astype(np.float64)
    data = mergeddata.pivot(index='trade_date', columns='con_code', values='weight').fillna(0)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
    
    # 生成日期索引
    daily_index = get_cffex_trading_days_by_date_range(start_date=data.index.min().date(), end_date=target_date)
    daily_index = pd.to_datetime([get_next_trading_day(dt) for dt in daily_index])
    
    # 填充每日权重数据
    daily_weights = data.reindex(daily_index, method='ffill').fillna(0)
    daily_weights.to_parquet(save_dir / f'{idxcode}.parquet')
    
    return 'SUCCESS', daily_weights
    

# %%
if __name__ == "__main__":
    delay = 0
    # index_code = ['000300']
    index_code = ["000016", "000300", "000905", "000852", "932000", "000985"]
    # index_code = [
    #     "512050", "510300", "159845", "159915", 
    #     "159338", "588000", "588190", "562500", "512480", 
    #     "588200", "159819", "159995", "159992", "159869", "512800"
    #     ]
    # save_dir = Path(r'D:/CNIndexFutures/timeseries/factor_factory/sample_data/weights_matrix_fr_sql')
    save_dir = Path(r'/mnt/data1/stockweights')
    flag_dir = save_dir / 'flag'
    save_dir.mkdir(parents=True, exist_ok=True)
    flag_dir.mkdir(parents=True, exist_ok=True)
    
    
    date_today = datetime.today().date()
    target_date = get_previous_n_trading_day(date_today, delay)
    is_last = is_last_trading_day_of_month(target_date)
    if is_last:
        time.sleep(60*60*4)
    
    if not is_last:
        wgt_dict = {}
        for idxcode in index_code: #["000016", "000300", "000905", "000852", "932000", "000985"]:  
            wgt_dict[idxcode] = index_weight_data(idxcode, target_date, save_dir)
        with open(flag_dir / f'{target_date}.json', 'w') as f:
            json.dump({}, f)
    else:
        wgt_dict = {}
        status_dict = {idxcode: '' for idxcode in index_code}
        
        while True:
            # Check all index codes
            for idxcode in index_code:
                if status_dict[idxcode] != 'SUCCESS':
                    status, res = index_weight_data(idxcode, target_date, save_dir, check_if_inc=True)
                    status_dict[idxcode] = status
                    if status == 'SUCCESS':
                        wgt_dict[idxcode] = res
            
            # Check if all statuses are 'SUCCESS'
            if all(status == 'SUCCESS' for status in status_dict.values()):
                break
            else:
                # Sleep for 5 minutes if not all are success
                time.sleep(300)  # 5 minutes in seconds
        
        # After successful completion, save the flag file
        with open(flag_dir / f'{target_date}.json', 'w') as f:
            json.dump({}, f)
            
      
