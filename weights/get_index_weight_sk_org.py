import pymysql
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt 
import sys 
import os
sys.path.insert(0, "/home/shaokai/")
# from stock.data_processing.database import *


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

def index_weight_data(idxcode, savedir='', sampleidx=None, samplecols=None, verbose=True):
    newdata = get_index_weight(idxcode)  
    if len(newdata) == 0: # zz2000 获取渠道不一样
        newdata = get_index_weight2(idxcode)  
    breakpoint()
    newdata['stock'] = newdata['STOCK_SYMBOL'] + "." + newdata["EXCHANGE_CD"]
    data = newdata.pivot(index='EFF_DATE', columns='stock',values='WEIGHT',)
    data.index = pd.to_datetime(data.index)
    if sampleidx and samplecols:
        # 纳入上月底的数据，确保第一条数据有权重 # 权重数据index为每月底
        lastweight = sampleidx.index[0] - relativedelta(months=1, day=1) # 上月初
        data = data[data.index > lastweight].fillna(0)
        indexlist = data.index[:1].append(sampleidx) 
        # 补上缺的股票
        missing = [s for s in data.columns if s not in samplecols]
        collist = samplecols.tolist() + missing
        d = data.reindex(index=indexlist, columns=collist).ffill().fillna(0)
    else:
        d = data
    # d.to_parquet(f"{savedir}/{idxcode}.parquet")
    if verbose: print(idxcode, (d>0).sum(axis=1))
    return d
    

if __name__ == "__main__":
    # twap = pd.read_parquet('/mnt/Data131/stock_data/stock_data/daily/twap.parquet') 
    savedir = "/mnt/Data/shaokai/try/indexweight"
    for idxcode in ["000300"]: #["000300", "000905", "000852", "932000"]:  
        d = index_weight_data(idxcode, savedir,)
      

 
 