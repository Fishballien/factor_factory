# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 15:13:33 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pymysql
import pandas as pd


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


# %% 
if __name__=='__main__':
    wgt = get_index_weight2(idxname="000016")