# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:09:42 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from pathlib import Path
import sys
from datetime import datetime
import yaml
import argparse
from datetime import date, timedelta, datetime


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.logutils import FishStyleLogger
from update.database_handler import DailyUpdateSender, DailyUpdateReader, DailyUpdateMsgSender
from update.loop_check import CheckDb, ProcessUpdateCoordinator
from utils.dateutils import trading_days_set
from core.generate_by_batch_v1 import BatchGenerate


# %%
def get_previous_n_trading_day(input_date: str, n: int) -> str:
    """
    获取指定日期的第 n 天前的最近一个交易日。

    参数:
        input_date (str): 输入日期，格式为 YYYYMMDD（如 "20200101"）
        n (int): 向前数第 n 天，从该日期找到最近的交易日
    
    返回:
        str: 第 n 天前的最近一个交易日，格式为 YYYYMMDD（如 "20191231"）
    """
    # 将输入日期转换为 datetime.date 类型
    input_date_dt = datetime.strptime(input_date, "%Y%m%d").date()

    # 计算第 n 天前的日期
    target_date = input_date_dt - timedelta(days=n)

    # 如果第 n 天前的日期不是交易日，找到最近的前一个交易日
    while target_date not in trading_days_set:
        target_date -= timedelta(days=1)

    # 返回结果转换为字符串格式 YYYYMMDD
    return target_date.strftime("%Y%m%d")


def update_factors(update_name=None, delay=1):
    
    # 更新至 ————
    date_today = datetime.today().strftime('%Y%m%d')
    target_date = get_previous_n_trading_day(date_today, delay)
    
    # 读取路径配置
    path_config = load_path_config(project_dir)
    param_dir = Path(path_config['workflow_param'])
    
    # 读取参数
    with open(param_dir / 'update_factors' / f'{update_name}.yaml', "r") as file:
        params = yaml.safe_load(file)
    # params = toml.load(param_dir / 'update_factors' / f'{update_name}.toml')
    
    # 数据库交互
    # Initialize logger and senders
    mysql_name = params['mysql_name']
    author = params['author']
    log = FishStyleLogger()
    daily_update_sender = DailyUpdateSender(mysql_name, author, log=log)
    daily_update_reader = DailyUpdateReader(mysql_name, log=log)
    msg_sender = DailyUpdateMsgSender(mysql_name, author, log=log)
    
    # Initialize check database and coordinator
    check_db_params = params['check_db_params']
    check_db = CheckDb(daily_update_reader, msg_sender, log, config=check_db_params)
    coordinator = ProcessUpdateCoordinator(check_db, daily_update_sender, msg_sender, log)
    coordinator.set_target_date(target_date)
    
    ## update
        
    # 用更新的股票ind，计算更新的org + org的历史与inc拼接 + 做时序变换
    fac_params = params['factors']
    batch_name = fac_params['batch_name']
    output = fac_params['output']
    dependency = fac_params['dependency']
    
    with coordinator(output, dependency):
        if not coordinator.skip_task:
            instance = BatchGenerate(batch_name)
            instance.run()

        
# %%
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-un', '--update_name', type=str, help='update_name')
    parser.add_argument('-dl', '--delay', type=int, help='delay')

    args = parser.parse_args()
    update_name, delay = args.update_name, args.delay
    
    
    update_factors(update_name, delay)

        
# %%
if __name__=='__main__':
    main()