# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:35:10 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd


from trans_operators.ts import zscore, zscore_fxwd


# %%
factor_path = 'D:/CNIndexFutures/timeseries/factor_factory/sample_data/factors/batch10/LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-org.parquet'


# %%
factor = pd.read_parquet(factor_path)


# %%
# factor_zscore = zscore(factor, period_str='1d')
factor_zscore_fxwd = zscore_fxwd(factor, period_str='1d')