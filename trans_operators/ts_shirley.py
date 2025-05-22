# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 10:50:05 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd
import numpy as np
import talib as ta
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression


# %%
#### 1. Fractional Differentiation Functions
def _get_weight_ffd(d, thres, lim):
    """
    Calculate weights for fractional differentiation.
    
    Args:
        d (float): Differentiation order (fractional)
        thres (float): Threshold for weight significance
        lim (int): Maximum number of weights to consider
        
    Returns:
        np.array: Weight vector for fractional differentiation
    """
    w, k = [1.], 1
    ctr = 0
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        ctr += 1
        if ctr == lim - 1:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def frac_diff_ffd(x, d, thres, lim, gap, shift):
    """
    Perform fractional differentiation on a time series using FFD method.
    
    Args:
        x (array): Input time series
        d (float): Differentiation order
        thres (float): Threshold for weight significance
        lim (int): Maximum number of weights
        gap (int): Initial gap of NaN values
        shift (int): Lag shift for the weights
        
    Returns:
        np.array: Fractionally differentiated time series
    """
    if lim is None:
        lim = len(x)
    w = _get_weight_ffd(d, thres, lim)
    output = [np.nan] * gap  # First part is NaN
    for i in range(gap, len(x)):
        data = x[max(i-lim, 0):i + 1]
        if shift > 0:
            weight = w[max(-len(data), -len(w)):(-shift)]
        else:
            weight = w[(-len(data)):]
        output.append(np.dot(weight.T, data[-len(weight):])[0])
    return np.array(output)

def fracdiff1d(ts, d, thres, lim, gap, shift):
    """
    Apply fractional differentiation to a pandas Series.
    
    Args:
        ts (pd.Series): Input time series
        d (float): Differentiation order
        thres (float): Threshold for weight significance
        lim (int): Maximum number of weights
        gap (int): Initial gap of NaN values
        shift (int): Lag shift for the weights
        
    Returns:
        pd.Series: Fractionally differentiated series with preserved index
    """
    tmp = ts.dropna()
    if len(tmp) <= gap:  # If less than gap, not enough data to calculate
        return ts * np.nan
    return pd.Series(frac_diff_ffd(tmp.values, d, thres, lim, gap, shift), index=tmp.index, name=ts.name).reindex(ts.index)

def fracdiff(df, d, thres=1e-5, lim=None, gap=10, shift=0):
    """
    Apply fractional differentiation to a DataFrame or Series.
    
    Args:
        df (pd.DataFrame or pd.Series): Input data
        d (float): Differentiation order
        thres (float): Threshold for weight significance
        lim (int): Maximum number of weights
        gap (int): Initial gap of NaN values
        shift (int): Lag shift for the weights
        
    Returns:
        pd.DataFrame or pd.Series: Fractionally differentiated data with same shape as input
    """
    if len(df.shape) == 1 or isinstance(df, pd.Series):
        return fracdiff1d(df, d, thres, lim, gap, shift)  # If single column
    else:
        # 应用到每一列并保持DataFrame结构
        return df.apply(lambda x: fracdiff1d(x, d, thres, lim, gap, shift))

def fracdiff_ma(df, d, window):
    """
    Apply fractional differentiation followed by moving average smoothing.
    
    Args:
        df (pd.DataFrame or pd.Series): Input data
        d (float): Differentiation order
        window (int): Moving average window size
        
    Returns:
        pd.DataFrame or pd.Series: Fractionally differentiated and smoothed data with same shape as input
    """
    # d represents differentiation order; window represents smoothing window size
    return fracdiff(df, d, shift=0).rolling(window).mean()

def fracdiff_rev(df, d, window):
    """
    Apply reversed fractional differentiation with moving average smoothing.
    
    Args:
        df (pd.DataFrame or pd.Series): Input data
        d (float): Differentiation order
        window (int): Moving average window size
        
    Returns:
        pd.DataFrame or pd.Series: Negative of the shifted fractional diff with MA with same shape as input
    """
    return -fracdiff(df, d, shift=1).rolling(window).mean()

def fracdiff_marev(df, d, window):
    """
    Complex fractional differentiation with multiple shifts.
    
    Args:
        df (pd.DataFrame or pd.Series): Input data
        d (float): Differentiation order
        window (int): Moving average window size
        
    Returns:
        pd.DataFrame or pd.Series: Combined fractional differentiation with various shifts with same shape as input
    """
    return fracdiff(df, d, shift=0).rolling(window).mean() - 2*fracdiff(df, d, shift=1) + fracdiff(df, d, shift=10)

# 修改技术指标函数
def ma_func(input_ts, wind):
    """
    Calculate Simple Moving Average (MA) for time series.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size for MA calculation
        
    Returns:
        pd.DataFrame or pd.Series: Moving average values with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        result = ta.MA(input_ts, timeperiod=wind, matype=0)
        result.iloc[0] = input_ts.iloc[0]
        for i in range(1, wind):
            result.iloc[i] = ta.MA(input_ts.iloc[:i+1], timeperiod=i, matype=0).iloc[-1]
        return result
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_series = ta.MA(series, timeperiod=wind, matype=0)
            result_series.iloc[0] = series.iloc[0]
            for i in range(1, wind):
                result_series.iloc[i] = ta.MA(series.iloc[:i+1], timeperiod=i, matype=0).iloc[-1]
            result_df[col] = result_series
        return result_df

def ema_func(input_ts, wind):
    """
    Calculate Exponential Moving Average (EMA) for time series.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size for EMA calculation
        
    Returns:
        pd.DataFrame or pd.Series: EMA values with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        result = ta.MA(input_ts, timeperiod=wind, matype=1)
        result.iloc[0] = input_ts.iloc[0]
        for i in range(1, wind):
            result.iloc[i] = ta.MA(input_ts.iloc[:i+1], timeperiod=i, matype=1).iloc[-1]
        return result
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_series = ta.MA(series, timeperiod=wind, matype=1)
            result_series.iloc[0] = series.iloc[0]
            for i in range(1, wind):
                result_series.iloc[i] = ta.MA(series.iloc[:i+1], timeperiod=i, matype=1).iloc[-1]
            result_df[col] = result_series
        return result_df

def bbands_func(input_ts, wind, up=2, down=2):
    """
    Calculate Bollinger Bands and generate trading signals.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size for Bollinger Bands
        up (float): Upper band deviation multiplier
        down (float): Lower band deviation multiplier
        
    Returns:
        pd.DataFrame or pd.Series: Trading signals (-1 for sell, 1 for buy) with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        upperband, middleband, lowerband = ta.BBANDS(input_ts, timeperiod=wind, nbdevup=2, nbdevdn=2, matype=0)
        for i in range(2, wind):
            upperband.iloc[i], middleband.iloc[i], lowerband.iloc[i] = [x.iloc[-1] for x in ta.BBANDS(input_ts.iloc[:i+1], timeperiod=i, nbdevup=up, nbdevdn=down, matype=0)]
        buy_sig = (input_ts >= upperband)
        sell_sig = (input_ts <= lowerband)
        factor_bbands = pd.Series(np.nan, input_ts.index)
        factor_bbands[buy_sig] = 1
        factor_bbands[sell_sig] = -1
        factor_bbands = factor_bbands.ffill()
        
        factor_bbands2 = pd.Series(np.nan, input_ts.index)
        factor_bbands2[(factor_bbands == 1) & (input_ts <= middleband)] = 0
        factor_bbands2[(factor_bbands == -1) & (input_ts >= middleband)] = 0  # Close position signals
        factor_bbands2[buy_sig] = 1
        factor_bbands2[sell_sig] = -1
        factor_bbands2[(factor_bbands2 == 1) & (input_ts < 0)] = 0
        factor_bbands2[(factor_bbands2 == -1) & (input_ts > 0)] = 0
        factor_bbands2 = factor_bbands2.ffill()
        return factor_bbands2
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_df[col] = bbands_func(series, wind, up, down)
        return result_df

def dema_func(input_ts, wind):
    """
    Calculate Double Exponential Moving Average (DEMA).
    DEMA = 2*EMA(close) - EMA(EMA(close))
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        pd.DataFrame or pd.Series: DEMA values with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        return ta.DEMA(input_ts, timeperiod=wind)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_df[col] = ta.DEMA(series, timeperiod=wind)
        return result_df

def kama_func(input_ts, wind):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA = a*close + (1-a)*KAMA_prev
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        pd.DataFrame or pd.Series: KAMA values with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        return ta.KAMA(input_ts, timeperiod=wind)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_df[col] = ta.KAMA(series, timeperiod=wind)
        return result_df

def htTrendline_func(input_ts):
    """
    Calculate Hilbert Transform Trendline.
    HT = (4*MA + 3*MA_prev + 2*MA_prev2 + MA_prev3)
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        
    Returns:
        pd.DataFrame or pd.Series: Hilbert Transform Trendline values with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        return ta.HT_TRENDLINE(input_ts)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_df[col] = ta.HT_TRENDLINE(series)
        return result_df

def t3_func(input_ts, wind):
    """
    Calculate T3 Moving Average (Triple EMA with smoothing).
    T3 = EMA3 + C*(EMA3-EMA2)
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        pd.DataFrame or pd.Series: T3 Moving Average values with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        return ta.T3(input_ts, wind)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_df[col] = ta.T3(series, wind)
        return result_df

# Momentum Indicators
def apo_func(input_ts, wind):
    """
    Calculate Absolute Price Oscillator (APO).
    Difference between fast and slow EMAs.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Long window (short window will be wind/2)
        
    Returns:
        pd.DataFrame or pd.Series: APO values with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        return ta.APO(input_ts, wind//2, wind, matype=0)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_df[col] = ta.APO(series, wind//2, wind, matype=0)
        return result_df

def cmo_func(input_ts, wind):
    """
    Calculate Chande Momentum Oscillator (CMO).
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        pd.DataFrame or pd.Series: CMO values with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        return ta.CMO(input_ts, wind)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_df[col] = ta.CMO(series, wind)
        return result_df

def macd_func(input_ts, wind):
    """
    Calculate Moving Average Convergence Divergence (MACD) signal line.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Long window (short window will be wind/2)
        
    Returns:  
        pd.DataFrame or pd.Series: MACD signal line with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        macd, macdsignal, macdhist = ta.MACD(input_ts, wind//2, wind, signalperiod=9)
        return macdsignal
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            macd, macdsignal, macdhist = ta.MACD(series, wind//2, wind, signalperiod=9)
            result_df[col] = macdsignal
        return result_df

def mom_func(input_ts, wind):
    """
    Calculate Momentum indicator.
    Current price minus price from N periods ago.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size (lookback period)
        
    Returns:
        pd.DataFrame or pd.Series: Momentum values with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        return ta.MOM(input_ts, wind)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_df[col] = ta.MOM(series, wind)
        return result_df

def rocp_func(input_ts, wind):
    """
    Calculate Rate of Change Percentage (ROCP).
    (Current - Previous)/Previous
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size (lookback period)
        
    Returns:
        pd.DataFrame or pd.Series: ROCP values with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        return ta.ROCP(input_ts, wind)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_df[col] = ta.ROCP(series, wind)
        return result_df

def rsi_func(input_ts, wind):
    """
    Calculate Relative Strength Index (RSI).
    Measures the speed and magnitude of price movements.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        pd.DataFrame or pd.Series: RSI values (0-100) with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        return ta.RSI(input_ts, wind)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            result_df[col] = ta.RSI(series, wind)
        return result_df


#### 2.2 Moving Average Crossover Signals
def MACDsignals_func(short_ma, long_ma):
    """
    Generate trading signals based on moving average crossovers.
    
    Args:
        short_ma (pd.Series): Short-term moving average
        long_ma (pd.Series): Long-term moving average
        
    Returns:
        pd.Series: Trading signals (-1 for sell, 1 for buy)
    """
    # Calculate crossover points
    crossings = pd.Series(np.nan, index=short_ma.index)
    crossings[(short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))] = 1
    crossings[(short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))] = -1
    
    return crossings.ffill()

def double_ma_rsi_signals(close_prices, short_ma, long_ma, rsi_period):
    """
    Generate position signals based on dual moving average crossover with RSI filter.
    
    Args:
        close_prices (pd.Series): Close price series
        short_ma (pd.Series): Short-term moving average
        long_ma (pd.Series): Long-term moving average
        rsi_period (int): RSI calculation period
        
    Returns:
        pd.Series: Position signals (1 for long, -1 for short, 0 for no position)
    """
    # Calculate RSI
    rsi = ta.RSI(close_prices, timeperiod=rsi_period)
    
    # Initialize signal Series
    signals = pd.Series(np.nan, index=close_prices.index)
    
    # Buy signal: short MA crosses above long MA and RSI < 70 (not overbought)
    buy_signals = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)) & (rsi < 70)
    signals[buy_signals] = 1
    
    # Sell signal: short MA crosses below long MA and RSI > 30 (not oversold)
    sell_signals = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1)) & (rsi > 30)
    signals[sell_signals] = -1
    
    # Fill NaN values to maintain continuous signals
    signals = signals.fillna(method='ffill')
    
    return signals

def triple_ma_signals(fast_ma, medium_ma, slow_ma):
    """
    Generate position signals based on triple moving average crossovers.
    
    Args:
        fast_ma (pd.Series): Fast moving average
        medium_ma (pd.Series): Medium-term moving average
        slow_ma (pd.Series): Slow moving average
        
    Returns:
        pd.Series: Position signals (1 for long, -1 for short)
    """
    # Initialize signal Series
    signals = pd.Series(np.nan, index=fast_ma.index)
    
    # Long signal: both fast and medium MAs cross above slow MA
    buy_signals = (fast_ma > slow_ma) & (medium_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1)) & (medium_ma.shift(1) <= slow_ma.shift(1))
    signals[buy_signals] = 1
    
    # Short signal: both fast and medium MAs cross below slow MA
    sell_signals = (fast_ma < slow_ma) & (medium_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1)) & (medium_ma.shift(1) >= slow_ma.shift(1))
    signals[sell_signals] = -1
    
    # Fill NaN values to maintain continuous signals
    signals = signals.fillna(method='ffill')
    
    return signals

def MACDEma_func(input_ts, wind):
    """
    Calculate MACD signals using Exponential Moving Averages.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Long window (short window will be wind/2)
        
    Returns:
        pd.DataFrame or pd.Series: MACD crossover signals with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        short_result = ema_func(input_ts, wind//2)
        long_result = ema_func(input_ts, wind)    
        return MACDsignals_func(short_result, long_result)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            short_result = ema_func(series, wind//2)
            long_result = ema_func(series, wind)
            result_df[col] = MACDsignals_func(short_result, long_result)
        return result_df

def DoubleEma_func(input_ts, wind):
    """
    Calculate double EMA signals with RSI filter.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Long window (short window will be wind/2)
        
    Returns:
        pd.DataFrame or pd.Series: Trading signals with RSI filter with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        short_result = ema_func(input_ts, wind//2)
        long_result = ema_func(input_ts, wind)   
        return double_ma_rsi_signals(input_ts, short_result, long_result, wind)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            short_result = ema_func(series, wind//2)
            long_result = ema_func(series, wind)
            result_df[col] = double_ma_rsi_signals(series, short_result, long_result, wind)
        return result_df

def TripleEma_func(input_ts, wind):
    """
    Calculate triple EMA signals.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Long window (others will be wind/2 and wind/4)
        
    Returns:
        pd.DataFrame or pd.Series: Triple MA crossover signals with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        fast_result = ema_func(input_ts, wind//4)
        medium_result = ema_func(input_ts, wind//2)
        long_result = ema_func(input_ts, wind)   
        return triple_ma_signals(fast_result, medium_result, long_result)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            fast_result = ema_func(series, wind//4)
            medium_result = ema_func(series, wind//2)
            long_result = ema_func(series, wind)
            result_df[col] = triple_ma_signals(fast_result, medium_result, long_result)
        return result_df

def MACDKama_func(input_ts, wind):
    """
    Calculate MACD signals using Kaufman Adaptive Moving Averages.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Long window (short window will be wind/2)
        
    Returns:
        pd.DataFrame or pd.Series: KAMA crossover signals with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        short_result = kama_func(input_ts, wind//2)
        long_result = kama_func(input_ts, wind)    
        return MACDsignals_func(short_result, long_result)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            short_result = kama_func(series, wind//2)
            long_result = kama_func(series, wind)
            result_df[col] = MACDsignals_func(short_result, long_result)
        return result_df

def DoubleKama_func(input_ts, wind):
    """
    Calculate double KAMA signals with RSI filter.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Long window (short window will be wind/2)
        
    Returns:
        pd.DataFrame or pd.Series: KAMA trading signals with RSI filter with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        short_result = kama_func(input_ts, wind//2)
        long_result = kama_func(input_ts, wind)   
        return double_ma_rsi_signals(input_ts, short_result, long_result, wind)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            short_result = kama_func(series, wind//2)
            long_result = kama_func(series, wind)
            result_df[col] = double_ma_rsi_signals(series, short_result, long_result, wind)
        return result_df

def TripleKama_func(input_ts, wind):
    """
    Calculate triple KAMA signals.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Long window (others will be wind/2 and wind/4)
        
    Returns:
        pd.DataFrame or pd.Series: Triple KAMA crossover signals with same shape as input
    """
    if isinstance(input_ts, pd.Series):
        fast_result = kama_func(input_ts, wind//4)
        medium_result = kama_func(input_ts, wind//2)
        long_result = kama_func(input_ts, wind)   
        return triple_ma_signals(fast_result, medium_result, long_result)
    else:
        # 为DataFrame创建相同结构的输出
        result_df = pd.DataFrame(index=input_ts.index)
        # 对每列应用处理
        for col in input_ts.columns:
            series = input_ts[col]
            fast_result = kama_func(series, wind//4)
            medium_result = kama_func(series, wind//2)
            long_result = kama_func(series, wind)
            result_df[col] = triple_ma_signals(fast_result, medium_result, long_result)
        return result_df

#### 3.3 Multi-timeframe Analysis
def termtrend_add(df, window, tfunc):
    """
    Aggregate technical indicators across multiple timeframes by averaging.
    
    Args:
        df (pd.DataFrame or pd.Series): Input data
        window (int): Starting window size
        tfunc (function): Technical indicator function to apply
        
    Returns:
        pd.Series: Average of indicators across timeframes
    """
    result = []
    while window >= 12:
        args = (df, window)
        result.append(globals()[tfunc](*args))  # Apply the indicator function
        window = window // 2
    result = pd.concat(result)
    result = result.groupby(result.index).mean()
    return result

def termtrend_winr(df, window, tfunc):
    """
    Aggregate technical indicators across multiple timeframes, 
    converting to signs (direction) first, then averaging.
    
    Args:
        df (pd.DataFrame or pd.Series): Input data
        window (int): Starting window size
        tfunc (function): Technical indicator function to apply
        
    Returns:
        pd.Series: Average signal direction across timeframes
    """
    result = []
    while window >= 12:
        args = (df, window)
        result.append(globals()[tfunc](*args))
        window = window // 2
    result = [np.sign(x) for x in result]  # Convert to direction signals
    result = pd.concat(result)
    result = result.groupby(result.index).mean()
    return result

def termtrend_plr(df, window, tfunc):
    """
    Positive-to-negative ratio of technical indicators across timeframes.
    
    Args:
        df (pd.DataFrame or pd.Series): Input data
        window (int): Starting window size
        tfunc (function): Technical indicator function to apply
        
    Returns:
        pd.Series: Ratio of positive to negative signals across timeframes
    """
    result = []
    while window >= 12:
        args = (df, window)
        result.append(globals()[tfunc](*args))
        window = window // 2
    result_sign = [np.sign(x) for x in result]

    # Separate positive and negative values
    result_positive = [result[i].mask(result_sign[i] < 0, 0) for i in range(len(result_sign))]
    result_negative = [result[i].mask(result_sign[i] > 0, 0) for i in range(len(result_sign))]

    result_positive = pd.concat(result_positive, axis=0)
    result_negative = pd.concat(result_negative, axis=0)
    result_positive = result_positive.groupby(result_positive.index).sum()
    result_negative = result_negative.groupby(result_negative.index).sum()
    
    # Calculate ratio of positive to negative signals
    result = (result_positive - abs(result_negative)) / (result_positive + abs(result_negative))
    return result

def termtrend_revadd(df, window, tfunc):
    """
    Compare current timeframe signals with historical timeframe signals.
    
    Args:
        df (pd.DataFrame or pd.Series): Input data
        window (int): Starting window size
        tfunc (function): Technical indicator function to apply
        
    Returns:
        pd.Series: Relative comparison between current and historical signals
    """
    window_d = window // (24*12)
    result = []
    while window >= 12:
        result.append(globals()[tfunc](df, window))
        window = window // 2
    result = pd.concat(result)
    result = result.groupby(result.index).mean()
    result_short = result.copy()

    # Calculate historical signals
    window = 24*12*24 + 12*24*window_d
    result = []
    while window >= 12*24*window_d:
        result.append(globals()[tfunc](df.shift(24*12*window_d).ffill().dropna(), window))
        window = window // 2
    result = pd.concat(result)
    result = result.groupby(result.index).mean()
    result_long = result.copy()

    # Compare current to historical
    result = (result_short - result_long) / (abs(result_short) + abs(result_long))
    return result

def termtrend_revwinr(df, window, tfunc):
    """
    Compare current vs historical signal directions across timeframes.
    
    Args:
        df (pd.DataFrame or pd.Series): Input data
        window (int): Starting window size
        tfunc (function): Technical indicator function to apply
        
    Returns:
        pd.Series: Relative comparison of signal directions
    """
    window_d = window // (24*12)
    result = []
    while window >= 12:
        result.append(globals()[tfunc](df, window))
        window = window // 2
    result = [np.sign(x) for x in result]  # Convert to directions
    result = pd.concat(result)
    result = result.groupby(result.index).mean()
    result_short = result.copy()

    # Calculate historical signals
    window = 24*12*24 + 12*24*window_d
    result = []
    while window >= 12*24*window_d:
        result.append(globals()[tfunc](df.shift(24*12*window_d).ffill().dropna(), window))
        window = window // 2
    result = [np.sign(x) for x in result]
    result = pd.concat(result)
    result = result.groupby(result.index).mean()
    result_long = result.copy()

    # Compare current to historical directions
    result = (result_short - result_long) / (abs(result_short) + abs(result_long))
    return result

def termtrend_revplr(df, window, tfunc):
    """
    Compare current vs historical positive-to-negative ratios across timeframes.
    
    Args:
        df (pd.DataFrame or pd.Series): Input data
        window (int): Starting window size
        tfunc (function): Technical indicator function to apply
        
    Returns:
        pd.Series: Relative comparison of positive-to-negative ratios
    """
    window_d = window // (24*12)
    result = []
    while window >= 12:
        result.append(globals()[tfunc](df, window))
        window = window // 2
    result_sign = [np.sign(x) for x in result]

    # Separate positive and negative values (current)
    result_positive = [result[i].mask(result_sign[i] < 0, 0) for i in range(len(result_sign))]
    result_negative = [result[i].mask(result_sign[i] > 0, 0) for i in range(len(result_sign))]

    result_positive = pd.concat(result_positive, axis=0)
    result_negative = pd.concat(result_negative, axis=0)
    result_positive = result_positive.groupby(result_positive.index).sum()
    result_negative = result_negative.groupby(result_negative.index).sum()
    result = (result_positive - abs(result_negative)) / (result_positive + abs(result_negative))
    result_short = result.copy()

    # Calculate historical signals
    window = 24*12*24 + 12*24*window_d
    result = []
    while window >= 12*24*window_d:
        result.append(globals()[tfunc](df.shift(24*12*window_d).ffill().dropna(), window))
        window = window // 2
    result_sign = [np.sign(x) for x in result]

    # Separate positive and negative values (historical)
    result_positive = [result[i].mask(result_sign[i] < 0, 0) for i in range(len(result_sign))]
    result_negative = [result[i].mask(result_sign[i] > 0, 0) for i in range(len(result_sign))]

    result_positive = pd.concat(result_positive, axis=0)
    result_negative = pd.concat(result_negative, axis=0)
    result_positive = result_positive.groupby(result_positive.index).sum()
    result_negative = result_negative.groupby(result_negative.index).sum()
    result = (result_positive - abs(result_negative)) / (result_positive + abs(result_negative))
    result_long = result.copy()

    # Compare current to historical P/L ratios
    result = (result_short - result_long) / (abs(result_short) + abs(result_long))
    return result

#### 4. Statistical Functions
def treg_func(input_ts, wind):
    """
    Calculate Linear Regression value at current price point.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size for regression
        
    Returns:
        Same type as input_ts: Linear regression values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = ta.LINEARREG(input_ts[col], wind)
        return result
    else:
        return ta.LINEARREG(input_ts, wind)

def regag_func(input_ts, wind):
    """
    Calculate Linear Regression Angle.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size for regression
        
    Returns:
        Same type as input_ts: Linear regression angle values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = ta.LINEARREG_ANGLE(input_ts[col], wind)
        return result
    else:
        return ta.LINEARREG_ANGLE(input_ts, wind)

def itc_func(input_ts, wind):
    """
    Calculate Linear Regression Intercept.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size for regression
        
    Returns:
        Same type as input_ts: Linear regression intercept values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = ta.LINEARREG_INTERCEPT(input_ts[col], wind)
        return result
    else:
        return ta.LINEARREG_INTERCEPT(input_ts, wind)

def slope_func(input_ts, wind):
    """
    Calculate Linear Regression Slope.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size for regression
        
    Returns:
        Same type as input_ts: Linear regression slope values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = ta.LINEARREG_SLOPE(input_ts[col], wind)
        return result
    else:
        return ta.LINEARREG_SLOPE(input_ts, wind)

def std_func(input_ts, wind):
    """
    Calculate Standard Deviation over specified window.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        Same type as input_ts: Standard deviation values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = ta.STDDEV(input_ts[col], wind)
        return result
    else:
        return ta.STDDEV(input_ts, wind)

def tsf_func(input_ts, wind):
    """
    Calculate Time Series Forecast.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        Same type as input_ts: Time series forecast values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = ta.TSF(input_ts[col], wind)
        return result
    else:
        return ta.TSF(input_ts, wind)

def var_func(input_ts, wind):
    """
    Calculate Variance over specified window.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        Same type as input_ts: Variance values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = ta.VAR(input_ts[col], wind)
        return result
    else:
        return ta.VAR(input_ts, wind)

def beta_func(input_ts, rtn, wind):
    """
    Calculate Beta (correlation) between two series.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        rtn (pd.Series): Reference series (typically market return)
        wind (int): Window size
        
    Returns:
        Same type as input_ts: Beta values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = ta.BETA(input_ts[col], rtn, wind)
        return result
    else:
        return ta.BETA(input_ts, rtn, wind)

def diff_func(input_ts, wind):
    """
    Calculate difference over specified lag.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Lag periods
        
    Returns:
        Same type as input_ts: Difference values (current - lagged)
    """
    if isinstance(input_ts, pd.DataFrame):
        return input_ts.diff(wind)
    else:
        return input_ts.diff(wind)

def DoubleDiff_func(input_ts, wind):
    """
    Calculate second-order difference (difference of differences).
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Lag periods
        
    Returns:
        Same type as input_ts: Second-order difference values
    """
    if isinstance(input_ts, pd.DataFrame):
        return input_ts.diff(wind).diff(wind)
    else:
        return input_ts.diff(wind).diff(wind)

def mean_std_func(input_ts, wind):
    """
    Calculate ratio of mean to standard deviation (signal-to-noise ratio).
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        Same type as input_ts: Mean/Std ratio values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = input_ts[col].rolling(wind, min_periods=1).mean() / input_ts[col].rolling(wind, min_periods=1).std()
        return result
    else:
        return input_ts.rolling(wind, min_periods=1).mean() / input_ts.rolling(wind, min_periods=1).std()

def xMeanStd_func(input_ts, wind):
    """
    Calculate Z-score: (value - mean) / standard deviation.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        Same type as input_ts: Z-score values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = (input_ts[col] - input_ts[col].rolling(wind, min_periods=1).mean()) / input_ts[col].rolling(wind, min_periods=1).std()
        return result
    else:
        return (input_ts - input_ts.rolling(wind, min_periods=1).mean()) / input_ts.rolling(wind, min_periods=1).std()

def skew_func(input_ts, wind):
    """
    Calculate skewness over specified window.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        Same type as input_ts: Skewness values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = input_ts[col].rolling(wind, min_periods=1).skew()
        return result
    else:
        return input_ts.rolling(wind, min_periods=1).skew()

def kurt_func(input_ts, wind):
    """
    Calculate kurtosis over specified window.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        Same type as input_ts: Kurtosis values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = input_ts[col].rolling(wind, min_periods=1).kurt()
        return result
    else:
        return input_ts.rolling(wind, min_periods=1).kurt()

def pct_func(input_ts, wind):
    """
    Calculate percentage change over specified lag.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Lag periods
        
    Returns:
        Same type as input_ts: Percentage change values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            result[col] = ((input_ts[col] - input_ts[col].shift(wind)) / abs(input_ts[col].shift(wind))).replace([np.inf, -np.inf], np.nan).ffill()
        return result
    else:
        return ((input_ts - input_ts.shift(wind)) / abs(input_ts.shift(wind))).replace([np.inf, -np.inf], np.nan).ffill()

def entropy_func(input_ts, wind):
    """
    Calculate Shannon entropy over specified window.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        wind (int): Window size
        
    Returns:
        Same type as input_ts: Entropy values
    """
    if isinstance(input_ts, pd.DataFrame):
        result = pd.DataFrame(index=input_ts.index)
        for col in input_ts.columns:
            # Calculate entropy for each window
            result[col] = input_ts[col].rolling(window=wind).apply(entropy)
        return result
    else:
        # Calculate entropy for each window
        return input_ts.rolling(window=wind).apply(entropy)


#### 5. Factor Interaction Functions
def filtDisH_func(input_ts, filt_fac):
    """
    Filter and weight time series by another factor.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series
        filt_fac (pd.Series): Filter factor
        
    Returns:
        pd.Series: Filtered and weighted average across timeframes
    """
    # Remove filtered parts with low filter factor values, multiply, and use momentum expression to reduce turnover
    if isinstance(input_ts, pd.DataFrame):
        input_ts = input_ts.iloc[:, 0]
    # Scale to [0,1]
    input_ts = DataPrepare.normalization(input_ts, clip_thres=None, scale_window='40d', sp='5min', scale_quantile=0.005) / 2 + 0.5
    factor_Higher = input_ts.mask(filt_fac < 0.3, np.nan)
    factor = factor_Higher * filt_fac
    
    # Average across multiple timeframes
    window = 24 * 12
    result = []
    while window >= 6:
        result.append(factor.rolling(window).mean())
        window = window // 2
    result = pd.concat(result)
    result = result.groupby(result.index).mean()
    return result

def RegTw_func(input_ts, y_ts):
    """
    Calculate rolling regression coefficient (beta) between input and target.
    
    Args:
        input_ts (pd.DataFrame or pd.Series): Input time series (factor)
        y_ts (pd.Series): Target time series (typically price/return)
        
    Returns:
        pd.Series: Rolling regression coefficient values
    """
    if isinstance(input_ts, pd.DataFrame):
        input_ts = input_ts.iloc[:, 0]
    # Normalize input
    input_ts = DataPrepare.normalization(input_ts, clip_thres=None, scale_window='40d', sp='5min', scale_quantile=0.005).dropna()
    
    # Initialize output Series
    output_df = pd.Series(index=input_ts.index)
    input_ts = input_ts.reindex(index=y_ts.index).fillna(0)
                
    window_size = 40 * 24 * 12  # Look back 40 days for regression
    
    # Calculate rolling regression coefficient
    for i in range(window_size, len(input_ts)):
        print(i)
        y_window = y_ts.iloc[i-window_size:i].values.reshape(-1, 1)
        x_window = input_ts.iloc[i-window_size:i].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_window, y_window)
        beta = model.coef_[0, 0]
        output_df.loc[input_ts.index[i]] = beta

    return output_df
