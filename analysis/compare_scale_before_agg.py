# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:46:44 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd
from pathlib import Path


# %%
import re

def contains_imb_wavg_pattern(input_string):
    """
    判断字符串是否包含符合 "imb*wavg" 模式的子字符串
    
    参数:
        input_string (str): 要检查的输入字符串
    
    返回:
        bool: 如果找到匹配则返回 True，否则返回 False
    
    示例:
        >>> contains_imb_wavg_pattern("这是一个imb01_wavg测试")
        True
        >>> contains_imb_wavg_pattern("imb01_iweuoiwuhfhjowe_wavg")
        True
        >>> contains_imb_wavg_pattern("imbwavg")
        True
        >>> contains_imb_wavg_pattern("imb_wavg")
        True
        >>> contains_imb_wavg_pattern("没有匹配")
        False
    """
    # 正则表达式模式: imb 开头，wavg 结尾，中间可以是任意字符
    pattern = r'imb.*wavg'
    
    # 使用 re.search 在字符串中查找匹配项
    match = re.search(pattern, input_string)
    
    # 如果找到匹配项，返回 True，否则返回 False
    return match is not None


# %%
eval_path = 'D:/mnt/CNIndexFutures/timeseries/factor_test/results/factor_evaluation/agg_batch10_scale_before_agg_or_not/factor_eval_160101_250326.csv'


# %%
eval_data = pd.read_csv(eval_path)


# %%
eval_data_selected = eval_data[eval_data['factor'].apply(lambda x: contains_imb_wavg_pattern(x) and 'imb06' not in x)]


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

# Assuming eval_data_selected is already loaded
# If not, you would load it here with:
# eval_data_selected = pd.read_csv('your_data_file.csv')

def plot_sharpe_ratio_histograms(data, process_names=None, figsize=(14, 8), bins=20):
    """
    Plot histograms of net_sharpe_ratio for selected process_names on a single figure.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataframe containing the data
    process_names : list or None
        List of process_names to plot. If None, all unique process_names will be used.
    figsize : tuple
        Figure size (width, height)
    bins : int
        Number of bins for the histogram
    """
    # Get unique process names if not specified
    if process_names is None:
        process_names = sorted(data['process_name'].unique().tolist())
    
    # Cute pastel colors
    cute_colors = [
        '#FF9AA2',  # pastel red
        # '#FFB7B2',  # pastel salmon
        # '#FFDAC1',  # pastel orange
        # '#E2F0CB',  # pastel green
        # '#B5EAD7',  # pastel mint
        '#C7CEEA',  # pastel blue
        '#D4A5A5',  # pastel rose
        '#9DC8C8',  # pastel teal
        '#FFFFB5',  # pastel yellow
        '#D8B9C3',  # pastel purple
        '#FCB9AA',  # pastel coral
        '#BEE4D2',  # pastel aqua
        '#FCBAD3',  # pastel pink
        '#A2D2FF',  # pastel sky blue
        '#F6BD60',  # pastel amber
    ]
    
    # Set figure style
    rcParams['font.family'] = 'DejaVu Sans'
    rcParams['font.size'] = 12
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get the range of sharpe ratios to use consistent bins across all histograms
    min_sharpe = data['net_sharpe_ratio'].min()
    max_sharpe = data['net_sharpe_ratio'].max()
    bin_edges = np.linspace(min_sharpe, max_sharpe, bins+1)
    
    # Loop through each process name and plot histogram
    for i, process_name in enumerate(process_names):
        # Get data for this process
        process_data = data[data['process_name'] == process_name]['net_sharpe_ratio']
        
        if len(process_data) == 0:
            continue
        
        # Plot histogram with cute colors (outline only)
        color_idx = i % len(cute_colors)
        histtype = 'step'  # 'step' draws unfilled histograms with lines
        
        ax.hist(
            process_data,
            bins=bin_edges,
            color=cute_colors[color_idx],
            edgecolor=cute_colors[color_idx],
            histtype=histtype,
            linewidth=2,
            alpha=1,
            label=f"{process_name} (n={len(process_data)})"
        )
    
    # Customize plot
    ax.set_title('Histogram of Net Sharpe Ratio by Process Name', fontsize=16)
    ax.set_xlabel('Net Sharpe Ratio', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Process Name', fontsize=10, title_fontsize=12)
    
    # Make x-axis integers if appropriate
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Function to create an interactive widget for selecting process names
def create_process_selector(data):
    """
    Create a widget to select which process_names to visualize.
    This function simply lists available process names and provides usage instructions.
    """
    unique_processes = sorted(data['process_name'].unique().tolist())
    print(f"Found {len(unique_processes)} unique process_names:")
    for i, process in enumerate(unique_processes):
        print(f"{i+1}. {process}")
    print("\nTo plot specific processes, call the function with:")
    print("selected_processes = [process_name1, process_name2, ...]")
    print("fig = plot_sharpe_ratio_histograms(eval_data_selected, selected_processes)")
    return unique_processes

# Example usage
if __name__ == "__main__":
    # List all unique process names
    process_names = create_process_selector(eval_data_selected)
    
    # Example: Plot histograms for first 5 process names
    selected_processes = process_names[:5]
    fig = plot_sharpe_ratio_histograms(eval_data_selected, selected_processes)
    plt.show()
    
    # Alternatively, plot all process names
    # fig = plot_sharpe_ratio_histograms(eval_data_selected)
    # plt.show()