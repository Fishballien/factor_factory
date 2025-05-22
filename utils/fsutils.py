# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:28:13 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import shutil
from pathlib import Path
from typing import Union
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# %%
def copy_file(source_path: Union[str, Path], target_path: Union[str, Path], overwrite: bool = True) -> bool:
    """
    复制文件从源路径到目标路径

    参数:
        source_path (str or Path): 源文件路径
        target_path (str or Path): 目标文件路径
        overwrite (bool, optional): 是否覆盖已存在的文件，默认为True

    返回:
        bool: 复制成功返回True，否则返回False
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    # 检查源文件是否存在
    if not source_path.exists() or not source_path.is_file():
        print(f"错误: 源文件 {source_path} 不存在或不是一个文件")
        return False
    
    # 检查目标目录是否存在，不存在则创建
    target_dir = target_path.parent
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查目标文件是否已存在
    if target_path.exists() and not overwrite:
        print(f"警告: 目标文件 {target_path} 已存在且不允许覆盖")
        return False
    
    try:
        shutil.copy2(source_path, target_path)
        return True
    except Exception as e:
        print(f"复制文件时出错: {e}")
        return False
    
    
def parallel_copy_files(file_pairs, max_workers=8):
    """
    使用多线程并行复制文件
    
    参数:
        file_pairs: 包含(源文件路径, 目标文件路径)元组的列表
        max_workers: 最大线程数，默认为8
        
    返回:
        tuple: (成功复制的文件数, 总耗时)
    """
    
    if not file_pairs:
        print("没有找到需要复制的文件。")
        return 0, 0
        
    print(f"开始并行复制 {len(file_pairs)} 个文件...")
    start_time = time.time()
    
    # 使用ThreadPoolExecutor进行并行复制
    with ThreadPoolExecutor(max_workers=min(len(file_pairs), max_workers)) as executor:
        # 为每个文件对提交复制任务
        future_to_file = {
            executor.submit(copy_file, src, dst): (src, dst) 
            for src, dst in file_pairs
        }
        
        # 收集结果
        success_count = 0
        for future in as_completed(future_to_file):
            src, dst = future_to_file[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                    print(f"成功复制: {src.name}")
                else:
                    print(f"复制失败: {src.name}")
            except Exception as e:
                print(f"复制 {src.name} 时发生异常: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"复制完成! 成功: {success_count}/{len(file_pairs)}, 用时: {duration:.2f} 秒")
    
    return success_count, duration