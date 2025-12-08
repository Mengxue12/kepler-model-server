#!/usr/bin/env python3
"""
脚本用于读取指定文件夹下的所有CSV文件，并打印出它们的路径名和行数。
"""

import os
import argparse
from pathlib import Path


def count_csv_lines(csv_path):
    """计算CSV文件的行数（不包括表头）"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # 使用enumerate来计数，从1开始（包括表头）
            line_count = sum(1 for _ in f)
            # 如果文件不为空，减去表头行
            if line_count > 0:
                return line_count - 1  # 减去表头
            return 0
    except Exception as e:
        print(f"读取文件 {csv_path} 时出错: {e}")
        return -1


def find_and_count_csv_files(directory):
    """查找目录下所有CSV文件并统计行数"""
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"错误: 目录 '{directory}' 不存在")
        return
    
    if not directory_path.is_dir():
        print(f"错误: '{directory}' 不是一个目录")
        return
    
    # 查找所有CSV文件
    csv_files = list(directory_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"在目录 '{directory}' 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件:\n")
    print(f"{'路径':<80} {'行数':>10}")
    print("-" * 95)
    
    total_lines = 0
    for csv_file in sorted(csv_files):
        # 获取相对路径或绝对路径
        file_path = csv_file.absolute()
        line_count = count_csv_lines(file_path)
        
        if line_count >= 0:
            print(f"{str(file_path):<80} {line_count:>10}")
            total_lines += line_count
        else:
            print(f"{str(file_path):<80} {'错误':>10}")
    
    print("-" * 95)
    print(f"{'总计':<80} {total_lines:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="读取文件夹下的所有CSV文件并打印路径名和行数"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="要扫描的文件夹路径"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="递归搜索子目录（默认已启用）"
    )
    
    args = parser.parse_args()
    
    find_and_count_csv_files(args.directory)


if __name__ == "__main__":
    main()

