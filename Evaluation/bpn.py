#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算隐写容量：每个碱基可以隐藏多少个0/1比特
批量处理整个目录中的所有文件对
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def count_bases_in_sequence(sequence_file):
    """
    计算DNA序列中的碱基总数
    
    Args:
        sequence_file: DNA序列文件路径
        
    Returns:
        int: 碱基总数
    """
    total_bases = 0
    
    with open(sequence_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                # 移除空格，只计算ATCG碱基
                bases = line.replace(' ', '')
                # 只计算有效的DNA碱基
                valid_bases = sum(1 for char in bases if char.upper() in 'ATCG')
                total_bases += valid_bases
    
    return total_bases

def count_bits_in_file(bits_file):
    """
    计算比特文件中的比特总数
    
    Args:
        bits_file: 比特文件路径
        
    Returns:
        int: 比特总数
    """
    total_bits = 0
    
    with open(bits_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                # 计算0和1的数量
                bits = sum(1 for char in line if char in '01')
                total_bits += bits
    
    return total_bits

def calculate_hiding_capacity(sequence_file, bits_file):
    """
    计算隐写容量

    Args:
        sequence_file: DNA序列文件路径
        bits_file: 隐藏比特文件路径

    Returns:
        tuple: (总碱基数, 总比特数, 每个碱基的隐藏容量)
    """
    # 计算总碱基数
    total_bases = count_bases_in_sequence(sequence_file)

    # 计算总比特数
    total_bits = count_bits_in_file(bits_file)

    # 计算每个碱基的隐藏容量
    if total_bases > 0:
        capacity_per_base = total_bits / total_bases
    else:
        capacity_per_base = 0

    return total_bases, total_bits, capacity_per_base

def find_file_pairs(stego_dir, bits_dir):
    """
    查找对应的隐写序列文件和比特文件对

    Args:
        stego_dir: 隐写序列目录
        bits_dir: 比特文件目录

    Returns:
        list: 文件对列表 [(sequence_file, bits_file, base_name, method), ...]
    """
    file_pairs = []

    # 遍历隐写序列目录
    for stego_file in os.listdir(stego_dir):
        if stego_file.endswith('.txt'):
            stego_path = os.path.join(stego_dir, stego_file)
            base_name = stego_file.replace('.txt', '')

            # 根据不同的方法构造对应的比特文件名
            bits_path = None
            method = ""

            # 1. fcpss 方法 (binary, differential, stability)
            if base_name.startswith('fcpssbinary_'):
                bits_name = base_name.replace('fcpssbinary_', 'fcpssbits_') + '_binary.txt'
                method = "fcpss_binary"
            elif base_name.startswith('fcpssdifferential_'):
                bits_name = base_name.replace('fcpssdifferential_', 'fcpssbits_') + '_differential.txt'
                method = "fcpss_differential"
            elif base_name.startswith('fcpssstability_'):
                bits_name = base_name.replace('fcpssstability_', 'fcpssbits_') + '_stability.txt'
                method = "fcpss_stability"

            # 2. fullfcpss 方法 (binary, differential, stability)
            elif base_name.startswith('fullfcpssbinary_'):
                bits_name = base_name.replace('fullfcpssbinary_', 'fullfcpssbits_') + '_binary.txt'
                method = "fullfcpss_binary"
            elif base_name.startswith('fullfcpssdifferential_'):
                bits_name = base_name.replace('fullfcpssdifferential_', 'fullfcpssbits_') + '_differential.txt'
                method = "fullfcpss_differential"
            elif base_name.startswith('fullfcpssstability_'):
                bits_name = base_name.replace('fullfcpssstability_', 'fullfcpssbits_') + '_stability.txt'
                method = "fullfcpss_stability"

            # 3. xlstmadg 方法
            elif base_name.startswith('xlstmadg_'):
                bits_name = base_name.replace('xlstmadg_', 'xlstmadgbits_') + '.txt'
                method = "xlstmadg"

            # 4. slstmfcpss 方法 (binary, differential, stability)
            # elif base_name.startswith('slstmfcpssbinary_'):
            #     bits_name = base_name.replace('slstmfcpssbinary_', 'slstmfcpssbits_') + '_binary.txt'
            #     method = "slstmfcpss_binary"
            elif base_name.startswith('slstmfcpssdifferential_'):
                bits_name = base_name.replace('slstmfcpssdifferential_', 'slstmfcpssbits_') + '_differential.txt'
                method = "slstmfcpss_differential"
            # elif base_name.startswith('slstmfcpssstability_'):
            #     bits_name = base_name.replace('slstmfcpssstability_', 'slstmfcpssbits_') + '_stability.txt'
            #     method = "slstmfcpss_stability"

            else:
                continue  # 跳过不识别的文件

            bits_path = os.path.join(bits_dir, bits_name)

            # 检查对应的比特文件是否存在
            if os.path.exists(bits_path):
                file_pairs.append((stego_path, bits_path, base_name, method))

    # 4. 处理 sparsamp 方法（这些文件没有对应的序列文件，需要特殊处理）
    # 先跳过 sparsamp，因为它们的命名规则不同

    return file_pairs

def process_genome(genome_name):
    """处理单个基因组的所有文件对并计算隐写容量"""
    # 设置目录路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
    stego_dir = os.path.join(base_dir, "Stego_DNA", genome_name)
    bits_dir = os.path.join(base_dir, "bits", genome_name)

    # 创建结果目录 - 保持原有的result目录结构
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 检查目录是否存在
    if not os.path.exists(stego_dir):
        print(f"警告：隐写序列目录不存在: {stego_dir}")
        return []

    if not os.path.exists(bits_dir):
        print(f"警告：比特文件目录不存在: {bits_dir}")
        return []

    # 查找文件对
    file_pairs = find_file_pairs(stego_dir, bits_dir)

    if not file_pairs:
        print(f"警告：{genome_name} 未找到匹配的文件对")
        return []

    print("=" * 80)
    print(f"处理基因组: {genome_name}")
    print("=" * 80)
    print(f"隐写序列目录: {stego_dir}")
    print(f"比特文件目录: {bits_dir}")
    print(f"找到 {len(file_pairs)} 个文件对")
    print("-" * 80)

    # 存储所有结果
    all_results = []

    # 处理每个文件对
    for i, (sequence_file, bits_file, base_name, method) in enumerate(file_pairs, 1):
        print(f"[{i}/{len(file_pairs)}] 处理: {base_name} ({method})")

        try:
            # 计算隐写容量
            total_bases, total_bits, capacity_per_base = calculate_hiding_capacity(sequence_file, bits_file)

            # 存储结果
            result = {
                'sequence_file': os.path.basename(sequence_file),
                'bits_file': os.path.basename(bits_file),
                'total_bases': total_bases,
                'total_bits': total_bits,
                'bpn': capacity_per_base
            }
            all_results.append(result)

            print(f"  碱基数: {total_bases:,}, 比特数: {total_bits:,}, BPN: {capacity_per_base:.6f}")

        except Exception as e:
            print(f"  错误: {str(e)}")
            continue

    # 保存结果
    if all_results:
        # 保存详细结果并获取DataFrame
        df = save_results_for_genome(all_results, result_dir, genome_name)

        # 生成统计文件
        generate_stats_for_genome(df, result_dir, genome_name)

        # 显示摘要
        show_summary_for_genome(all_results, genome_name)

    return all_results

def process_all_genomes():
    """处理所有基因组"""
    # 基因组列表 - 先测试一个基因组
    # genomes = ["ASM141792v1"]
    genomes = ["ASM141792v1", "ASM286374v1", "ASM400647v1", "ASM949793v1"]

    print("开始处理所有基因组的隐写容量计算...")
    print(f"共需处理 {len(genomes)} 个基因组")

    all_genome_results = {}

    for genome in genomes:
        print(f"\n{'='*100}")
        print(f"开始处理基因组: {genome}")
        print(f"{'='*100}")

        results = process_genome(genome)
        all_genome_results[genome] = results

    # 生成总体统计
    print(f"\n{'='*100}")
    print("所有基因组处理完成 - 总体统计")
    print(f"{'='*100}")

    for genome, results in all_genome_results.items():
        if results:
            bpn_values = [r['bpn'] for r in results]
            total_bases = sum(r['total_bases'] for r in results)
            total_bits = sum(r['total_bits'] for r in results)
            print(f"{genome}: {len(results)} 个文件对, 平均BPN: {sum(bpn_values)/len(bpn_values):.6f}, "
                  f"总碱基: {total_bases:,}, 总比特: {total_bits:,}")
        else:
            print(f"{genome}: 无有效数据")

    print(f"{'='*100}")

def save_results_for_genome(results, result_dir, genome_name):
    """为单个基因组保存结果到文件"""
    # 修改命名风格，与collected_results.py期望的格式一致
    csv_file = os.path.join(result_dir, f"bpn_results_{genome_name}.csv")

    # 创建DataFrame用于更好的数据处理
    df_data = []
    for result in results:
        # 从sequence_file中提取method名称（去掉.txt后缀）
        method_name = result['sequence_file'].replace('.txt', '')
        efficiency = (result['bpn'] / 2) * 100

        df_data.append({
            'method': method_name,
            'sequence_file': result['sequence_file'],
            'bits_file': result['bits_file'],
            'total_bases': result['total_bases'],
            'total_bits': result['total_bits'],
            'bpn': result['bpn'],
            'efficiency_percent': efficiency
        })

    # 保存详细结果
    df = pd.DataFrame(df_data)
    # 按照BPN从大到小排序
    df = df.sort_values('bpn', ascending=False)
    df.to_csv(csv_file, index=False)
    print(f"\n{genome_name} 详细结果已保存到: {csv_file} (按BPN降序排列)")

    return df

def generate_stats_for_genome(df, result_dir, genome_name):
    """生成统计文件，类似gcd_tmd_klp.py的逻辑"""
    # 统计文件路径，采用与gcd_tmd_klp.py相同的命名风格
    stats_file = os.path.join(result_dir, f"{genome_name}_bpn_stats.csv")

    print(f"\n生成BPN统计结果：{stats_file}")

    # 提取基础名称和参数（类似gcd_tmd_klp.py的逻辑）
    df['base_name'] = df['method'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    # 从base_name中提取序列长度和分割参数
    def extract_params(name):
        parts = name.split('_')
        if len(parts) >= 3:
            length = parts[-2]  # 倒数第二个部分通常是长度
            split = parts[-1]   # 最后一个部分通常是分割参数
            return f"{length}_{split}"
        return name

    df['params'] = df['base_name'].apply(extract_params)

    # 按照参数和方法名排序生成统计
    stats = []
    # 首先获取所有唯一的参数组合
    param_groups = sorted(df['params'].unique())

    for param in param_groups:
        # 对于每个参数组合，获取所有方法
        param_data = df[df['params'] == param]
        for name in sorted(param_data['base_name'].unique()):
            group = df[df['base_name'] == name]
            if len(group) > 0:
                mean_bpn = group['bpn'].mean()
                stats_dict = {
                    'method': name,
                    'BPN': f"{mean_bpn:.6f}±{group['bpn'].std():.6f}",
                    'efficiency': f"{group['efficiency_percent'].mean():.2f}±{group['efficiency_percent'].std():.2f}",
                    'count': len(group),
                    'mean_bpn_sort': mean_bpn  # 用于排序的数值列
                }
                stats.append(stats_dict)

    # 保存统计结果
    stats_df = pd.DataFrame(stats)
    # 按照平均BPN从大到小排序
    stats_df = stats_df.sort_values('mean_bpn_sort', ascending=False)
    # 删除用于排序的辅助列
    stats_df = stats_df.drop('mean_bpn_sort', axis=1)
    stats_df.to_csv(stats_file, index=False)
    print(f"✅ BPN统计结果已保存到: {stats_file} (按平均BPN降序排列)")

    return stats_df

def show_summary_for_genome(results, genome_name):
    """显示单个基因组的统计摘要"""
    if not results:
        return

    bpn_values = [r['bpn'] for r in results]
    total_bases_sum = sum(r['total_bases'] for r in results)
    total_bits_sum = sum(r['total_bits'] for r in results)

    print(f"\n{'-'*60}")
    print(f"{genome_name} 统计摘要")
    print(f"{'-'*60}")
    print(f"处理文件数: {len(results)}")
    print(f"总碱基数: {total_bases_sum:,}")
    print(f"总比特数: {total_bits_sum:,}")
    print(f"平均BPN: {sum(bpn_values)/len(bpn_values):.6f} 比特/碱基")
    print(f"最小BPN: {min(bpn_values):.6f} 比特/碱基")
    print(f"最大BPN: {max(bpn_values):.6f} 比特/碱基")
    print(f"整体BPN: {total_bits_sum/total_bases_sum:.6f} 比特/碱基")
    print(f"{'-'*60}")

def main():
    """主函数"""
    process_all_genomes()

if __name__ == "__main__":
    main()
