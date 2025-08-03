import os
import ViennaRNA as RNA
import pandas as pd

# 计算DNA二级结构自由能
def calculate_dna_delta_g(sequence):
    if len(sequence) == 0:
        raise ValueError("序列不能为空")
    md = RNA.md()
    md.rna = False  # 指定为DNA
    fc = RNA.fold_compound(sequence, md)
    structure, mfe = fc.mfe()  # mfe 为最低自由能 ΔG
    return mfe

# 读取并合并DNA序列
def read_and_merge_dna(file_path):
    with open(file_path, 'r') as file:
        dna_sequences = file.readlines()
    merged_sequence = ''.join(seq.strip().replace(' ', '') for seq in dna_sequences)
    return merged_sequence

# 分割长序列为固定长度
def split_sequence(sequence, chunk_size=1000):
    return [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]

# 计算序列的平均ΔG和统计信息
def calculate_sequence_stats(chunks):
    """计算序列片段的ΔG统计信息"""
    delta_g_values = []
    for chunk in chunks:
        try:
            dg = calculate_dna_delta_g(chunk)
            delta_g_values.append(dg)
        except Exception as e:
            print(f"计算片段ΔG时出错: {e}")
            continue

    if not delta_g_values:
        return None, None, None, None

    import numpy as np
    mean_dg = np.mean(delta_g_values)
    std_dg = np.std(delta_g_values)
    min_dg = np.min(delta_g_values)
    max_dg = np.max(delta_g_values)

    return mean_dg, std_dg, min_dg, max_dg

# 主程序 - 改进版本
def main(natural_file, stego_file):
    # 加载天然DNA和隐写DNA序列
    natural_sequence = read_and_merge_dna(natural_file)
    stego_sequence = read_and_merge_dna(stego_file)

    # 分割长序列为固定长度片段
    chunk_size = 100  # 可以调整为50, 100, 200等
    natural_chunks = split_sequence(natural_sequence, chunk_size=chunk_size)
    stego_chunks = split_sequence(stego_sequence, chunk_size=chunk_size)
    # print('natural_chunks:', natural_chunks[0])
    # print('stego_chunks:', len(stego_chunks[0]))
    # 计算天然序列的ΔG统计信息
    nat_mean, nat_std, _, _ = calculate_sequence_stats(natural_chunks)

    # 计算隐写序列的ΔG统计信息
    stego_mean, stego_std, _, _ = calculate_sequence_stats(stego_chunks)

    if nat_mean is None or stego_mean is None:
        raise ValueError("无法计算序列的ΔG统计信息")

    # 计算关键差异指标
    mean_diff = abs(nat_mean - stego_mean)  # 平均ΔG差异
    relative_mean_diff = abs(nat_mean - stego_mean) / abs(nat_mean) * 100 if nat_mean != 0 else 0

    return {
        'natural_dg': f"{nat_mean:.3f}±{nat_std:.3f}",
        'stego_dg': f"{stego_mean:.3f}±{stego_std:.3f}",
        'mean_dg_diff': round(mean_diff, 3),
        'relative_diff_percent': round(relative_mean_diff, 2)
    }

def get_processed_files(csv_file):
    """获取已经处理过的文件列表"""
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            # 检查是否有表头，如果没有则手动指定列名
            if 'stego_file' not in df.columns:
                # 根据代码中的result_row结构，手动指定列名
                expected_columns = ['base_name', 'natural_file', 'stego_file', 'natural_dg', 'stego_dg', 'mean_dg_diff', 'relative_diff_percent']
                if len(df.columns) == len(expected_columns):
                    df.columns = expected_columns
                else:
                    print(f"警告: CSV文件列数不匹配，期望{len(expected_columns)}列，实际{len(df.columns)}列")
                    return set()
            return set(df['stego_file'].values)
        except Exception as e:
            print(f"读取CSV文件时出错: {e}")
            return set()
    return set()

if __name__ == '__main__':
    # 文件基础信息
    files_name = ['ASM286374v1']
    # files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1']

    # 根路径设置
    root_path = '/home/fan/Code4idea/xLSTMstega'
    results_dir = os.path.join(root_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'free_energy'), exist_ok=True)

    for base_name in files_name:
        # 结果文件路径
        output_file = os.path.join(results_dir, 'free_energy', f'{base_name}_dna_energy_analysis.csv')

        # 获取已处理的文件列表
        processed_files = get_processed_files(output_file)
        print(f"\n已处理的文件数量：{len(processed_files)}")

        print(f"\n正在处理：{base_name}")
        base_dir = os.path.join(root_path, 'Dataset')

        # 固定使用一个天然DNA文件
        natural_file = os.path.join(base_dir, f'{base_name}_198_3.txt')

        if not os.path.exists(natural_file):
            print(f"天然DNA文件不存在，跳过：{natural_file}")
            continue

        print(f"\n处理天然DNA文件：{natural_file}")

        # 预先计算天然DNA序列的ΔG统计信息（只计算一次）
        print("计算天然DNA序列的ΔG统计信息...")
        natural_sequence = read_and_merge_dna(natural_file)
        chunk_size = 100
        natural_chunks = split_sequence(natural_sequence, chunk_size=chunk_size)
        nat_mean, nat_std, _, _ = calculate_sequence_stats(natural_chunks)

        if nat_mean is None:
            print(f"无法计算天然序列的ΔG统计信息，跳过：{natural_file}")
            continue

        print(f"天然序列ΔG: {nat_mean:.3f}±{nat_std:.3f}")

        # 隐写DNA序列路径列表
        stego_dirs = [
            # os.path.join(root_path, 'Stego_DNA/Baselines', base_name),
            os.path.join(root_path, 'Stego_DNA', base_name)
        ]

        # 遍历每个隐写目录
        for stego_dir in stego_dirs:
            if not os.path.exists(stego_dir):
                print(f"路径不存在，跳过：{stego_dir}")
                continue

            print(f"处理隐写目录：{stego_dir}")

            # 遍历目录下的隐写文件
            for stego_file in os.listdir(stego_dir):
                if stego_file.endswith('.txt'):
                    stego_file_path = os.path.join(stego_dir, stego_file)
                    filename = stego_file[:-4]  # 去掉.txt后缀

                    # 检查是否已处理
                    if stego_file in processed_files:
                        print(f"文件已处理，跳过：{stego_file}")
                        continue

                    print(f"处理新文件：{stego_file}")

                    try:
                        # 只计算隐写序列的ΔG统计信息
                        stego_sequence = read_and_merge_dna(stego_file_path)
                        stego_chunks = split_sequence(stego_sequence, chunk_size=chunk_size)
                        stego_mean, stego_std, _, _ = calculate_sequence_stats(stego_chunks)

                        if stego_mean is None:
                            print(f"无法计算隐写序列的ΔG统计信息，跳过：{stego_file}")
                            continue

                        # 计算差异指标
                        mean_diff = abs(nat_mean - stego_mean)
                        relative_mean_diff = abs(nat_mean - stego_mean) / abs(nat_mean) * 100 if nat_mean != 0 else 0

                        # 准备结果字典
                        result_dict = {
                            'natural_dg': f"{nat_mean:.3f}±{nat_std:.3f}",
                            'stego_dg': f"{stego_mean:.3f}±{stego_std:.3f}",
                            'mean_dg_diff': round(mean_diff, 3),
                            'relative_diff_percent': round(relative_mean_diff, 2)
                        }

                        # 准备当前结果
                        result_row = {
                            "base_name": base_name,
                            "natural_file": os.path.basename(natural_file),
                            "stego_file": stego_file,
                            **result_dict  # 展开所有统计结果
                        }

                        # 实时写入CSV文件
                        df_row = pd.DataFrame([result_row])

                        # 如果是第一次写入，包含表头；否则追加模式
                        if not os.path.exists(output_file):
                            df_row.to_csv(output_file, index=False, encoding='utf-8', mode='w')
                        else:
                            df_row.to_csv(output_file, index=False, encoding='utf-8', mode='a', header=False)

                        print(f"处理完成并写入: {stego_file}")

                    except Exception as e:
                        print(f"处理文件 {stego_file_path} 时出错: {e}")

        # 计算并保存统计结果
        if os.path.exists(output_file):
            stats_output_file = os.path.join(results_dir, 'free_energy', f'{base_name}_dna_energy_stats.csv')
            print(f"\n生成统计结果：{stats_output_file}")

            # 读取结果并计算统计
            result_df = pd.read_csv(output_file)
            if not result_df.empty:
                # 提取方法名（从隐写文件名中）
                result_df['method'] = result_df['stego_file'].apply(lambda x: '_'.join(x.split('_')[:-1]))

                # 计算统计结果
                stats = []
                for method in sorted(result_df['method'].unique()):
                    group = result_df[result_df['method'] == method]
                    stats_dict = {
                        'method': method,
                        'mean_dg_diff': f"{group['mean_dg_diff'].mean():.3f}±{group['mean_dg_diff'].std():.3f}",
                        'relative_diff_percent': f"{group['relative_diff_percent'].mean():.2f}±{group['relative_diff_percent'].std():.2f}",
                        'count': len(group)
                    }
                    stats.append(stats_dict)

                stats_df = pd.DataFrame(stats)
                stats_df.to_csv(stats_output_file, index=False)
                print(f"统计结果已保存到：{stats_output_file}")

    print("\n处理完成！所有文件已更新。")
