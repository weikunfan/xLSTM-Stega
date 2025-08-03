import os
import numpy
import random
import copy
import numpy as np
import cg_tm_kl
import pandas as pd

def count_lines(file_path):
    """统计文件的实际行数"""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def get_processed_files(csv_file):
    """获取已经处理过的文件列表"""
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return set(df['path'].values)
    return set()

if __name__ == '__main__':
    # files_name = ['ASM141792v1']
    # files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1']
    files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1']

    file_line_counts = {}
    
    # 根路径设置
    root_path = '/home/fan/Code4idea/xLSTMstega'
    results_dir = os.path.join(root_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    for base_name in files_name:
        # 结果文件路径
        writefile = os.path.join(results_dir, f'{base_name}_all_gcb_tmb_klp.csv')
        
        # 获取已处理的文件列表
        processed_files = get_processed_files(writefile)
        print(f"\n已处理的文件数量：{len(processed_files)}")

        # 初始化结果DataFrame
        result = pd.DataFrame(columns=['path', 'GCd', 'Tmd', 'KLp'])
        if os.path.exists(writefile):
            result = pd.read_csv(writefile)

        print(f"\n正在处理：{base_name}")
        base_dir = os.path.join(root_path, 'Dataset')

        # 遍历每个文件
        for file in os.listdir(base_dir):
            if file.endswith('.txt') and file.startswith(f'{base_name}_'):
                file_path = os.path.join(base_dir, file)
                
                # 提取文件参数
                try:
                    parts = file.split('_')
                    length_str = parts[1]
                    devide_ = int(parts[2].split('.')[0])
                    len_sc = int(length_str)
                except (IndexError, ValueError):
                    print(f"警告：文件名 {file} 不符合预期格式，跳过处理。")
                    continue

                # 打印原始DNA序列信息
                actual_lines = count_lines(file_path)
                file_line_counts[file_path] = actual_lines
                print(f"\n处理文件：{file_path}")
                print(f"文件行数：{actual_lines}")

                # 隐写DNA序列路径
                stego_dir = os.path.join(root_path, 'Stego_DNA', base_name)
                # stego_dir = os.path.join(root_path, 'Stego_DNA','Baselines', base_name)
                
                if not os.path.exists(stego_dir):
                    print(f"路径不存在，跳过：{stego_dir}")
                    continue
                    
                print(f"\n处理路径：{stego_dir}")
                
                # 遍历目录下的文件
                for stega_file in os.listdir(stego_dir):
                    if stega_file.endswith('.txt'):
                        full_path = os.path.join(stego_dir, stega_file)
                        filename = stega_file[:-4]  # 去掉.txt后缀
                        
                        # 检查是否已处理
                        processed_files = get_processed_files(writefile)
                        if filename in processed_files:
                            print(f"文件已处理，跳过：{filename}")
                            continue
                            
                        print(f"处理新文件：{filename}")

                        # 根据隐写文件行数，确定处理的行数
                        stega_lines = count_lines(full_path)
                        if stega_lines == 0:
                            print(f"警告：文件 {filename} 为空，跳过处理")
                            continue
                        
                        end_sc = stega_lines

                        # 读取并检查处理后的序列
                        linesori = cg_tm_kl.txt_process_sc_duo(file_path, len_sc=len_sc, beg_sc=0, 
                                                             end_sc=end_sc, PADDING=False, flex=0, 
                                                             devide_num=devide_)
                        linessc = cg_tm_kl.txt_process_sc_duo(full_path, len_sc=len_sc, beg_sc=0, 
                                                            end_sc=end_sc, PADDING=False, flex=0, 
                                                            devide_num=devide_)
                        
                        # 检查处理后的序列是否为空
                        if not linesori or not linessc:
                            print(f"警告：处理后的序列为空，跳过文件 {filename}")
                            continue

                        try:
                            # 计算度量
                            CGd = cg_tm_kl.CG_b(linesori, linessc, len_sc=len_sc)
                            Tmd = cg_tm_kl.Tmb(linesori, linessc, len_sc=len_sc)
                            klds = cg_tm_kl.KLDoubleStrand(linessc, linesori)
                        except ZeroDivisionError:
                            print(f"警告：计算度量时出现除零错误，跳过文件 {filename}")
                            continue
                        except Exception as e:
                            print(f"警告：处理文件 {filename} 时出现错误: {str(e)}")
                            continue
                        
                        # 添加新结果
                        new_row = pd.DataFrame({
                            'path': [filename], 
                            'GCd': [CGd], 
                            'Tmd': [Tmd], 
                            'KLp': [klds],
                        })
                        result = pd.concat([result, new_row], ignore_index=True)
                        
                        # 即时保存结果
                        result.sort_values(by='path').to_csv(writefile, index=False)
                        print(f"已保存结果：{filename}")

        # 计算并保存统计结果
        stats_writefile = os.path.join(results_dir, f'{base_name}_all_stats.csv')
        print(f"\n生成统计结果：{stats_writefile}")
        
        # 提取基础名称和参数
        result['base_name'] = result['path'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        
        # 从base_name中提取序列长度和分割参数
        def extract_params(name):
            parts = name.split('_')
            if len(parts) >= 3:
                length = parts[-2]  # 倒数第二个部分通常是长度
                split = parts[-1]   # 最后一个部分通常是分割参数
                return f"{length}_{split}"
            return name
        
        result['params'] = result['base_name'].apply(extract_params)
        
        # 按照参数和方法名排序
        stats = []
        # 首先获取所有唯一的参数组合
        param_groups = sorted(result['params'].unique())
        
        for param in param_groups:
            # 对于每个参数组合，获取所有方法
            param_data = result[result['params'] == param]
            for name in sorted(param_data['base_name'].unique()):
                group = result[result['base_name'] == name]
                stats_dict = {
                    'method': name,
                    'GCd': f"{group['GCd'].mean():.3f}±{group['GCd'].std():.3f}",
                    'Tmd': f"{group['Tmd'].mean():.3f}±{group['Tmd'].std():.3f}",
                    'KLp': f"{group['KLp'].mean()*1000:.3f}±{group['KLp'].std()*1000:.3f}",
                }
                stats.append(stats_dict)
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(stats_writefile, index=False)

print("\n处理完成！所有文件已更新。")
