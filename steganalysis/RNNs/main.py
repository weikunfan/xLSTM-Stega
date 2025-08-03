import os
import pathlib
import pandas as pd
import logger
from logger import Logger
import cg_tm_kl
import numpy as np
import antoencoder
import torch

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# GPU 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 验证 GPU 是否可用
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

def data_analysis(temp,type):
    accurcy_average = np.mean(pd.to_numeric(temp[type]).tolist())
    accurcy_up = max(pd.to_numeric(temp[type]).tolist()) - accurcy_average
    accurcy_down = accurcy_average - min(pd.to_numeric(temp[type]).tolist())
    accurcy_std = np.std((pd.to_numeric(temp[type]).tolist()))
    return accurcy_average,accurcy_up,accurcy_down,accurcy_std

def data_analysis_bpn(temp):
    bpn_average = np.mean(pd.to_numeric(temp['bpn']).tolist())
    bpn_std = np.std((pd.to_numeric(temp['bpn']).tolist()))
    return bpn_average,bpn_std

def data_analysis_ebpn(temp):
    ebpn_average = np.mean(pd.to_numeric(temp['ebpn']).tolist())
    ebpn_std = np.std((pd.to_numeric(temp['ebpn']).tolist()))
    return ebpn_average, ebpn_std

def csv_generate(pathsc, file_csv_rnns, ori, sample_num, Seqlength):
    # 处理隐写序列件
    raw_pos = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc=Seqlength, beg_sc=0, end_sc=sample_num, PADDING=False, flex=10, num1=Seqlength)
    raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))

    # 修正路径构建，使用 temp 目录作为基础
    temp_dir = os.path.dirname(os.path.dirname(file_csv_rnns))
    pathwrite_sc = os.path.join(temp_dir, 'RNNs', 'raw_pos.txt')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(pathwrite_sc), exist_ok=True)
    
    with open(pathwrite_sc, 'w') as f1:
        for line in raw_pos:
            f1.write(line + '\n')

    # 处理原始序列文件
    raw_neg = cg_tm_kl.txt_process_sc_duo(ori, len_sc=Seqlength, beg_sc=0, end_sc=sample_num, PADDING=False, flex=10, num1=Seqlength)
    raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))

    pathwrite_ori = os.path.join(temp_dir, 'RNNs', 'raw_neg.txt')
    with open(pathwrite_ori, 'w') as f2:
        for line in raw_neg:
            f2.write(line + '\n')

    # 数据验证和平衡
    print(f"隐写序列数量: {len(raw_pos)}")
    print(f"天然序列数量: {len(raw_neg)}")

    if len(raw_pos) == 0 or len(raw_neg) == 0:
        print("警告: 序列数据为空!")
        return

    # 设置固定样本数量为1000
    target_samples = 1000
    print(f"平衡前 - 隐写序列: {len(raw_pos)}, 天然序列: {len(raw_neg)}")
    print(f"将各取 {target_samples} 个样本进行训练")

    # 检查样本数量是否足够
    if len(raw_pos) < target_samples:
        print(f"警告: 隐写序列数量不足 ({len(raw_pos)} < {target_samples})")
        target_samples = len(raw_pos)
    if len(raw_neg) < target_samples:
        print(f"警告: 天然序列数量不足 ({len(raw_neg)} < {target_samples})")
        target_samples = min(target_samples, len(raw_neg))

    # 随机采样以确保数据多样性
    import random
    raw_pos = random.sample(raw_pos, target_samples)
    raw_neg = random.sample(raw_neg, target_samples)

    print(f"平衡后 - 隐写序列: {len(raw_pos)}, 天然序列: {len(raw_neg)}")
    print(f"最终使用样本数: 隐写={len(raw_pos)}, 天然={len(raw_neg)}, 总计={len(raw_pos) + len(raw_neg)}")

    # 直接合并序列数据（避免文件读写错误）
    all_sequences = raw_pos + raw_neg
    labels = [1] * len(raw_pos) + [0] * len(raw_neg)

    # 验证数据一致性
    assert len(all_sequences) == len(labels), f"序列数量与标签数量不匹配!"

    # 检查序列长度
    seq_lengths = [len(seq.replace(' ', '')) for seq in all_sequences]
    print(f"序列长度范围: {min(seq_lengths)} - {max(seq_lengths)}")

    # 创建DataFrame
    save = pd.DataFrame({
        'text': all_sequences,
        'label': labels
    })

    # 打乱数据
    save = save.sample(frac=1).reset_index(drop=True)

    csv_filename = f"{os.path.splitext(os.path.basename(pathsc))[0]}.csv"
    file_ = os.path.join(file_csv_rnns, csv_filename)
    save.to_csv(file_, index=False)

    print(f"数据已保存到: {file_}")
    print(f"最终数据形状: {save.shape}")
    print(f"标签分布: {save['label'].value_counts().to_dict()}")

    # 显示前几个样本
    print("\n前3个样本:")
    for i in range(min(3, len(save))):
        text_preview = save.iloc[i]['text'][:50] + "..." if len(save.iloc[i]['text']) > 50 else save.iloc[i]['text']
        print(f"  样本{i+1}: 标签={save.iloc[i]['label']}, 文本='{text_preview}'")

if __name__ == '__main__':
    Classifier_modes = {}
    Classifier_modes['RNNs'] = True
    files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1']
    # files_name = ['ASM141792v1']
    # files_name = ['ASM286374v1']
    # ASM和GCA的映射关系
    asm_to_gca = {
        'ASM141792v1': 'GCA_001417925',
        'ASM286374v1': 'GCA_002863745',
        'ASM400647v1': 'GCA_004006475',
        'ASM949793v1': 'GCA_009497935',
        'ASM1821919v1': 'GCA_018219195'
    }
    for base_name in files_name:
        gca_name = asm_to_gca[base_name]

        
        # 设置目录路径
        antisteganalysis_dir = f'/home/fan/Code4idea/xLSTMstega/results/antisteganalysis/RNNs/{base_name}'
        temp_dir = os.path.join(antisteganalysis_dir, 'temp')
        pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)

        Path_result_save = os.path.join(temp_dir, 'RNNs')
        file_csv_rnns = os.path.join(temp_dir, 'csv')
        
        pathlib.Path(Path_result_save).mkdir(parents=True, exist_ok=True)
        pathlib.Path(file_csv_rnns).mkdir(parents=True, exist_ok=True)
        
        Path_generated_file = f'/home/fan/Code4idea/xLSTMstega/Stego_DNA/{base_name}'
        RNNs_Result_csv_file = os.path.join('/home/fan/Code4idea/xLSTMstega/results/antisteganalysis/RNNs', f'{gca_name}_rnns_results.csv')
        log_file = os.path.join(Path_result_save, 'RNNs_log.txt')

        # index = ['3']
        index = ['1', '2', '3','4','5','6']
        for ind in index:
            len_sc = 198 if int(ind) in [1,3,6] else 200
            file_csv_rnns = os.path.join(Path_result_save, 'csv')
            pathlib.Path(file_csv_rnns).mkdir(parents=True, exist_ok=True)

            print(f"Processing index {ind}, sequence length {len_sc}")
            print(f"CSV files will be saved to: {file_csv_rnns}")

            file_generated = [
                os.path.join(Path_generated_file),
                f'/home/fan/Code4idea/xLSTMstega/Stego_DNA/Baselines/{base_name}',
                # os.path.join(Path_generated_file, str(len_sc), 'VAE', 'lr_0.003','discop2'),
                # os.path.join(Path_generated_file, str(len_sc), 'huffman')
            ]

            log_file = os.path.join(Path_result_save, 'RNNs_log.txt')
            logger = Logger(log_file)

            file_original = os.path.join(f'/home/fan/Code4idea/xLSTMstega/Dataset/{base_name}_{len_sc}_{ind}.txt')

            with open(file_original, 'r') as f1:
                lines = f1.readlines()

            if Classifier_modes['RNNs']:
                PATH_generated = []
                PATH_generated_csv = []

                # 1. 首先检查文件生成过程
                for generated_path in file_generated:
                    for file in os.listdir(generated_path):
                        if os.path.isfile(os.path.join(generated_path, file)):
                            # 添加文件名检查，避免重复
                            if f'_{len_sc}_{ind}' in file and file.endswith('.txt'):
                                file_path = os.path.join(generated_path, file)
                                # 检查是否已经处理过该文件
                                if file_path not in PATH_generated:
                                    print(f"Processing new file: {file}")
                                    PATH_generated.append(file_path)
                                    file_original = os.path.join(f'/home/fan/Code4idea/xLSTMstega/Dataset/{base_name}_{len_sc}_{ind}.txt')
                                    # 设置固定样本数量为1000
                                    sample_num = 1000
                                    print(f"使用固定样本数量: {sample_num}")
                                    csv_generate(file_path, file_csv_rnns, file_original, sample_num, Seqlength=len_sc)

                # 2. 收集生成的CSV文件
                processed_files = set()  # 使用集合来跟踪已处理的文件
                for root, dirs, files in os.walk(file_csv_rnns):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 检查文件是否已经处理过
                        if file_path not in processed_files:
                            PATH_generated_csv.append(file_path)
                            processed_files.add(file_path)
                            print('Found new csv file:', file)

                # 3. 处理CSV文件
                for file_g in PATH_generated_csv:
                    p_ = os.path.basename(file_g).split('.')[0]
                    parts = p_.split('_')
                    
                    # 添加基本的错误检查
                    if len(parts) >= 4:  # 确保有足够的部分
                        # 首先检查是否已存在结果
                        if os.path.exists(RNNs_Result_csv_file):
                            existing_results = pd.read_csv(RNNs_Result_csv_file)
                            if any(existing_results['path'] == p_):
                                print(f"Results for {p_} already exist, skipping...")
                                continue  # 跳过后续处理
                        
                        prefix = parts[0]
                        len_sc = int(parts[1])
                        ind = int(parts[2])
                        repetition = int(parts[3])
                        
                        print('Processing file:', p_)
                        
                        # 只有在结果不存在时才进行训练
                        rnns_reaclls, rnns_accurcy, rnns_pres, rnns_f1s = antoencoder.RNNs_Classifier(file_g)

                        new_row = pd.DataFrame([{
                            'path': p_,
                            'recall': rnns_reaclls,
                            'accuracy': rnns_accurcy,
                            'precision': rnns_pres,
                            'f1_score': rnns_f1s
                        }])

                        if os.path.exists(RNNs_Result_csv_file):
                            new_row.to_csv(RNNs_Result_csv_file, mode='a', header=False, index=False)
                            print(f"Results for {p_} appended to {RNNs_Result_csv_file}")
                        else:
                            new_row.to_csv(RNNs_Result_csv_file, index=False)
                            print(f"Created new results file with {p_}")

        # 在每个数据集处理完成后,对结果文件进行排序和统计
        if os.path.exists(RNNs_Result_csv_file):
            # 读取结果文件
            df = pd.read_csv(RNNs_Result_csv_file)

            # 按path列排序
            df_sorted = df.sort_values('path')

            # 创建新的DataFrame来存储统计结果
            stats_results = []

            # 提取unique的模型前缀和参数组合(不包括最后的重复次数)
            paths = df_sorted['path'].str.rsplit('_', n=1).str[0].unique()

            # 对每个组合计算统计值
            for base_path in paths:
                # 获取同组的所有结果
                group = df_sorted[df_sorted['path'].str.startswith(base_path + '_')]

                # 计算各指标的均值和标准差，转换为百分比并保留两位小数
                # 按照新的顺序：recall, accuracy, precision, f1-score
                stats = {
                    'path': base_path,
                    'recall': f"{group['recall'].mean()*100:.2f}±{group['recall'].std()*100:.2f}%",
                    'accuracy': f"{group['accuracy'].mean()*100:.2f}±{group['accuracy'].std()*100:.2f}%",
                    'precision': f"{group['precision'].mean()*100:.2f}±{group['precision'].std()*100:.2f}%",
                    'f1_score': f"{group['f1_score'].mean()*100:.2f}±{group['f1_score'].std()*100:.2f}%"
                }
                stats_results.append(stats)

            # 创建统计结果DataFrame并排序
            stats_df = pd.DataFrame(stats_results)
            stats_df = stats_df.sort_values('path')

            # 保存统计结果到新文件
            stats_file = RNNs_Result_csv_file.replace('.csv', '_stats.csv')
            stats_df.to_csv(stats_file, index=False)

            # 保存原始排序结果
            df_sorted.to_csv(RNNs_Result_csv_file, index=False)

            print(f"Original results have been sorted by path in {RNNs_Result_csv_file}")
            print(f"Statistical results have been saved to {stats_file}")