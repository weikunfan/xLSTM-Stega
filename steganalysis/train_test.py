import os
import random
import glob

def copy_random_lines_balanced(source_file, destination_file, train_and_val_file, steg_file_lines, percentage=10):
    """
    根据隐写文件的行数来平衡数据集

    Args:
        source_file: 源文件路径（天然序列）
        destination_file: 测试集输出文件
        train_and_val_file: 训练集输出文件
        steg_file_lines: 隐写文件的行数
        percentage: 测试集比例（默认10%）
    """
    with open(source_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)

    # 根据隐写文件行数确定使用的样本数量
    # 确保天然序列和隐写序列数量一致
    max_samples = min(total_lines, steg_file_lines)

    print(f"天然序列总行数: {total_lines}")
    print(f"隐写序列行数: {steg_file_lines}")
    print(f"平衡后使用样本数: {max_samples}")

    # 从天然序列中随机选择max_samples行
    if total_lines > max_samples:
        selected_all_indices = random.sample(range(total_lines), max_samples)
        balanced_lines = [lines[i] for i in selected_all_indices]
    else:
        balanced_lines = lines

    # 从平衡后的数据中划分训练集和测试集
    num_test = int(max_samples * percentage / 100)
    test_indices = random.sample(range(len(balanced_lines)), num_test)

    with open(destination_file, 'w') as f_dest, open(train_and_val_file, 'w') as f_train_val:
        for idx, line in enumerate(balanced_lines):
            if idx in test_indices:
                f_dest.write(line)
            else:
                f_train_val.write(line)

    print(f"测试集样本数: {num_test}")
    print(f"训练集样本数: {len(balanced_lines) - num_test}")

def copy_random_lines(source_file, destination_file, train_and_val_file, percentage):
    """保持原有函数以兼容性"""
    with open(source_file, 'r') as f:
        lines = f.readlines()

    num_lines = len(lines)
    num_lines_to_copy = int(num_lines * percentage / 100)

    selected_indices = random.sample(range(num_lines), num_lines_to_copy)

    with open(destination_file, 'w') as f_dest, open(train_and_val_file, 'w') as f_train_val:
        for idx, line in enumerate(lines):
            if idx in selected_indices:
                f_dest.write(line)
            else:
                f_train_val.write(line)

# 处理不同的k值
for k in [3,4,5,6]:
# for k in [1, 2]:
# for k in [3,4,5,6]:
    # files_name = ['ASM141792v1']
    files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1']
    for base_name in files_name:
        seqlen = 198 if k in [1, 3, 6] else 200

        Original_path = f"/home/fan/Code4idea/xLSTMstega/Dataset"
        
        # 定义stego路径列表
        stego_paths = [
            # f"/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/{base_name}/{seqlen}",
            f"/home/fan/Code4idea/xLSTMstega/Stego_DNA/{base_name}",
            f'/home/fan/Code4idea/xLSTMstega/Stego_DNA/Baselines/{base_name}',
            # 添加其他路径...
        ]
        
        raw_source_file = f"{Original_path}/{base_name}_{seqlen}_{k}.txt"

        # 遍历每个stego路径
        for stego_path in stego_paths:
            # print('stego_path:', stego_path)
            # 获取该目录下所有符合模式的文件
            pattern = f"{stego_path}/*_{seqlen}_{k}_*.txt"
            stego_files = glob.glob(pattern)
            # print('stego_files:', stego_files)
            
            for steg_source_file in stego_files:
                # 从文件名中提取method和file_num
                filename = os.path.basename(steg_source_file)
                parts = filename.split('_')
                method = parts[0]  # 假设文件名格式为 method_seqlen_k_filenum.txt
                file_num = parts[3].split('.')[0]  # 去掉.txt后缀
                
                # 检查源文件是否存在
                if not os.path.exists(steg_source_file):
                    print(f"Skipping {steg_source_file} - file not found")
                    continue

                # Check if the directory for the method already exists
                method_dir = f"./k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}"
                if not os.path.exists(method_dir):
                    os.makedirs(f"{method_dir}/test", exist_ok=True)
                    os.makedirs(f"{method_dir}/train_val", exist_ok=True)
                else:
                    print(f"Directory for method {method} already exists, skipping creation.")

                # 构建输出文件路径，加入file_num以区分不同的源文件
                raw_destination_val_file = f"./k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/test/raw.txt"
                raw_destination_train_file = f"./k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/train_val/raw.txt"

                steg_destination_val_file = f"./k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/test/steg.txt"
                steg_destination_train_file = f"./k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/train_val/steg.txt"
            
                # Check if the destination files already exist
                if not (os.path.exists(raw_destination_val_file) and os.path.exists(raw_destination_train_file) and
                        os.path.exists(steg_destination_val_file) and os.path.exists(steg_destination_train_file)):
                    print(f"\nProcessing k={k}, method={method}, file {file_num}...")

                    # 首先获取隐写文件的行数
                    with open(steg_source_file, 'r') as f:
                        steg_lines = len(f.readlines())

                    print(f"Splitting raw data from: {raw_source_file}")
                    print(f"Splitting steg data from: {steg_source_file}")

                    # 使用平衡函数，确保天然序列和隐写序列数量一致
                    copy_random_lines_balanced(raw_source_file, raw_destination_val_file, raw_destination_train_file, steg_lines, 10)
                    copy_random_lines(steg_source_file, steg_destination_val_file, steg_destination_train_file, 10)

                    print(f"Completed processing for k={k}, method={method}, file {file_num}")
                else:
                    print(f"Files for method {method}, file {file_num} already exist, skipping processing.")
                    print(f"Raw validation data saved to: {os.path.abspath(raw_destination_val_file)}")  # 打印绝对路径
