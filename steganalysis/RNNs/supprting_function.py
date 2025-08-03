import os
import numpy as np
import pandas as pd
import cg_tm_kl
import pathlib
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
def csv_generate(pathsc, file_csv, ori, sample_num, Seqlength):
    # 设置生成文件的目录
    base_name = os.path.basename(os.path.dirname(pathsc))  # 获取 base_name
    save_dir = f'/home/fan/Code/VAE_Synthetic_Steganography/ExperimentData/antisteganalysis/RF/{base_name}'
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 处理隐写序列文件
    raw_pos = cg_tm_kl.txt_process_sc_duo(pathsc, len_sc=Seqlength, beg_sc=0, end_sc=sample_num, PADDING=False, flex=10, num1=Seqlength)
    raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
    
    pathwrite_sc = os.path.join(save_dir, 'raw_pos.txt')
    
    with open(pathwrite_sc, 'w') as f1:
        for line in raw_pos:
            f1.write(line + '\n')

    # 处理原始序列文件
    raw_neg = cg_tm_kl.txt_process_sc_duo(ori, len_sc=Seqlength, beg_sc=0, end_sc=sample_num, PADDING=False, flex=10, num1=Seqlength)
    raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))

    pathwrite_ori = os.path.join(save_dir, 'raw_neg.txt')
    with open(pathwrite_ori, 'w') as f2:
        for line in raw_neg:
            f2.write(line + '\n')

    # 读取文件并合并数据
    pos = pd.read_csv(pathwrite_sc, header=None)
    neg = pd.read_csv(pathwrite_ori, header=None)

    # 将 pos 和 neg 转为一维数组
    x = np.concatenate((pos.values.flatten(), neg.values.flatten()))
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    # 创建数据框并添加文本和标签
    save = pd.DataFrame(columns=['text', 'label'], index=range(len(x)))
    save['text'] = x
    save['label'] = y

    # 生成 CSV 文件路径并保存
    csv_filename = f"{os.path.splitext(os.path.basename(pathsc))[0]}.csv"
    file_ = os.path.join(file_csv, csv_filename)
    save.to_csv(file_, index=False)

    print(f"Data saved to {file_}")
