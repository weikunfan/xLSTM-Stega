import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def get_data(path):
    print(f"Loading data from: {path}")
    file_csv = path
    data = pd.read_csv(file_csv)

    # 数据验证
    print(f"原始数据形状: {data.shape}")
    print(f"列名: {data.columns.tolist()}")

    # 如果 'text' 列名存在大小写差异，可以统一处理
    data.columns = data.columns.str.strip()  # 去除列名中的空格

    if 'text' not in data.columns:
        raise KeyError("Data does not contain 'text' column")

    print(f"标签分布: {data['label'].value_counts().to_dict()}")
    
    y = data[['label']].values
    y = y.reshape(-1,)
    y = [int(value) for value in y]
    
    x = data[['text']].values
    print(f"文本样本数: {len(x)}, 标签样本数: {len(y)}")

    # 检查前几个样本
    print("前3个样本:")
    for i in range(min(3, len(x))):
        text_preview = str(x[i][0])[:50] + "..." if len(str(x[i][0])) > 50 else str(x[i][0])
        print(f"  样本{i+1}: 标签={y[i]}, 文本='{text_preview}'")

    data_x = []
    invalid_chars_count = 0

    for i in range(len(x)):
        temp = []
        text = str(x[i][0])  # 确保是字符串，获取实际的文本字符串

        # 统计无效字符
        invalid_chars = [char for char in text if char not in ['A', 'T', 'C', 'G', ' ']]
        if invalid_chars:
            invalid_chars_count += 1
            print(f"  样本{i+1}包含无效字符: {set(invalid_chars)}")

        for char in text:
            if char != ' ':  # 去除空格
                if char in ['A', 'T', 'C', 'G']:  # 只保留有效DNA字符
                    temp.append(char)

        if len(temp) == 0:
            print(f"警告: 样本{i+1}没有有效的DNA字符")
            temp = ['A']  # 添加默认字符避免空序列

        data_x.append(temp[:200])  # 截取前200个字符

    print(f"包含无效字符的样本数: {invalid_chars_count}")
    
    # 固定编码映射，避免LabelEncoder的随机性
    char_to_num = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    # 转换为数字编码
    data_x_encoded = []
    for seq in data_x:
        encoded_seq = []
        for char in seq:
            if char in char_to_num:
                encoded_seq.append(char_to_num[char])
            else:
                encoded_seq.append(0)  # 默认为A
        # 确保长度为200
        if len(encoded_seq) < 200:
            encoded_seq.extend([0] * (200 - len(encoded_seq)))
        else:
            encoded_seq = encoded_seq[:200]
        data_x_encoded.append(encoded_seq)

    data_x = np.array(data_x_encoded)
    print(f"编码后数据形状: {data_x.shape}")
    print(f"编码范围: {data_x.min()} - {data_x.max()}")
    
    # 使用固定的One-Hot编码
    data_x_final = []
    for seq in data_x:
        # 手动创建one-hot编码，确保一致性
        seq_onehot = np.zeros((len(seq), 4))  # 4类：A,T,C,G
        for i, char_idx in enumerate(seq):
            if 0 <= char_idx <= 3:
                seq_onehot[i, char_idx] = 1
            else:
                seq_onehot[i, 0] = 1  # 默认为A
        data_x_final.append(seq_onehot)
    
    data_x_final = np.array(data_x_final)

    # 最终验证
    print(f"处理后序列数量: {len(data_x)}")
    print(f"最终数据形状: {data_x_final.shape}")
    print(f"数据类型: {data_x_final.dtype}")
    print(f"标签数量: {len(y)}")

    # 验证数据一致性
    assert len(data_x_final) == len(y), f"特征数量({len(data_x_final)})与标签数量({len(y)})不匹配!"

    return data_x_final, y


# # 获取训练测试数据
# def get_data(path):
#     # file_csv = './data/fxyTLSM.csv'
#     file_csv = path
#     data = pd.read_csv(file_csv, index_col=0)
#     print(data.head(5))
#     y = data[['label']].values
#     y = y.reshape(-1, )
#     print('helllooooo1')
#     y = [int(value) for value in y]
#     x = data[['text']].values
#     print('helllooooo1')
#     data_x = []
#     print(len(x[0][0]))
#     for i in range(len(x)):
#         temp = []
#         for j in range(len(x[i][0])):
#             if x[i][0][j] != ' ':
#                 temp.append(x[i][0][j])
#         data_x.append(temp[:200])  # 取200个字符
#     data_x = np.array(data_x)
#     shape1 = data_x.shape[0]
#     shape2 = data_x.shape[1]
#     data_x = data_x.reshape(-1, 1)
#     lb = LabelEncoder()
#     lb.fit(data_x)
#     print(lb.classes_)
#     data_x = lb.transform(data_x)
#     print(data_x[0])
#     data_x = data_x.reshape((shape1, shape2))
#     print(data_x)
#     data_x_final = []
#     onehot = OneHotEncoder()
#     onehot.fit([[0], [1], [2], [3]])
#     for i in range(data_x.shape[0]):
#         temp_data = data_x[i].reshape(-1, 1)
#         temp_data = onehot.transform(temp_data).toarray()
#         data_x_final.append(temp_data)
#     data_x_final = np.array(data_x_final)
#     print(data_x_final)
#     print(y)
#     return data_x_final, y

def trans_sign(pred_sign, proba):
    trans_data = []
    for i in range(len(pred_sign)):
        if abs(pred_sign[i][0]) > proba:
            trans_data.append(1)
        else:
            trans_data.append(0)
    return trans_data
