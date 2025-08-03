import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import cg_tm_kl
from data_helper import get_data, trans_sign
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score,recall_score,precision_score,f1_score
import numpy as np
import pandas
import os
import logger
from logger import Logger
def PathSlect(PATH):
    PathOutput = []
    for path in PATH:
        if path.find('StegaSeq')> 0:
            PathOutput.append(path)

    return PathOutput
# 检查是否可以使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
#  自定义数据集类
class SequenceDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# 自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, time_steps, dimension):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder_lstm1 = nn.LSTM(dimension, 32, batch_first=True, dropout=0.5)
        self.encoder_lstm2 = nn.LSTM(32, 2, batch_first=True, dropout=0.5)
        
        # 解码器
        self.decoder_lstm1 = nn.LSTM(2, 2, batch_first=True, dropout=0.5)
        self.decoder_lstm2 = nn.LSTM(2, 32, batch_first=True, dropout=0.3)
        self.decoder_dense = nn.Linear(32, dimension)

    def forward(self, x):
        # 编码
        encoded1, _ = self.encoder_lstm1(x)
        encoded2, _ = self.encoder_lstm2(encoded1)
        
        # 解码
        decoded1, _ = self.decoder_lstm1(encoded2)
        decoded2, _ = self.decoder_lstm2(decoded1)
        decoded = self.decoder_dense(decoded2)
        
        return decoded, encoded2

# LSTM 分类器
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步
        x = self.relu(self.fc1(lstm_out))
        x = self.sigmoid(self.fc2(x))
        return x

# CNN 分类器
class CNNClassifier(nn.Module):
    def __init__(self, input_dim, time_steps):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=4, padding='same')
        self.conv2 = nn.Conv1d(input_dim, 64, kernel_size=8, padding='same')
        self.conv3 = nn.Conv1d(input_dim, 64, kernel_size=12, padding='same')
        
        # 使用自适应池化，避免维度计算错误
        self.pool1 = nn.AdaptiveMaxPool1d(1)
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.pool3 = nn.AdaptiveMaxPool1d(1)
        
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(192, 64)  # 64*3=192 (3个卷积���的输出)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 转换输入维度以适应卷积层
        x = x.permute(0, 2, 1)
        
        # 三个并行的卷积层
        c1 = self.pool1(self.relu(self.conv1(x)))
        c2 = self.pool2(self.relu(self.conv2(x)))
        c3 = self.pool3(self.relu(self.conv3(x)))
        
        # 连接三个卷积层的输出
        concat = torch.cat((c1, c2, c3), dim=1)
        flatten = concat.view(concat.size(0), -1)
        
        # 全连接层
        x = self.dropout1(flatten)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc2(x))
        return x

def RNNs_Classifier(path):
    print(f"\nTraining RNNs classifier on file: {os.path.basename(path)}")
    
    # 加载数据
    data_x_final, y = get_data(path)

    # 检查数据平衡性
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"数据分布: {dict(zip(unique_labels, counts))}")

    # 设置固定样本数量为每类1000个
    target_samples_per_class = 2000

    # 如果数据不平衡或超过目标数量，进行平衡处理
    if len(set(counts)) > 1 or max(counts) > target_samples_per_class:
        print(f"将平衡到每类 {target_samples_per_class} 个样本")

        # 分离不同类别的数据
        balanced_X = []
        balanced_y = []

        for label in unique_labels:
            label_indices = [i for i, l in enumerate(y) if l == label]
            current_count = len(label_indices)

            if current_count < target_samples_per_class:
                print(f"警告: 类别 {label} 样本数不足 ({current_count} < {target_samples_per_class})")
                # 使用所有可用样本
                selected_indices = label_indices
            else:
                # 随机选择target_samples_per_class个样本
                selected_indices = np.random.choice(label_indices, target_samples_per_class, replace=False)

            balanced_X.extend([data_x_final[i] for i in selected_indices])
            balanced_y.extend([y[i] for i in selected_indices])

        data_x_final = np.array(balanced_X)
        y = balanced_y

        # 重新检查平衡后的分布
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"平衡后数据分布: {dict(zip(unique_labels, counts))}")

    X_train, X_test, y_train, y_test = train_test_split(data_x_final, y, test_size=0.2, shuffle=True, stratify=y)
    
    # 创建数据加载器
    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 初始化模型
    autoencoder = Autoencoder(X_train.shape[1], X_train.shape[2]).to(device)
    optimizer = optim.SGD(autoencoder.parameters(), lr=0.01, momentum=0.7)
    criterion = nn.MSELoss()
    
    # 训练自编码器
    print("Training autoencoder...")
    autoencoder.train()
    autoencoder_epochs = 1  # 增加训练轮数
    for epoch in range(autoencoder_epochs):
        epoch_loss = 0
        batch_count = 0
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            decoded, encoded = autoencoder(batch_X)
            loss = criterion(decoded, batch_X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f'Autoencoder Epoch {epoch+1}/{autoencoder_epochs}, Loss: {avg_loss:.6f}')

        # 早停机制
        if avg_loss < 0.001:
            print(f"Early stopping at epoch {epoch+1} due to low loss")
            break
    
    # 使用编码器提取特征
    autoencoder.eval()
    train_features = []
    test_features = []
    
    with torch.no_grad():
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            _, encoded = autoencoder(batch_X)
            train_features.append(encoded.cpu().numpy())
            
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            _, encoded = autoencoder(batch_X)
            test_features.append(encoded.cpu().numpy())
    
    train_features = np.concatenate(train_features)
    test_features = np.concatenate(test_features)
    
    # 合并原始特征和编码特征
    train_combined = np.concatenate([X_train, train_features], axis=2)
    test_combined = np.concatenate([X_test, test_features], axis=2)
    
    # 训练 CNN 分类器
    cnn = CNNClassifier(train_combined.shape[2], train_combined.shape[1]).to(device)
    optimizer = optim.Adam(cnn.parameters())
    criterion = nn.BCELoss()
    
    # 创建新的数据加载器
    train_dataset = SequenceDataset(train_combined, y_train)
    test_dataset = SequenceDataset(test_combined, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 训练 CNN
    cnn.train()
    for epoch in range(40):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = cnn(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
    
    # 评估模型
    cnn.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = cnn(batch_X)
            probs = outputs.squeeze().cpu().numpy()
            predictions = (probs > 0.5).astype(int)
            all_preds.extend(predictions)
            all_probs.extend(probs)

    # 计算指标
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 详细分析
    pred_dist = np.bincount(y_pred)
    label_dist = np.bincount(y_test)
    prob_stats = {
        'mean': np.mean(y_probs),
        'std': np.std(y_probs),
        'min': np.min(y_probs),
        'max': np.max(y_probs)
    }

    print(f"=== 评估结果 ===")
    print(f"Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
    print(f"预测分布: {pred_dist}, 真实分布: {label_dist}")
    print(f"概率统计: mean={prob_stats['mean']:.3f}, std={prob_stats['std']:.3f}")

    # 异常检测
    if accuracy > 0.95:
        print("⚠️  警告: 准确率异常高，可能存在数据泄露!")
    if prob_stats['std'] < 0.1:
        print("⚠️  警告: 预测概率方差过小，模型过于确定!")

    # 按照要求的顺序返回: recall, accuracy, precision, f1-score
    return recall, accuracy, precision, f1

# if __name__ == '__main__':
#     modes =['660','888','847']

#     for mode in modes:
#         # file_ori = r'C:\Users\Administrator\Desktop\epoch\Loss_minium_Ep=53_660' # 生成文件路径 文件格式：txt
#         file_ori = r"D:\Desktop\seqs\baseline\{}".format(mode)
#         file_csv = r'D:\Desktop\seqs\baseline\{}_csv'.format(mode)  # csv文件路径，原文件与生成文件1：1
#         # ori      = r'D:\Desktop\seqs\660\read_in_20\OriginalData\read_20.txt'  # 原文件路径

#         file_result_score = r'D:\Desktop\seqs\660\CLASSFINAL\S'  # 单个csv文件的得分
#         file_pic_result = r'D:\Desktop\seqs\660\CLASSFINAL\P'  # csv文件中的得分图

#         file_finaly = r'D:\Desktop\seqs\baseline\{}.csv'.format(mode)  # 总csv文件的得分值集合

#         if mode == '888':
#             sample_num = 4500
#         elif mode == '847':
#             sample_num = 4200  # 生成文件的语句数目，660：3300，888：4500
#         else:
#             sample_num = 3300
#         PATH = []
#         for root, dirs, files in os.walk(file_ori):
#             for file in files:
#                 PATH.append(os.path.join(root, file))
#         # PATH = PathSlect(PATH)
#         BPN = {}
#         VAR = {}

#         for p in PATH:
#             p_ = p[p.rfind('\\') + 1:  len(p) - 4]
#             BPN[p_] = cg_tm_kl.find_bpn(p)
#             VAR[p_] = cg_tm_kl.find_var(p)
#             # if p_.find('fxy6') > 0 or p_.find('fxy3') or p_.find('fxy33') or p_.find('fxy66') > 0:
#             #    SeqLength = 198
#             #    ori = r'D:\Desktop\seqs\{}\read_3\OriginalData\read_3.txt'.format(mode)
#             # else:
#             #    SeqLength = 200
#             #    ori = r'D:\Desktop\seqs\{}\read_2\OriginalData\read_2.txt'.format(mode)
#             SeqLength = 198
#             ori = r'D:\Desktop\seqs\{}\read_3\OriginalData\read_3.txt'.format(mode)
#             csv_shengc(p, file_csv, ori, sample_num, Seqlength=SeqLength)

#         PATH_csv = []
#         for root, dirs, files in os.walk(file_csv):
#             for file in files:
#                 PATH_csv.append(os.path.join(root, file))

#         temp = pd.DataFrame(columns=['path', 'accurcy', 'reacll_score', 'pres', 'f1s', 'bpn', 'ebpn', 'var'])

#         for p in PATH_csv:
#             bpn = 0
#             ebpn = 0
#             var = 0
#             p_ = p[p.rfind('\\') + 1:  len(p) - 4]
#             accurcy, reaclls, pres, f1s = main(p, result_score=file_result_score, result_pic=file_pic_result)
#             bpn = float(BPN[p_])
#             ebpn = (1 - accurcy) * bpn * 2
#             var = float(VAR[p_])
#             temp = temp.append([{'path': p_, 'accurcy': accurcy, 'reacll_score': reaclls, 'pres': pres, 'f1s': f1s,
#                                  'bpn': bpn, 'ebpn': ebpn, 'var': var}], ignore_index=True)
#             print(temp)

#         print(temp)
#         temp.to_csv(file_finaly)

#         AdgScoreAcc, AriScoreAcc = [], []
#         AdgScoreRec, AriScoreRec = [], []
#         AdgScorePre, AriScorePre = [], []
#         AdgScoreF1s, AriScoreF1s = [], []
#         for index, row in temp.iterrows():
#             t = row['path']
#             num = row['path'].find('adg')
#             if row['path'].find('adg') >= 0:
#                 AdgScoreAcc.append(float(row['accurcy']))
#                 AdgScoreRec.append(float(row['reacll_score']))
#                 AdgScorePre.append(float(row['pres']))
#                 AdgScoreF1s.append(float(row['f1s']))
#             else:
#                 AriScoreAcc.append(float(row['accurcy']))
#                 AriScoreRec.append(float(row['reacll_score']))
#                 AriScorePre.append(float(row['pres']))
#                 AriScoreF1s.append(float(row['f1s']))
#         print('AdgAcc:{} AdgReS:{}'.format(np.mean(np.array(AdgScoreAcc)), np.mean(np.array(AdgScoreRec))))
#         print('------')
#         print('AriAcc:{} AriReS:{}'.format(np.mean(np.array(AriScoreAcc)), np.mean(np.array(AriScoreRec))))
#         '''
        # # 获取数据
        # data_x_final, y = get_data()

        # # 划分数据
        # X_train, X_test, y_train, y_test = train_test_split(data_x_final, y, test_size=0.2, shuffle=True)

        # # 模型的建立
        # autoEncoder, encoder = build_autoEncoder(X_train.shape[1], X_train.shape[2])

        # # autoencoder 的训练
        # adam = Adam(lr=0.0001)
        # sgd = SGD(lr=0.01, momentum=0.7)
        # autoEncoder.compile(optimizer=sgd, loss='mse')
        # # print(autoEncoder.summary())
        # history = autoEncoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=1, batch_size=64)

        # # 通过encoder提取的特征
        # output_encoder_train = encoder.predict(X_train)
        # output_encoder_test = encoder.predict(X_test)
        # # print('特征维度')
        # # print(output_encoder_train.shape)

        # # 将特征与原始数据堆叠
        # data_concat_train = np.concatenate([X_train, output_encoder_train], axis=2)
        # data_concat_test = np.concatenate([X_test, output_encoder_test], axis=2)
        # # print('堆叠后���度')
        # # print(data_concat_train.shape)

        # # 将特征提取出来作为下一个预测网络的输入
        # # lstm 预测
        # lstm = build_lstm(data_concat_train.shape[1], data_concat_train.shape[2])
        # lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # history_lstm = lstm.fit(data_concat_train, y_train, validation_split=0.2, epochs=1, batch_size=64)

        # # cnn 预测
        # cnn = build_cnn(data_concat_train.shape[1], data_concat_train.shape[2])
        # cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # history_cnn = cnn.fit(data_concat_train, y_train, validation_split=0.2, epochs=20, batch_size=64)

        # # 测试集的预测
        # y_pred_proba_lstm = lstm.predict(data_concat_test)
        # y_pred_proba_cnn = cnn.predict(data_concat_test)
        # # print(y_pred_proba_lstm)
        # #y_pred_lstm = np.argmax(y_pred_proba_lstm, axis=1)
        # y_pred_lstm = trans_sign(y_pred_proba_lstm, 0.5)

        # # print(y_pred_proba_cnn)
        # y_pred_cnn = trans_sign(y_pred_proba_cnn, 0.5)
        # # y_pred_cnn = np.argmax(y_pred_proba_cnn, axis=1)
        # # print(y_pred_cnn)



        # # 打印结果
        # # print('accuracy lstm:')
        # # print(accuracy_score(y_test, y_pred_lstm))
        # print('accuracy cnn:')
        # print(accuracy_score(y_test, y_pred_cnn))

        # save = pd.DataFrame(columns=['prob', 'label'], index=range(len(y_test)))
        # save['prob'] = np.squeeze(y_pred_proba_cnn)
        # save['label'] = y_test

        # ori_score = []
        # sc_score = []
        # for index,row in save.iterrows():
        #     if row['label'] == 0.0:
        #         ori_score.append(row['prob'])
        #     else:
        #         sc_score.append(row['prob'])

        # save.to_csv('score_LSB.csv')

        # plt.figure(1)
        # plt.plot(history.history['loss'], c='r', label='loss')
        # plt.plot(history.history['val_loss'], c='b', label='val_loss')
        # plt.legend()

        # plt.figure(2)
        # plt.plot(history_lstm.history['loss'], c='r', label='loss')
        # plt.plot(history_lstm.history['val_loss'], c='b', label='val_loss')
        # plt.legend()
        # plt.title('LSTM')

        # plt.figure(3)
        # plt.plot(history_cnn.history['loss'], c='r', label='loss')
        # plt.plot(history_cnn.history['val_loss'], c='b', label='val_loss')
        # plt.legend()
        # plt.title('CNN')

        # plt.show()
        # '''

