import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import pickle
from itertools import cycle
import pickle
from sklearn.utils import shuffle
from torch.utils.data import random_split, TensorDataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import csv


k_values = [4]
# k_values = [3, 4, 5, 6]
# files_name = ['ASM141792v1']
files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1']
# method = 'ssDiscop' 3:66 4:50

BATCH_SIZE = 512
EPOCH = 3
Num_class = 2
Num_layers = 1
LAMDA = 0.08
LR = 0.001
BN_DIM = 66 if k_values == [3] else 50 # 66 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_variable(file_name, variable):
    # 创建目录
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    # 保存文件
    file_object = open(file_name, "wb")
    pickle.dump(variable, file_object)
    file_object.close()

# Add this near the top of the file, after imports
rootPath = "/home/fan/Code4idea/xLSTMstega/Antisteganalysis/DS_CLF"

# Then modify the get_alter_loaders function
def get_alter_loaders(k, base_name, file_num):
    map_file = f"{rootPath}/data_processing/k_{k}/DNA_map_k_{k}.txt"
    File_Embed = f"{rootPath}/data_processing/k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/train_val/steg.txt"
    File_NoEmbed = f"{rootPath}/data_processing/k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/train_val/raw.txt"
    pklfile_train = f'{rootPath}/data_processing/k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/pklfiles/DNA_train.pkl'
    pklfile_val = f'{rootPath}/data_processing/k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/pklfiles/DNA_val.pkl'
    if not os.path.exists(pklfile_train):

        with open(map_file, 'r') as map_file:
            dna_map = {}
            for line in map_file:
                key, value = line.strip().split()
                dna_map[key] = value

        raw_filepath = File_NoEmbed
        steg_filepath = File_Embed

        with open(raw_filepath, 'r') as raw_file:
            raw_data = raw_file.readlines()
            raw_data_mapped = [' '.join([dna_map[base] for base in line.split()]) for line in raw_data]

        with open(steg_filepath, 'r') as steg_file:
            steg_data = steg_file.readlines()
            steg_data_mapped = [' '.join([dna_map[base] for base in line.split()]) for line in steg_data]

        data = raw_data_mapped + steg_data_mapped
        labels = [0] * len(raw_data) + [1] * len(steg_data)

        # 显示数据平衡信息
        print(f"训练数据平衡信息:")
        print(f"  天然序列数量: {len(raw_data)}")
        print(f"  隐写序列数量: {len(steg_data)}")
        print(f"  总数据量: {len(data)}")

        data_shuffled,labels_shuffled = shuffle(data,labels)
        print('data长度: ',len(data_shuffled))

        data_shuffled = torch.tensor([list(map(int, sample.split())) for sample in data_shuffled])
        labels_shuffled = torch.tensor(labels_shuffled)
        dataset = TensorDataset(data_shuffled, labels_shuffled)



        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        save_variable(pklfile_train, train_loader)
        save_variable(pklfile_val, val_loader)

        print("训练集长度:", len(train_loader))
        print("验证集长度:", len(val_loader))


    else:  
        train_loader = pickle.load(open(pklfile_train, 'rb'))  
        val_loader = pickle.load(open(pklfile_val, 'rb'))  

        print("训练集长度:", len(train_loader))
        print("验证集长度:", len(val_loader))


    return train_loader,val_loader




def convert_to_loader_CL(x_train, y_train, x_val, y_val, batch_size):

    x_train_tensor = torch.Tensor(x_train)
    y_train_tensor = torch.Tensor(y_train)
    x_val_tensor = torch.Tensor(x_val)
    y_val_tensor = torch.Tensor(y_val)


    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    steg_indices = [i for i, label in enumerate(y_train) if label[1] == 1]

    cover_indices = [i for i, label in enumerate(y_train) if label[1] == 0]


    train_steg_dataset = Subset(train_dataset, steg_indices)

    train_cover_dataset = Subset(train_dataset, cover_indices)

    train_steg_loader = DataLoader(train_steg_dataset, batch_size=batch_size, shuffle=True)
    train_cover_loader = DataLoader(train_cover_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    return train_steg_loader, train_cover_loader, val_loader


class Model_CL(nn.Module):
    def __init__(self, num_layers):
        super(Model_CL, self).__init__()
        self.embedding = nn.Embedding(BATCH_SIZE, 128)
        self.position_embedding = PositionalEncoding(128)
        self.transformer_encoder_layers = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layers, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(num_features=BN_DIM)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = x.long()
        emb_x = self.embedding(x)
        emb_x += self.position_embedding(emb_x)


        emb_x = emb_x.permute(1, 0, 2)  
        outputs = self.transformer_encoder(emb_x)
        #print(outputs.size())
        outputs = self.bn(outputs.permute(1, 0, 2))
        outputs = self.pooling(outputs.permute(0, 2, 1)).squeeze(2)

        return outputs


class Classifier_CL(nn.Module):
    def __init__(self, num_layers, num_class=Num_class):
        super(Classifier_CL, self).__init__()
        self.model = Model_CL(num_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_class)

    def forward(self, x):

        x_unsup = self.model(x)
        x_sup_1 = torch.zeros(x_unsup.size(0) // 3, x_unsup.size(1)).to(device)
        x_sup_2 = torch.zeros(x_unsup.size(0) // 3, x_unsup.size(1)).to(device)
        for i in range(x_sup_1.size(0)):
            x_sup_1[i] = x_unsup[3 * i]
            x_sup_2[i] = x_unsup[3 * i + 2]
        x_sup_1 = self.dropout(x_sup_1)
        x_sup_1 = self.fc(x_sup_1)
        x_sup_1 = F.softmax(x_sup_1, dim=1)
        x_sup_2 = self.dropout(x_sup_2)
        x_sup_2 = self.fc(x_sup_2)
        x_sup_2 = F.softmax(x_sup_2, dim=1)

        return x_unsup, x_sup_1, x_sup_2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1536):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0),:x.size(1),:x.size(2)]
        return x


def compute_CL_loss(y_pred,lamda=LAMDA):
    row = torch.arange(0,y_pred.shape[0],3,device='cuda')
    col = torch.arange(y_pred.shape[0], device='cuda')
    col = torch.where(col % 3 != 0)[0].cuda()
    y_true = torch.arange(0,len(col),2,device='cuda')
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    

    similarities = torch.index_select(similarities,0,row)
    similarities = torch.index_select(similarities,1,col)
    

    similarities = similarities / lamda
    

    loss = F.cross_entropy(similarities,y_true)
    return torch.mean(loss)


def train_val_model_CL(model, train_1_loader, train_0_loader, val_loader, optimizer, loss_fun_sup, k, base_name, gca_name, file_num, num_epochs=EPOCH):
    global result_path
    result_path = f'{rootPath}/data_processing/k_{k}/{method}/{base_name}/out_put/CL_{k}_file{file_num}'

    best_acc = 0.0
    best_metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_num = 0
        for (inputs_1, labels_1), (inputs_0, labels_0) in zip(train_1_loader, cycle(train_0_loader)):
            inputs_1, labels_1, inputs_0, labels_0 = inputs_1.to(device), labels_1.to(device), inputs_0.to(device), labels_0.to(device)

            labels_1 = labels_1.long()
            labels_0 = labels_0.long()

            inputs_size = min(inputs_1.size(0), inputs_0.size(0))
            inputs_1 = inputs_1[:inputs_size]
            inputs_0 = inputs_0[:inputs_size]
            labels_1 = labels_1[:inputs_size]
            labels_0 = labels_0[:inputs_size]

            input_final = torch.zeros(inputs_size*3, inputs_1.size(1)).to(device)
            for i in range(inputs_size):
                input_final[3*i] = inputs_0[i % inputs_size]
                input_final[3*i+1] = inputs_0[(i+1)  % inputs_size]
                input_final[3*i+2] = inputs_1[i  % inputs_size]

            optimizer.zero_grad()
            outputs_unsup, outputs_sup_1, outputs_sup_2 = model(input_final)
            loss_sup_1 = loss_fun_sup(outputs_sup_1, labels_0)
            loss_sup_2 = loss_fun_sup(outputs_sup_2, labels_1)
            loss_sup = (loss_sup_1 + loss_sup_2) / 2
            loss_unsup = compute_CL_loss(outputs_unsup)
            loss = loss_sup + loss_unsup*0.001
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs_size * 2
            total_num += inputs_size * 2

        epoch_loss = running_loss / total_num
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        model.eval()
        all_predictions = []
        all_labels = []
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                input_final = torch.zeros(inputs.size(0) * 3, inputs.size(1)).to(device)
                for i in range(inputs.size(0)):
                    input_final[3 * i] = inputs[i]
                    input_final[3 * i + 1] = inputs[i]
                    input_final[3 * i + 2] = inputs[i]
                _, outputs_sup_1, outputs_sup_2 = model(input_final)
                _, predicted_1 = torch.max(outputs_sup_1, 1)
                _, predicted_2 = torch.max(outputs_sup_2, 1)
                
                all_predictions.extend(predicted_1.cpu().numpy())
                all_predictions.extend(predicted_2.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                labels = labels.squeeze()
                total_preds += labels.size(0) * 2
                correct_preds += (predicted_1 == labels).sum().item()
                correct_preds += (predicted_2 == labels).sum().item()

        accuracy = correct_preds / total_preds
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1: {f1:.4f}")

        is_best = accuracy > best_acc
        if is_best:
            best_acc = accuracy
            best_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'metrics': best_metrics
            }, is_best, prefix=result_path + '/')

            f = open(os.path.join(result_path, "result.txt"), 'a')
            f.write(f"loaded best_checkpoint (epoch {epoch}, accuracy {best_acc:.4f}, precision {precision:.4f}, recall {recall:.4f}, f1 {f1:.4f})\n")
            f.close()
            testDNA(k, base_name, file_num)

    # 训练结束后，记录最准确率到CSV
    seqlen = 198 if k in [3, 6] else 200
    stego_file = f"{method}_{seqlen}_{k}_{file_num}.txt"
    write_results_to_csv(
        base_name,
        gca_name, 
        stego_file, 
        best_metrics['accuracy'],
        best_metrics['precision'],
        best_metrics['recall'],
        best_metrics['f1']
    )





def save_checkpoint(state, is_best, prefix):

    if is_best:
        directory = os.path.dirname(prefix)
        if not os.path.exists(directory):
            os.makedirs(directory)


        torch.save(state, prefix + 'model_best.pth.tar')
        print('save beat check :' + prefix + 'model_best.pth.tar')




def parse_sample_test(file_path):

    file = open(file_path, 'r')
    lines = file.readlines()
    sample = []
    for line in lines:

        line = [int(l) for l in line.split()]
        sample.append(line)
    return sample





def write_results_to_csv(base_name, gca_name, stego_file, accuracy, precision, recall, f1):
    # 使用GCA名称作为文件名
    csv_file = f'{rootPath}/{gca_name}_DS_CLF_results.csv'

    # 读取现有结果（如果存在）
    existing_results = {}
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # 跳过表头
            for row in reader:
                existing_results[row[0]] = {
                    'recall': float(row[1].rstrip('%')),
                    'accuracy': float(row[2].rstrip('%')),
                    'precision': float(row[3].rstrip('%')),
                    'f1': float(row[4].rstrip('%'))
                }

    # 转换当前指标为百分比
    current_metrics = {
        'recall': recall * 100,
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'f1': f1 * 100
    }

    # 检查是否需要更新
    should_update = True
    if stego_file in existing_results:
        # 只有当新的准确率更高时才更新
        if current_metrics['accuracy'] <= existing_results[stego_file]['accuracy']:
            should_update = False

    if should_update:
        # 更新或添加新结果
        existing_results[stego_file] = current_metrics

        # 写入所有结果
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Stego_File', 'Recall', 'Accuracy', 'Precision', 'F1_Score'])

            # 按文件名排序写入结果
            for file_name in sorted(existing_results.keys()):
                metrics = existing_results[file_name]
                writer.writerow([
                    file_name,
                    f"{metrics['recall']:.2f}%",
                    f"{metrics['accuracy']:.2f}%",
                    f"{metrics['precision']:.2f}%",
                    f"{metrics['f1']:.2f}%"
                ])

        print(f"Updated best result for {stego_file}")
    else:
        print(f"Skipped update for {stego_file} (current accuracy not better than existing)")


def test_model_with_best_checkpoint(k, base_name, file_num):
    seqlen = 198 if k in [3, 6] else 200
    
    global result_path
    result_path = f'{rootPath}/data_processing/k_{k}/{method}/{base_name}/out_put/CL_{k}_file{file_num}'
    model = Classifier_CL(num_layers=Num_layers)
    model = model.to(device)

    best_checkpoint = torch.load(os.path.join(result_path, 'model_best.pth.tar'),weights_only=False)
    print('load bestcheck from :', os.path.join(result_path, 'model_best.pth.tar'))
    model.load_state_dict(best_checkpoint['model'])

    test_loader = get_alter_loaders_test(k, base_name, file_num)

    model.eval()
    correct_preds = 0
    num_labels_1 = 0
    num_labels_0 = 0
    num_correct_1_pred_1 = 0
    num_correct_0_pred_1 = 0
    num_correct_1_pred_2 = 0
    num_correct_0_pred_2 = 0
    total_preds = 0
    with torch.no_grad():
        correct_positive = 0  
        total_positive = 0  
        correct_negative = 0  
        total_negative = 0  
        total_samples = 0  
        all_predictions = []
        all_labels = []

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            #print(labels.size())
            input_final = torch.zeros(inputs.size(0) * 3, inputs.size(1)).to(device)
            for i in range(inputs.size(0)):
                input_final[3 * i] = inputs[i]
                input_final[3 * i + 1] = inputs[i]
                input_final[3 * i + 2] = inputs[i]
            _, outputs_sup_1, outputs_sup_2 = model(input_final)
            _, predicted_1 = torch.max(outputs_sup_1, 1)
            _, predicted_2 = torch.max(outputs_sup_2, 1)

            #_, labels = torch.max(labels, 1)
            input_final = torch.zeros(inputs.size(0) * 3, inputs.size(1)).to(device)

            total_preds += labels.size(0) * 2
            correct_preds += (predicted_1 == labels).sum().item()
            correct_preds += (predicted_2 == labels).sum().item()
            
            num_labels_1 += torch.sum(labels == 1).item()
            num_labels_0 += torch.sum(labels == 0).item()
            
            num_correct_1_pred_1 += torch.sum((predicted_1 == labels) & (labels == 1)).item()
            num_false_1_pred_1 = torch.sum((predicted_1 != 1) & (labels == 1)).item()

            num_correct_0_pred_1 += torch.sum((predicted_1 == labels) & (labels == 0)).item()
            num_false_0_pred_1 = torch.sum((predicted_1 != 0) & (labels == 0)).item()

            num_correct_1_pred_2 += torch.sum((predicted_2 == labels) & (labels == 1)).item()
            num_false_1_pred_2 = torch.sum((predicted_2 != 1) & (labels == 1)).item()


            num_correct_0_pred_2 += torch.sum((predicted_2 == labels) & (labels == 0)).item()
            num_false_0_pred_2 = torch.sum((predicted_2 != 0) & (labels == 0)).item()
            

            all_predictions.extend(predicted_1.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())


    accuracy_1_num = num_correct_1_pred_1 + num_correct_1_pred_2
    accuracy_0_num = num_correct_0_pred_1 + num_correct_0_pred_2
    accuracy_1 = accuracy_1_num / (num_labels_1 * 2)
    accuracy_0 = accuracy_0_num / (num_labels_0 * 2)
    accuracy_bi = accuracy_1 / 2 + accuracy_0 / 2
    accuracy = correct_preds / total_preds

    # 计算各���指标
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    total_samples = len(all_labels)
    total_accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / total_samples if total_samples != 0 else 0
    total_precision = precision_score(all_labels, all_predictions)
    total_recall = recall_score(all_labels, all_predictions)
    total_f1 = f1_score(all_labels, all_predictions)
    
    # 构造隐写文件名
    seqlen = 198 if k in [3, 6] else 200
    stego_file = f"{method}_{seqlen}_{k}_{file_num}.txt"
    
    # 写入所有指标到CSV
    write_results_to_csv(base_name, gca_name, stego_file, total_accuracy, total_precision, total_recall, total_f1)

    print(f"Positive Sample Accuracy: {accuracy_1:.4f}")
    print(f"Negative Sample Accuracy: {accuracy_0:.4f}")
    print(f"Total Accuracy: {total_accuracy:.4f}")

    print(f"Total Precision: {total_precision:.4f}")
    print(f"Total Recall: {total_recall:.4f}")
    print(f"Total F1 Score: {total_f1:.4f}")

    f = open(os.path.join(result_path, "result.txt"), 'a')
    f.write(f"Positive Sample Accuracy: {accuracy_1:.4f}\n")
    f.write(f"Negative Sample Accuracy: {accuracy_0:.4f}\n")
    f.write("Total Accuracy: %.4f\n" % total_accuracy)
    f.write("Total Precision: %.4f\n" % total_precision)
    f.write("Total Recall: %.4f\n" % total_recall)
    f.write("Total F1 Score: %.4f\n" % total_f1)
    f.close()






def get_alter_loaders_test(k, base_name, file_num):
    map_file = f"{rootPath}/data_processing/k_{k}/DNA_map_k_{k}.txt"
    File_Embed = f"{rootPath}/data_processing/k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/test/steg.txt"
    File_NoEmbed = f"{rootPath}/data_processing/k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/test/raw.txt"
    pklfile_test = f'{rootPath}/data_processing/k_{k}/{method}/{base_name}/data_k_{k}_file{file_num}/pklfiles/DNA_test.pkl'
    if not os.path.exists(pklfile_test):

        with open(map_file, 'r') as map_file:
            dna_map = {}
            for line in map_file:
                key, value = line.strip().split()
                dna_map[key] = value

        raw_filepath = File_NoEmbed
        steg_filepath = File_Embed

        with open(raw_filepath, 'r') as raw_file:
            raw_data = raw_file.readlines()
            raw_data_mapped = [' '.join([dna_map[base] for base in line.split()]) for line in raw_data]

        with open(steg_filepath, 'r') as steg_file:
            steg_data = steg_file.readlines()
            steg_data_mapped = [' '.join([dna_map[base] for base in line.split()]) for line in steg_data]


        data = raw_data_mapped + steg_data_mapped
        labels = [0] * len(raw_data) + [1] * len(steg_data)

        # 显示测试数据平衡信息
        print(f"测试数据平衡信息:")
        print(f"  天然序列数量: {len(raw_data)}")
        print(f"  隐写序列数量: {len(steg_data)}")
        print(f"  总数据量: {len(data)}")

        data_shuffled,labels_shuffled = shuffle(data,labels)
        print('data长度: ',len(data_shuffled))
        
        data_shuffled = torch.tensor([list(map(int, sample.split())) for sample in data_shuffled])
        labels_shuffled = torch.tensor(labels_shuffled)

        dataset = TensorDataset(data_shuffled, labels_shuffled)




        test_loader = DataLoader( dataset, batch_size=BATCH_SIZE, shuffle=False)


        save_variable(pklfile_test, test_loader)

        print("测试集长度:", len(test_loader))

    else: 
        test_loader = pickle.load(open(pklfile_test, 'rb'))  

        print("测试集长度:", len(test_loader))

    return test_loader






def testDNA(k, base_name, file_num):
    test_model_with_best_checkpoint(k, base_name, file_num)


def write_average_results_to_csv(base_name):
    input_csv = f'./{base_name}_DS_CLF_results.csv'
    output_csv = f'./{base_name}_DS_CLF_stats.csv'
    
    if not os.path.exists(input_csv):
        print(f"Warning: Results file {input_csv} not found. Skipping statistics calculation.")
        return
    
    # 读取原始结果
    results = {}
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # 获取表头
        for row in reader:
            filename = row[0]
            metrics = [float(val.rstrip('%')) for val in row[1:]]  # 移除百分号并转换为浮点数
            
            # 移除文件名中的最后的 _数字.txt 部分
            base_filename = '_'.join(filename.split('_')[:-1])
            
            if base_filename not in results:
                results[base_filename] = []
            results[base_filename].append(metrics)
    
    # 检查是否有数据需要写入
    if not results:
        print(f"Warning: No results found in {input_csv}. Skipping statistics calculation.")
        return
    
    # 计算均值和标准差
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stego_File', 'Recall', 'Accuracy', 'Precision', 'F1_Score'])

        for base_filename, metrics_list in results.items():
            if len(metrics_list) > 1:  # 确保有多个值可以计算
                metrics_array = np.array(metrics_list)
                means = np.mean(metrics_array, axis=0)
                stds = np.std(metrics_array, axis=0)

                formatted_stats = [
                    f"{mean:.2f}±{std:.2f}%"
                    for mean, std in zip(means, stds)
                ]

                writer.writerow([base_filename] + formatted_stats)


def has_been_processed(base_name, method, k, file_num):
    """检查文件是否已经处理过"""
    csv_file = f'./{gca_name}_DS_CLF_results.csv'
    if not os.path.exists(csv_file):
        return False
        
    seqlen = 198 if k in [3, 6] else 200
    target_file = f"{method}_{seqlen}_{k}_{file_num}.txt"
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if row[0] == target_file:
                return True
    return False

# 根据method设置epoch数
def get_epoch_by_method(method):
    if method.lower() in ['xlstmadg']:
        return 300
    return 80  # 默认epoch数

if __name__ == '__main__':
    # print('\nDefault EPOCH = ', EPOCH)
    # print('Num_layers = ', Num_layers)
    asm_to_gca = {
    'ASM141792v1': 'GCA_001417925',
    'ASM286374v1': 'GCA_002863745',
    'ASM400647v1': 'GCA_004006475',
    'ASM949793v1': 'GCA_009497935',
    'ASM1821919v1': 'GCA_018219195'
    }
    for base_name in files_name:
        gca_name = asm_to_gca[base_name]

        print(f"\nProcessing genome: {base_name} (GCA: {gca_name})")
        for k in k_values:
            print(f"\nProcessing k = {k}")
            seqlen = 198 if k in [3, 6] else 200
            
            # 获取所有可能的方法
            data_dir = f"{rootPath}/data_processing/k_{k}"
            if not os.path.exists(data_dir):
                print(f"Skipping k={k} - directory not found: {data_dir}")
                continue
                
            methods = []
            for method_dir in os.listdir(data_dir):
                if os.path.isdir(os.path.join(data_dir, method_dir)) and method_dir != "DNA_map_k_{k}":
                    methods.append(method_dir)
            
            print(f"Found methods: {methods}")
            
            # 处理每个方法
            for method in methods:
                # 根据method设置epoch数
                current_epoch = get_epoch_by_method(method)
                print(f"\nProcessing method: {method}, EPOCH = {current_epoch}")
                
                method_dir = f"./data_processing/k_{k}/{method}/{base_name}"
                if not os.path.exists(method_dir):
                    print(f"Skipping method {method} - directory not found: {method_dir}")
                    continue
                    
                file_nums = []
                try:
                    for dir_name in os.listdir(method_dir):
                        if dir_name.startswith('data_k_') and '_file' in dir_name:
                            file_num = dir_name.split('file')[-1]
                            file_nums.append(file_num)
                except FileNotFoundError:
                    print(f"Skipping method {method} - directory not found: {method_dir}")
                    continue
                
                if not file_nums:
                    print(f"No files found for method {method}")
                    continue
                    
                print(f"\nProcessing method: {method}, file numbers: {file_nums}")
                
                # 处理每个文件
                for file_num in file_nums:
                    # 检查是否已处理过
                    if has_been_processed(gca_name, method, k, file_num):
                        print(f"Skipping {method}_{seqlen}_{k}_{file_num}.txt - already processed")
                        continue
                        
                    print(f"\nProcessing file number: {file_num}")
                    
                    # 创建所有必要的目录
                    base_dir = f'{rootPath}/data_processing/k_{k}/{method}/{base_name}'
                    dirs_to_create = [
                        f'{base_dir}/data_k_{k}_file{file_num}/train_val',
                        f'{base_dir}/data_k_{k}_file{file_num}/test',
                        f'{base_dir}/data_k_{k}_file{file_num}/pklfiles',
                        f'{base_dir}/out_put/CL_{k}_file{file_num}'
                    ]
                    
                    for dir_path in dirs_to_create:
                        os.makedirs(dir_path, exist_ok=True)
                    
                    result_path = f'{base_dir}/out_put/CL_{k}_file{file_num}'
                    
                    train_loader, val_loader = get_alter_loaders(k, base_name, file_num)
                    model = Classifier_CL(num_layers=Num_layers).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    loss_fun_sup = nn.CrossEntropyLoss()
                    
                    train_steg_loader = []
                    train_cover_loader = []
                    
                    for inputs, labels in train_loader:
                        for i in range(len(labels)):
                            if labels[i] == 1:
                                train_steg_loader.append((inputs[i], labels[i]))
                            else:
                                train_cover_loader.append((inputs[i], labels[i]))
                    
                    train_1_loader = torch.utils.data.DataLoader(train_steg_loader, batch_size=train_loader.batch_size, shuffle=True)
                    train_0_loader = torch.utils.data.DataLoader(train_cover_loader, batch_size=train_loader.batch_size, shuffle=True)
                    
                    train_val_model_CL(
                        model, 
                        train_1_loader, 
                        train_0_loader, 
                        val_loader, 
                        optimizer, 
                        loss_fun_sup, 
                        k, 
                        base_name,
                        gca_name,  # 添加GCA名称参数
                        file_num, 
                        num_epochs=current_epoch
                    )
    
    print("\nAll training completed. Calculating statistics...")
    # 在所有处理完成后，计算并保存平均结果
    for base_name in files_name:
        gca_name = asm_to_gca[base_name]
        write_average_results_to_csv(gca_name)  # 使用GCA名称
    print("Statistics calculation completed.")
