import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import argparse
import sys
from logger import Logger
# from torchsummary import summary
#from trend import ple
import numpy as np
#import KL
import data
import textcnn
import cg_tm_kl
import os
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda x: x.lower() == 'true')
    parser.add_argument("--filename", type=str, default='lsb')
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--stop", type=int, default=2)
    args = parser.parse_args(sys.argv[1:])
    return args


args = get_args()


def main(data_helper):
    # ======================
    # 超参数
    # ======================

    STOP = args.stop

    all_var = locals()
    print()
    '''
    for var in all_var:
        if var != "var_name":
            logger.info("{0:15} ".format(var))
            logger.info(all_var[var])
    print()
    '''
    # ======================
    # 数据
    # ======================

    # ======================
    # 构建模型
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = textcnn.TextCNN(
        vocab_size=data_helper.vocab_size,
        embed_size=EMBED_SIZE,
        filter_num=FILTER_NUM,
        filter_size=FILTER_SIZE,
        class_num=CLASS_NUM,
        dropout_rate=DROPOUT_RATE
    )
    model.to(device)
    # 	summary(model, (20,))
    criteration = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    early_stop = 0
    best_acc = 0
    best_reacll = 0
    best_precison = 0
    F1score = 0

    # ======================
    # 训练与测试
    # ======================
    for epoch in range(EPOCH):
        generator_train = data_helper.train_generator(BATCH_SIZE)
        generator_test = data_helper.test_generator(BATCH_SIZE)
        train_loss = []
        train_acc = []
        while True:
            try:
                text, label = generator_train.__next__()
            except:
                break
            optimizer.zero_grad()
            y = model(torch.from_numpy(text).long().to(device))
            loss = criteration(y, torch.from_numpy(label).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            y = y.cpu().detach().numpy()
            train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

        test_loss = []
        test_acc = []
        test_precision = []
        test_recall = []
        while True:
            with torch.no_grad():
                try:
                    text, label = generator_test.__next__()
                except:
                    break
                y = model(torch.from_numpy(text).long().to(device))
                loss = criteration(y, torch.from_numpy(label).long().to(device))
                test_loss.append(loss.item())
                y = y.cpu().numpy()
                test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

                for i in range(len(y)):
                    if np.argmax(y[i]) == 1:
                        test_precision += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
                for i in range(len(y)):
                    if label[i] ==1:
                        test_recall += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

        logger.info("epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}"
                    .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))

        if np.mean(test_precision) > best_precison:
            best_precison = np.mean(test_precision)

        if np.mean(test_recall) > best_reacll:
            best_reacll = np.mean(test_recall)

        if np.mean(test_acc) > best_acc:
            best_acc = np.mean(test_acc)
        else:
            early_stop += 1
        if early_stop >= STOP:
            try:
                F1score = float(2 *(best_reacll*best_precison) / (best_reacll+best_precison))
            except:
                F1score = 0
            logger.info('best recall: {:.4f}, best acc: {:.4f}, best precision:{:.4f}, F1score:{:.4f}'.format(best_reacll, best_acc, best_precison, F1score))

    # if (epoch + 1) % SAVE_EVERY == 0:
    # 			print('saving parameters')
    # 			os.makedirs('models', exist_ok=True)
    # 			torch.save(model.state_dict(), 'models/textcnn-' + str(epoch) + '.pkl')
    # logger.info('best acc: {:.4f}'.format(best_acc))

    # 按照新的顺序返回：recall, accuracy, precision, f1-score
    return best_reacll, best_acc, best_precison, F1score


def calculate_and_save_stats(df, output_file):
    """计算并保存统计结果"""
    # 从文件名中提取方法名
    df['method'] = df['FileName'].apply(lambda x: '_'.join(x.split('_')[:3]))
    
    stats = []
    for method in df['method'].unique():
        group = df[df['method'] == method]
        stats_dict = {
            'method': method,
            # 将所有指标转换为百分比并保留两位小数
            'accuracy': f"{group['accuracy'].mean()*100:.2f}±{group['accuracy'].std()*100:.2f}%",
            'recall': f"{group['recall'].mean()*100:.2f}±{group['recall'].std()*100:.2f}%",
            'precision': f"{group['precision'].mean()*100:.2f}±{group['precision'].std()*100:.2f}%",
            'F1': f"{group['f1_score'].mean()*100:.2f}±{group['f1_score'].std()*100:.2f}%"
        }
        stats.append(stats_dict)
    
    # 创建统计结果DataFrame并保存
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_file, index=False)
    print(f"Statistics saved to {output_file}")


def get_processed_files(csv_path):
    """获取已经处理过的文件列表"""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return set(df['FileName'].values)
    return set()


if __name__ == '__main__':
    BATCH_SIZE = 50
    EMBED_SIZE = 350
    FILTER_NUM = 512
    # FILTER_SIZE = [3]
    FILTER_SIZE = [3, 4, 5, 6]
    CLASS_NUM = 2
    DROPOUT_RATE = 0.4
    EPOCH = args.epoch
    LEARNING_RATE = 0.0001
    SAVE_EVERY = 20
    # SL = 4
    # 原始文件路径定义
    files_name = ['ASM286374v1']
    # files_name = ['ASM141792v1', 'ASM286374v1', 'ASM400647v1', 'ASM949793v1']
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
        Path_result_save = f'/home/fan/Code4idea/xLSTMstega/results/antisteganalysis/TextCNN/{gca_name}'
        Path_final_result = f'/home/fan/Code4idea/xLSTMstega/results/antisteganalysis/TextCNN'
        Path_generated_file = f'/home/fan/Code4idea/xLSTMstega/Stego_DNA/{base_name}'
        
        os.makedirs(Path_result_save, exist_ok=True)
        csv_path = os.path.join(Path_final_result, f'{gca_name}_textcnn.csv')
        
        # 获取已处理的文件列表
        processed_files = get_processed_files(csv_path)
        print(f"\n已处理的文件数量：{len(processed_files)}")

        # 设置原始文件路径
        # index = ['1']  # 可扩展为 ['3', '4', '5', '6']
        index = ['1', '2', '3', '4', '5', '6']  # 可扩展为 ['3', '4', '5', '6']
        
        
        # 根据索引设置数据处理
        for ind in index:
            # 确定序列长度
            len_sc = 198 if int(ind) in [1, 3, 6] else 200
            # print('len_sc', len_sc)
            file_original = f'/home/fan/Code4idea/xLSTMstega/Dataset/{base_name}_{len_sc}_{ind}.txt'

            # 处理原始序列
            print(f"\nProcessing original file: {file_original}")
            try:
                with open(file_original, 'r') as f:
                    raw_pos = cg_tm_kl.txt_process_sc_duo(file_original, len_sc=len_sc, beg_sc=0, 
                                                        # end_sc=None, PADDING=False, flex=80, num1=3, tiqu=False)
                                                        end_sc=None, PADDING=False, flex=80, num1=int(ind), tiqu=False)
                raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
                print(f"Original file has {len(raw_pos)} sequences.")
            except FileNotFoundError:
                print(f"Error: Original file not found.")
                continue

            ff = pd.DataFrame(columns=['FileName', 'accuracy', 'recall', 'precision', 'f1_score'])

            # 定义要处理的目录列表
            generated_dirs = [
                f'/home/fan/Code4idea/xLSTMstega/Stego_DNA/Baselines/{base_name}',
                f'{Path_generated_file}',  # 基础目录
            ]

            # 遍历每个目录
            for generated_dir in generated_dirs:
                if not os.path.exists(generated_dir):
                    print(f"目录不存在，跳过：{generated_dir}")
                    continue
                    
                print(f"\n处理目录：{generated_dir}")
                
                for file in os.listdir(generated_dir):
                    if file.endswith(f'_{len_sc}_{ind}_1.txt') or \
                       file.endswith(f'_{len_sc}_{ind}_2.txt') or \
                       file.endswith(f'_{len_sc}_{ind}_3.txt'):
                        
                        path = os.path.join(generated_dir, file)
                        file_name = file[:-4]  # 去掉.txt后缀
                        
                        # 检查是否已处理
                        if file_name in processed_files:
                            print(f"跳过已处理的文件：{file_name}")
                            continue
                            
                        print(f"\n处理新文件：{file_name}")
                        
                        # 设置日志
                        log_dir = os.path.join(Path_result_save, 'log_files')
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, f"{file_name}_LoggerFile.txt")
                        logger = Logger(log_file)

                        # 处理隐写序列
                        raw_neg = cg_tm_kl.txt_process_sc_duo(path, len_sc=len_sc, beg_sc=0, 
                                                            # end_sc=None, PADDING=False, flex=80, num1=3, tiqu=False)
                                                            end_sc=None, PADDING=False, flex=80, num1=int(ind), tiqu=False)
                        raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))
                        logger.info(f"处理序列数量: {len(raw_neg)}")

                        # 训练模型
                        data_helper = data.DataHelper([raw_pos, raw_neg], use_label=True)
                        recall, acc, p, f1 = main(data_helper)

                        # 记录结果 - 按照新顺序：recall, accuracy, precision, f1-score
                        results = pd.DataFrame([{
                            'FileName': file_name,
                            'recall': recall,
                            'accuracy': acc,
                            'precision': p,
                            'f1_score': f1
                        }])
                        ff = pd.concat([ff, results], ignore_index=True)
                        
                        # 即时保存结果
                        if os.path.exists(csv_path):
                            results.to_csv(csv_path, mode='a', header=False, index=False)
                        else:
                            results.to_csv(csv_path, index=False)
                            
                        print(f"已保存结果：{file_name}")

            # 更新统计结果
            if not ff.empty:
                calculate_and_save_stats(
                    pd.read_csv(csv_path), 
                    os.path.join(Path_final_result, f'{gca_name}_textcnn_stats.csv')
                )

    print("\n处理完成！所有文件已更新。")



