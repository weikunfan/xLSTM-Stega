import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from omegaconf import OmegaConf

# 添加上级目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dna_xlstm.modelling_xlstm import xLSTMConfig, xLSTMLMHeadModel
from model.logger import Logger
from remove_model import removeModel
import utils  # 假设与Model_training.py相同目录下的utils模块

rootPath = "/home/fan/Code4idea/xLSTMstega"
print(torch.__version__)
print(torch.cuda.is_available())

def train_xlstm(
    file_path,                # 输入数据文件路径
    code_name="xlstm_dna",    # 模型代码名称
    seq_length=1024,          # 序列长度
    sl="3",                # 读取索引
    save_path=os.path.join(rootPath, 'checkpoints'),     # 模型保存路径
    mode="train",          # 训练模式
    repeat="1",            # 模型编号
    learning_rate=0.0001,     # 学习率
    running_mode="s",          # 运行模式: 't'=测试, 's'=开始训练, 'c'=继续训练,
):

    # 检查输入文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 输入文件 '{file_path}' 不存在!")
        return None
    
    # 检查文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            print(f"错误: 输入文件 '{file_path}' 为空!")
            return None
        print(f"文件内容前100个字符: {content[:100]}...")
        print(f"文件总长度: {len(content)} 字符")
    
    # 加载xlstm.yaml配置
    configPath = os.path.join(rootPath, 'model/xlstm_sm.yaml')
    xlstm_config = OmegaConf.load(configPath)
    config_dict = xlstm_config.config
    
    # 设置训练参数
    RATIO = 0.9
    WORD_DROP = 0
    MIN_LEN = 1  # 将最小长度设置为较小的值，以确保有足够的序列可用
    MAX_LEN = 2000  # 增大最大长度以适应更多序列
    BATCH_SIZE = 30
    TEMPERATURE = 2.0  # 温度参数，>1使概率分布更平滑
    
    # 使用yaml中的模型参数
    EMBED_SIZE = config_dict.d_model
    HIDDEN_DIM = config_dict.d_model  # 使用相同的维度
    NUM_LAYERS = config_dict.n_layer
    DROPOUT_RATE = config_dict.dropout
    
    # 训练轮数设置
    if running_mode == 't':  # test
        EPOCH = 10
    elif running_mode == 's':  # start_train
        EPOCH = 20
    elif running_mode == 'c':  # continue_train
        EPOCH = 50

    LEARNING_RATE = learning_rate
    MAX_GENERATE_LENGTH = seq_length
    GENERATE_EVERY = 5
    PRINT_EVERY = 1
    SEED = 100
    
    # 创建日志目录和文件
    log_file = os.path.join(save_path, f"SL_{sl}", f'SLR_{learning_rate}', f"Repeat_{repeat}", f"{code_name}.txt")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 设置Logger
    logger = Logger(log_file)
    logger.info(
        f'Epoch {EPOCH} Lr {LEARNING_RATE} BatchSize {BATCH_SIZE} EmbedSize {EMBED_SIZE} HiddenDim {HIDDEN_DIM}'
    )
    
    # 数据处理
    data_path = file_path
    train_path = os.path.join(save_path, f"SL_{sl}", f'SLR_{learning_rate}', f"Repeat_{repeat}", 'info', "train.txt")
    test_path = os.path.join(save_path, f"SL_{sl}", f'SLR_{learning_rate}', f"Repeat_{repeat}", 'info', "test.txt")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    # 创建词汇表
    try:
        vocabulary = utils.Vocabulary(
            data_path,
            max_len=MAX_LEN,
            min_len=MIN_LEN,
            word_drop=WORD_DROP
        )
        print(f"词汇表大小: {vocabulary.vocab_size}")
    except Exception as e:
        print(f"创建词汇表时出错: {e}")
        return None
    
    # 分割语料库
    try:
        utils.split_corpus(data_path, train_path, test_path, max_len=MAX_LEN, min_len=MIN_LEN, ratio=RATIO, seed=SEED)
        
        # 检查分割后的文件
        with open(train_path, 'r', encoding='utf-8') as f:
            train_content = f.read().strip()
            print(f"训练集大小: {len(train_content)} 字符")
            
        with open(test_path, 'r', encoding='utf-8') as f:
            test_content = f.read().strip()
            print(f"测试集大小: {len(test_content)} 字符")
            
        if not train_content or not test_content:
            print("警告: 训练集或测试集为空!")
            print("尝试调整最小/最大长度参数...")
            
            # 如果分割后的文件为空，尝试使用更宽松的长度限制
            MIN_LEN = 1
            MAX_LEN = 5000
            print(f"重新尝试分割语料库，最小长度={MIN_LEN}，最大长度={MAX_LEN}")
            utils.split_corpus(data_path, train_path, test_path, max_len=MAX_LEN, min_len=MIN_LEN, ratio=RATIO, seed=SEED)
            
            # 再次检查
            with open(train_path, 'r', encoding='utf-8') as f:
                train_content = f.read().strip()
                print(f"调整后训练集大小: {len(train_content)} 字符")
                
            with open(test_path, 'r', encoding='utf-8') as f:
                test_content = f.read().strip()
                print(f"调整后测试集大小: {len(test_content)} 字符")
                
            if not train_content or not test_content:
                print("错误: 即使调整参数后，训练集或测试集仍为空!")
                return None
    except Exception as e:
        print(f"分割语料库时出错: {e}")
        return None
    
    # 构建语料库
    try:
        train = utils.Corpus(train_path, vocabulary, max_len=MAX_LEN, min_len=MIN_LEN)
        print(f"训练语料库句子数: {train.sentence_num}")
        
        test = utils.Corpus(test_path, vocabulary, max_len=MAX_LEN, min_len=MIN_LEN)
        print(f"测试语料库句子数: {test.sentence_num}")
        
        if train.sentence_num == 0 or test.sentence_num == 0:
            print("错误: 训练集或测试集中没有符合长度要求的句子!")
            print(f"当前设置: 最小长度={MIN_LEN}, 最大长度={MAX_LEN}")
            return None
    except Exception as e:
        print(f"构建语料库时出错: {e}")
        return None
    
    train_generator = utils.Generator(train.corpus, vocabulary=vocabulary)
    test_generator = utils.Generator(test.corpus, vocabulary=vocabulary)
    
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建xLSTM模型
    # 设置配置参数
    xlstm_cfg = xLSTMConfig(
        d_model=config_dict.d_model,
        n_layer=config_dict.n_layer,
        vocab_size=vocabulary.vocab_size,
        pad_vocab_size_multiple=config_dict.pad_vocab_size_multiple,
        max_length=MAX_LEN,
        m_conv1d_kernel_size=config_dict.m_conv1d_kernel_size,
        m_conv1d_causal=config_dict.m_conv1d_causal,
        m_qkv_proj_blocksize=config_dict.m_qkv_proj_blocksize,
        m_num_heads=config_dict.m_num_heads,
        m_proj_factor=config_dict.m_proj_factor,
        m_backend=config_dict.m_backend,
        m_chunk_size=config_dict.m_chunk_size,
        m_position_embeddings=config_dict.m_position_embeddings,
        m_bias=config_dict.m_bias,
        s_num_heads=config_dict.s_num_heads,
        s_conv1d_kernel_size=config_dict.s_conv1d_kernel_size,
        s_conv1d_causal=config_dict.s_conv1d_causal,
        s_lstm_at=config_dict.s_lstm_at,
        s_proj_factor=config_dict.s_proj_factor,
        s_round_proj_up_dim_up=config_dict.s_round_proj_up_dim_up,
        s_round_proj_up_to_multiple_of=config_dict.s_round_proj_up_to_multiple_of,
        s_position_embeddings=config_dict.s_position_embeddings,
        dropout=config_dict.dropout,
        bidirectional=config_dict.bidirectional,
        bidirectional_alternating=config_dict.bidirectional_alternating,
        rcps=config_dict.rcps,
    )
    
    model = xLSTMLMHeadModel(xlstm_cfg, device=device)
    model.to(device)
    
    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {total_trainable_params:,}")
    
    # 设置损失函数和优化器
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    
    # 加载之前的模型（如果是继续训练）
    LOAD_EPOCH = 0
    
    # 训练循环
    best_loss = 1000000
    step = 0
    early_stop = 0
    STOP = 100
    
    for epoch in range(int(LOAD_EPOCH) + 1, EPOCH + int(LOAD_EPOCH)):
        train_g = train_generator.build_generator(BATCH_SIZE)
        test_g = test_generator.build_generator(BATCH_SIZE)
        train_loss = []
        
        # 训练模式
        model.train()
        while True:
            try:
                text = train_g.__next__()
            except StopIteration:
                break
                
            optimizer.zero_grad()
            try:
                text = np.array(text).reshape((BATCH_SIZE, -1))
            except:
                print(f"文本形状异常: {np.shape(text)}")
                continue
                
            text_in = text[:, :-1]
            text_target = text[:, 1:]
            
            # 使用xLSTM模型前向传播
            output = model(torch.from_numpy(text_in).long().to(device))
            y = torch.log_softmax(output.logits, dim=-1)  # 应用log_softmax获取log概率
            
            yyy = y.reshape(-1, vocabulary.vocab_size)
            xxx = torch.from_numpy(text_target).reshape(-1).long().to(device)
            loss = criterion(yyy, xxx)
            loss.backward()
            
            optimizer.step()
            train_loss.append(loss.item())
            step += 1
            torch.cuda.empty_cache()
        
        # 测试模式
        test_loss = []
        model.eval()
        with torch.no_grad():
            while True:
                try:
                    text = test_g.__next__()
                except StopIteration:
                    break
                    
                text_in = text[:, :-1]
                text_target = text[:, 1:]
                
                output = model(torch.from_numpy(text_in).long().to(device))
                y = torch.log_softmax(output.logits, dim=-1)
                
                loss = criterion(
                    y.reshape(-1, vocabulary.vocab_size),
                    torch.from_numpy(text_target).reshape(-1).long().to(device)
                )
                test_loss.append(loss.item())
                torch.cuda.empty_cache()
        
        # 记录训练和测试损失
        train_loss_avg = np.mean(train_loss)
        test_loss_avg = np.mean(test_loss)
        logger.info(f'Epoch {epoch}   训练损失 {train_loss_avg:.4f}    测试损失 {test_loss_avg:.4f}')
        print(f'Epoch {epoch}   训练损失 {train_loss_avg:.4f}    测试损失 {test_loss_avg:.4f}')
        
        # 保存最佳模型
        if test_loss_avg < best_loss:
            best_loss = test_loss_avg
            print('-----------------------------------------------------')
            print('保存最佳模型参数')
            # save_path_model = os.path.join(save_path, mode, f"read_{index}", f"M_{model_num}")
            # os.makedirs(save_path_model, exist_ok=True)
            
            model_save_path = os.path.join(save_path, f'SL_{sl}',f'SLR_{learning_rate}', f'Repeat_{repeat}')
            os.makedirs(model_save_path, exist_ok=True)
            model_path = os.path.join(model_save_path, f"{code_name}-{repeat}-{epoch}-{test_loss_avg:.4f}.pkl")
            torch.save(model.state_dict(), model_path)
            print(f'模型已保存至: {model_path}')
            print('-----------------------------------------------------')
            early_stop = 0  # 重置早停计数器
        else:
            early_stop += 1
            # print(f'早停计数: {early_stop}/{STOP}')
        
        if early_stop >= STOP:
            print(f'达到早停条件，训练结束')
            break
        
        # 生成示例文本
        if (epoch + 1) % GENERATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                # 生成文本
                x = torch.LongTensor([[vocabulary.w2i['_BOS']]] * 1).to(device)
                
                for i in range(MAX_GENERATE_LENGTH):
                    output = model(x)
                    logits = output.logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    x = torch.cat([x, next_token], dim=1)
                
                x = x.cpu().numpy()
            
            print('-----------------------------------------------------')
            print('生成的DNA序列示例:')
            for i in range(x.shape[0]):
                # 修改为用空格分隔每个token
                output_text = ' '.join([vocabulary.i2w[_] for _ in list(x[i, :]) if _ not in
                                [vocabulary.w2i['_BOS'], vocabulary.w2i['_EOS'], vocabulary.w2i['_PAD']]])
                print(output_text)
            print('-----------------------------------------------------')
    
    logger.info(f"训练完成: {file_path} - {code_name}")
    
    # 记录训练结果到总日志文件
    log_dir = os.path.join(rootPath, 'log')
    os.makedirs(log_dir, exist_ok=True)
    dataset_name = os.path.basename(file_path).split('_')[0]
    summary_log_path = os.path.join(log_dir, f'{dataset_name}_training.log')
    
    with open(summary_log_path, 'a', encoding='utf-8') as f:
        dataset_name = os.path.basename(file_path).split('_')[0]
        f.write(f"SL: {sl}, Repeat: {repeat}, 最佳损失: {best_loss:.6f}, 学习率: {learning_rate}\n")
    
    return model

if __name__ == "__main__":
    # 创建日志目录
    log_dir = os.path.join(rootPath, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    # 数据集列表
    datasets = ['ASM141792v1']
    
    # sl值列表
    sl_values = ['3']
    # sl_values = ['3', '4', '5', '6']
    
    # 学习率列表
    learning_rates = [0.0003]
    # learning_rates = [0.0002, 0.0003, 0.0004, 0.0005]
    
    # 重复次数
    repeats = 3
    
    # 基本参数设置
    code_name = "xlstm"               # 模型代码名称
    save_path = os.path.join(rootPath, 'checkpoints')
    mode = "train"                     # 训练模式
    running_mode = "s"                # 运行模式: 't'=测试, 's'=开始训练, 'c'=继续训练
    
    # 循环处理每个数据集
    for dataset in datasets:
        save_path = os.path.join(rootPath, 'checkpoints', f'{dataset}')
        print(f"处理数据集: {dataset}")
        
        # 循环处理每个sl值
        for sl in sl_values:
            # 根据sl值确定序列长度
            seq_length = 198 if sl in ['1', '3', '6'] else 200
            print(f"  处理序列长度: {seq_length}, sl值: {sl}")
            
            # 构建文件路径
            file_path = os.path.join(rootPath, f'Dataset/{dataset}_{seq_length}_{sl}.txt')
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"  警告: 文件 {file_path} 不存在，跳过此配置")
                continue
            
            # 循环处理每个学习率
            for learning_rate in learning_rates:
                print(f"    使用学习率: {learning_rate}")
                
                # 循环进行多次重复训练
                for repeat_num in range(1, repeats + 1):
                    repeat = str(repeat_num)
                    print(f"      开始第 {repeat} 次重复训练")
                    
                    # 构建模型保存路径 - 添加学习率信息到路径
                    model_save_path = os.path.join(save_path, f'SL_{sl}', f'SLR_{learning_rate}', f'Repeat_{repeat}')
                    os.makedirs(model_save_path, exist_ok=True)
                    
                    # 调用训练函数
                    try:
                        trained_model = train_xlstm(
                            file_path=file_path,
                            code_name=code_name,
                            seq_length=seq_length,
                            sl=sl,
                            save_path=save_path,
                            mode=mode,
                            repeat=repeat,
                            learning_rate=learning_rate,
                            running_mode=running_mode
                        )
                        
                        # 清理冗余模型
                        removeModel(model_save_path)
                        print(f"      第 {repeat} 次重复训练完成")
                    except Exception as e:
                        print(f"      训练出错: {str(e)}")
                        continue
                    
                    # 释放GPU内存
                    if trained_model is not None:
                        del trained_model
                    torch.cuda.empty_cache()
                
                print(f"    完成学习率: {learning_rate}的所有重复训练")
            
            print(f"  完成sl值: {sl}的所有学习率训练")
        
        print(f"完成数据集: {dataset}的所有训练")
    
    print("所有训练任务完成")