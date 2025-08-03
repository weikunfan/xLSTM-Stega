import os
import sys
import torch

# 添加上级目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入训练和隐写模块
from model.xlstm_training import train_xlstm
from model.stega_adg import main_stega
from model.remove_model import removeModel

# 定义根路径
rootPath = "/home/fan/Code4idea/xLSTMstega"

def count_lines(file_path):
    """计算文件的总行数"""
    with open(file_path, 'r') as f:
        line_count = sum(1 for line in f)
    return line_count

def main():
    """主函数，协调训练和隐写过程"""
    # 创建日志目录
    log_dir = os.path.join(rootPath, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置参数
    run_mode = "train"  # 运行模式: "train"=仅训练, "stega"=仅隐写, "both"=训练后隐写
    # datasets = ['ASM141792v1']  # 要处理的数据集列表
    datasets = ['ASM141792v1', "ASM286374v1", 'ASM400647v1', 'ASM949793v1']  # 要处理的数据集列表
    # sl_values = ["1", "2", "3", "4", "5", "6"]  # 要处理的sl值列表
    sl_values = ["3", "4", "5", "6"]  # 要处理的sl值列表
    # sl_values = ["3"]  # 要处理的sl值列表
    learning_rates = [0.0004]  # 要使用的学习率列表
    repeats = 3  # 每个配置的重复次数
    
    # 训练参数
    code_name = "slstm"  # 模型代码名称
    save_path = os.path.join(rootPath, 'checkpoints')
    mode = "train"  # 训练模式
    running_mode = "s"  # 训练运行模式: 't'=测试, 's'=开始训练, 'c'=继续训练
    
    # 隐写参数
    debug = False  # 是否开启调试模式
    
    # 循环处理每个数据集
    for dataset in datasets:
        dataset_save_path = os.path.join(save_path, f'{dataset}')
        print(f"处理数据集: {dataset}")
        
        # 循环处理每个sl值
        for sl in sl_values:
            # 根据sl值确定序列长度
            seq_length = 198 if sl in ['1', '3', '6'] else 200
            print(f"  处理序列长度: {seq_length}, sl值: {sl}")
            
            # 构建文件路径
            file_path = os.path.join(rootPath, f'Dataset/{dataset}_{seq_length}_{sl}.txt')
            lines = count_lines(file_path)
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"  警告: 文件 {file_path} 不存在，跳过此配置")
                continue
            
            # 训练模型
            if run_mode in ["train", "both"]:
                # 循环处理每个学习率
                for learning_rate in learning_rates:
                    print(f"    使用学习率: {learning_rate}")
                    
                    # 循环进行多次重复训练
                    for repeat_num in range(1, repeats + 1):
                        repeat = str(repeat_num)
                        print(f"      开始第 {repeat} 次重复训练")
                        
                        # 构建模型保存路径 - 添加学习率信息到路径
                        model_save_path = os.path.join(dataset_save_path, f'SL_{sl}', f'SLR_{learning_rate}', f'Repeat_{repeat}')
                        os.makedirs(model_save_path, exist_ok=True)
                        
                        # 调用训练函数
                        try:
                            trained_model = train_xlstm(
                                file_path=file_path,
                                code_name=code_name,
                                seq_length=seq_length,
                                sl=sl,
                                save_path=dataset_save_path,
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
                
                print(f"  完成sl值: {sl}的所有训练")
            
            # 进行隐写
            if run_mode in ["stega", "both"]:
                print(f"  开始sl值: {sl}的隐写过程")
                
                # 循环进行多次重复隐写
                for repeat_num in range(1, repeats + 1):
                    repeat = str(repeat_num)
                    print(f"      开始第 {repeat} 次隐写")
                    
                    # 对每个学习率都进行隐写
                    for learning_rate in learning_rates:
                        print(f"        使用学习率: {learning_rate}的模型进行隐写")
                        
                        # 调用隐写函数
                        try:
                            main_stega(
                                file_path=file_path,
                                code_name=code_name,
                                seq_length=seq_length,
                                sl=sl,
                                save_path=dataset_save_path,
                                mode=mode,
                                repeat=repeat,
                                # num_rows=lines,
                                num_rows=1000,
                                debug=debug,
                                learning_rate=learning_rate
                            )
                            print(f"        学习率 {learning_rate} 的第 {repeat} 次隐写完成")
                        except Exception as e:
                            print(f"        隐写出错: {str(e)}")
                            continue
                        
                        # 释放GPU内存
                        torch.cuda.empty_cache()
                    
                    print(f"      完成第 {repeat} 次隐写")
                
                print(f"  完成sl值: {sl}的所有隐写")
        
        print(f"完成数据集: {dataset}的所有处理")
    
    print("所有任务完成")

if __name__ == "__main__":
    main()