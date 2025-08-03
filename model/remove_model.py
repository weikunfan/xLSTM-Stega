import os
def removeModel(path):
    # 存储所有模型文件的路径
    File = []
    for root, dirs, files in os.walk(path):
        for file in files:
            File.append(os.path.join(root, file))

    # 过滤出 .pkl 模型文件
    Models = [f for f in File if f.endswith('.pkl')]
    
    # 记录所有文件的名称和对应的信息
    model_info = []  # 存储每个文件的 (模型名称, epoch 数, 损失值) 元组
    for m in Models:
        m_name = os.path.basename(m)[:-4]  # 去掉文件扩展名
        parts = m_name.split('-')
        
        # 检查文件名是否符合 'modelname-experiment-epoch-loss' 的格式
        try:
            if len(parts) >= 4: # 检查是否包含损失值
                base_name = '-'.join(parts[:2])  # 基本名称（模型名称 + 实验编号）
                epoch_num = int(parts[2])  # epoch 数
                loss_val = float(parts[3])  # 提取损失值
                model_info.append((base_name, epoch_num, loss_val, m))
            else:
                # 处理原有格式 'modelname-experiment-epoch'
                base_name = '-'.join(parts[:-1])  # 基本名称（模型名称 + 实验编号）
                epoch_num = int(parts[-1])  # epoch 数
                model_info.append((base_name, epoch_num, float('inf'), m))
        except (ValueError, IndexError):
            print(f"Skipping file {m} due to incompatible format.")
    
    # 按模型名称分组
    models_by_name = {}
    for base_name, epoch_num, loss_val, filepath in model_info:
        if base_name not in models_by_name:
            models_by_name[base_name] = []
        models_by_name[base_name].append((epoch_num, loss_val, filepath))
    
    # 对每个模型名称保留最佳模型
    for base_name, epoch_loss_files in models_by_name.items():
        # 如果有损失值，按损失值排序；否则按 epoch 数排序（保留最大的）
        has_loss_values = any(loss < float('inf') for _, loss, _ in epoch_loss_files)
        
        if has_loss_values:
            # 保留损失值最小的模型
            best_model = min(epoch_loss_files, key=lambda x: x[1])
        else:
            # 保留 epoch 数最大的模型
            best_model = max(epoch_loss_files, key=lambda x: x[0])
        
        best_filepath = best_model[2]
        print(f"Keeping model: {os.path.basename(best_filepath)}")
        
        # 删除其他模型
        for _, _, filepath in epoch_loss_files:
            if filepath != best_filepath:
                os.remove(filepath)
                print(f"Removed: {os.path.basename(filepath)}")