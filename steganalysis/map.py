# 使用列表明确指定要处理的 k 值
for k in [1, 2]:  # 直接列出需要的 k 值
# for k in [3, 4, 5, 6]:  # 直接列出需要的 k 值
    # 映射表的初始值
    mapping = {}
    
    # 根据 k 的值生成映射表
    bases = ["A", "C", "G", "T"]
    count = 3  # 从3开始编号
    
    # 生成所有可能的k-mer组合
    for i in range(len(bases)**k):
        key = ''.join([bases[(i // (len(bases)**j)) % len(bases)] for j in range(k)])
        mapping[key] = count
        count += 1
    
    # 创建对应的目录（如果不存在）
    import os
    os.makedirs(f"./k_{k}", exist_ok=True)
    
    # 将映射表写入文件
    with open(f"./k_{k}/DNA_map_k_{k}.txt", "w") as f:
        for key, value in mapping.items():
            f.write(f"{key}\t{value}\n")
    
    print(f"Generated mapping for {k}-mer with {len(mapping)} combinations")