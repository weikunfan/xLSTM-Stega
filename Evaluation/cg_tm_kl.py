import numpy
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import scipy.stats
import math
dictComplement = {}
dictComplement['A'] = 'T'
dictComplement['T'] = 'A'
dictComplement['C'] = 'G'
dictComplement['G'] = 'C'
def txt_process_sc_FromKl(lines,len_sc,divied):
    ALL = []
    num1 = divied
    temp_out = ''
    for line in lines:
        temp = ''
        temp1 = ''

        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]

        temp = temp[:len_sc]
        temp_out += temp
        if len(temp) == len_sc:
            for i in range(0, len(temp), num1):
                temp1 += temp[i:i + num1] + ' '

            ALL.append(temp1)

    return ALL, temp_out
def txt_process_sc(lines):
    ALL = []

    temp_out = ''
    for line in lines:
        temp = ''
        temp1 = ''
        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]

        # temp = temp[:len_sc]
        # temp_out += temp
        # if len(temp) == len_sc:
        #    for i in range(0, len(temp) - 1, 2):
        #        temp1 += temp[i] + temp[i + 1] + ' '

        #    ALL.append(temp1)


        temp_out += temp
        for i in range(0, len(temp) - 1, 2):
            temp1 += temp[i] + temp[i + 1] + ' '

        ALL.append(temp1)

    # ALL = ALL[beg_sc:end_sc]
    return ALL, temp_out
def txt_process(lines,length,beg,end):
    ALL = []
    temp_out = ''
    for line in lines:
        temp = ''
        temp1 = ''

        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]

        temp = temp[:length]
        temp_out += temp
        if len(temp) == length:
            for i in range(0, len(temp) - 1, 2):
                temp1 += temp[i] + temp[i + 1] + ' '

            ALL.append(temp1)

    ALL = ALL[beg:end]

    return ALL, temp_out

def txt_process_sc_duo(dp_sc,len_sc,beg_sc,end_sc,PADDING,flex,devide_num):
    with open(dp_sc,"r",encoding='gbk') as f1:
        lines = f1.readlines()

    ALL = []
    num = []

    for line in lines:
        temp = ''
        temp1 = ''
        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]
        len1 = len(temp)
        num.append(len1)

        temp = temp[:len_sc]
        if PADDING == True:
            if len(temp) > (len_sc-flex):
                for i in range(0, len(temp), devide_num):
                    temp1 += temp[i :i+ devide_num] + ' '

                ALL.append(temp1)

        else:
            if len(temp) == len_sc:
                for i in range(0, len(temp), devide_num):
                    temp1 += temp[i:i + devide_num] + ' '
                temp = temp1[:len(temp1)-1]
                ALL.append(temp)



    ALL = ALL[beg_sc:end_sc]

    return ALL

def find_bpn(path):
    with open(path,"r") as f1:
        lines = f1.readlines()

    bpn = 0
    flag = 0
    for line in lines:
        temp = line.find("mean:rl") + len('mean:rl')
        if line.find("mean:rl") > 0 :
            bpn = line[temp:-1]
            flag = 1

    if flag == 0:
        BPN = []
        for line in lines:
            temp = line.find("rl:") + len('rl:')
            if line.find("rl:") > 0:
                BPN.append(float(line[temp : -1 ]))

        bpn = np.mean(np.array(BPN))

    return bpn

def str_to_list(lines):
    out = []
    for line in lines:
        line = line.split(" ")
        out += line

    return out

def C_G(line):
    num = 0
    for i in range(len(line)):
        if (line[i] == 'C') or (line[i] == 'G'):
            num += 1

    return num / len(line)

def melting(line):
    dic_temp = {}
    for i in range(len(line)):
        dic_temp[line[i]] = dic_temp.get(line[i],0) + 1

    nG = dic_temp.get('G')
    nA = dic_temp.get('A')
    nC = dic_temp.get('C')
    nT = dic_temp.get('T')

    tm = 64.9 + 41*( (nG + nC -16.4) / (nG + nA + nT + nC) )


    return tm

def CG_b(line_ori,line_sc,len_sc):
    line_sc_, all_sc = txt_process_sc_FromKl(line_sc,len_sc,divied=1)
    line_ori_, all_ori = txt_process_sc_FromKl(line_ori,len_sc,divied=1)

    line_sc = str_to_list(line_sc_)  # 原始生成数据，需要进行处理ori_two.txt
    line_ori = str_to_list(line_ori_)

    CGORI = C_G(line_ori)
    CGSC  = C_G(line_sc)

    CGBias = np.abs( CGORI - CGSC) / CGORI

    return CGBias * 100

def Tmb(line_ori,line_sc,len_sc):
    line_sc_, all_sc = txt_process_sc_FromKl(line_sc, len_sc, divied=1)
    line_ori_, all_ori = txt_process_sc_FromKl(line_ori, len_sc, divied=1)

    line_sc = str_to_list(line_sc_)  # 原始生成数据，需要进行处理ori_two.txt
    line_ori = str_to_list(line_ori_)

    TMORI = melting(line_ori)
    TMSC = melting(line_sc)

    TMBias = np.abs(TMORI - TMSC) / TMORI

    return TMBias * 100
def KL(DICsc,DICori):
    sc = []
    ori = []
    for bases, Pxy in DICsc.items():
        sc.append(Pxy)

    for bases, Pxy in DICori.items():
        ori.append(Pxy)

    ori = ori / np.sum(ori)
    sc = sc / np.sum(sc)
    ZipScore = list(zip(sc,ori))

    KLD = 0
    KLD1 = 0
    for ScScore, OriScore in ZipScore:
        TEMP = OriScore / ScScore
        KLD += -( ScScore * math.log(OriScore / ScScore,math.e ) )
        KLD1 += ScScore * np.log( ScScore / OriScore )

    OUT = scipy.stats.entropy(ori ,sc)
    '''
    x = [0.14285714, 0.04761905, 0.15873016, 0.07936508, 0.15873016, 0.06349206,0.11111111, 0.0952381,  0.12698413, 0.01587302]
    y = [0.0952381 , 0.07936508, 0.15873016, 0.01587302, 0.11111111, 0.14285714, 0.14285714, 0.0952381,  0.03174603, 0.12698413]
    ZIPxy = list(zip(x,y))
    KLD = 0
    KLD1 = 0
    for ScScore, OriScore in ZIPxy:
        TEMP = OriScore / ScScore
        KLD += -(ScScore * math.log(OriScore / ScScore, math.e))
        KLD1 += ScScore * np.log(ScScore / OriScore)
    OUT = scipy.stats.entropy(x, y)
    '''
    return OUT
def SequenceComplement(lines):
    lines_list, line_str = txt_process(lines,length=200,beg=0,end=3300)

    #line_sc = str_to_list(lines_list) # 原始生成数据，需要进行处理ori_two.txt

    #line_sc = line_sc[: len(line_sc) - 1]

    # str_temp_ori = list(''.join(line_ori))

    line_str_tolist = list(line_str)

    linecomplement = []

    for character in line_str_tolist:
        if len(character) == 1:
            complement = dictComplement[character]
            linecomplement.append(complement)
        else:
            complement0 = dictComplement[character[0]]
            complement1 = dictComplement[character[1]]
            temp = complement0 + complement1
            linecomplement.append(temp)

    linecomplement = ''.join(linecomplement)
    return linecomplement,line_str

def pxy_doubleseq(line, sigle, two_base):
    BaseX = ['A', 'T', 'C', 'G']
    BaseY = ['A', 'T', 'C', 'G']
    BaseMartix = []
    DicSingleBaseNum = {}
    DicTwoBaseNum = {}
    DicP = {}

    for x in BaseX:
        for y in BaseY:
            bases = x + y
            BaseMartix.append(bases)

    for base, num in sigle:
        DicSingleBaseNum[base] = num

    for bases, num in two_base:
        DicTwoBaseNum[bases] = num

    for bases in BaseMartix:
        FirstBase = bases[0]
        SecondBase = bases[1]
        FirstBaseComplement = dictComplement[FirstBase]
        SecondBaseComplement = dictComplement[SecondBase]
        basesComplement = str(FirstBaseComplement + SecondBaseComplement)
        fXYDoubleseq = 0.5 * (DicTwoBaseNum[bases] + DicTwoBaseNum[basesComplement])
        fXDoubleseq  = 0.5 * (DicSingleBaseNum[FirstBase] + DicSingleBaseNum[FirstBaseComplement])
        fYDoubleseq  = 0.5 * (DicSingleBaseNum[SecondBase] + DicSingleBaseNum[SecondBaseComplement])

        PXY = (fXYDoubleseq * 2 * len(line)) / (fXDoubleseq * fYDoubleseq)

        DicP[bases] = PXY

    return DicP

def KLDoubleStrand(line_sc,line_ori):
    line_sc_complement,line_sc = SequenceComplement(line_sc)

    line_sc_doublesequence = line_sc + line_sc_complement

    temp = []
    temp.append(line_sc_doublesequence)

    line_sc_doublesplit, _ = txt_process_sc(temp)
    line_sc_doublesplit_list = line_sc_doublesplit[0].split(' ')[ : -1]

    str_temp_sc = list(_)

    singlebase = sorted(collections.Counter(str_temp_sc).items(), key=lambda x: x[1], reverse=True)

    word_distribution_sc = sorted(collections.Counter(line_sc_doublesplit_list).items(), key=lambda x: x[1],
                                  reverse=True)  # 获得文件中各个单词的分布

    sc_pxy = pxy_doubleseq(str_temp_sc, singlebase, word_distribution_sc)

    ############
    line_ori_complement, line_ori = SequenceComplement(line_ori)

    line_ori_doublesequence = line_ori + line_ori_complement

    temp = []
    temp.append(line_ori_doublesequence)

    line_ori_doublesplit, __ = txt_process_sc(temp)
    line_ori_doublesplit_list = line_ori_doublesplit[0].split(' ')[: -1]

    str_temp_ori = list(__)

    singlebase = sorted(collections.Counter(str_temp_ori).items(), key=lambda x: x[1], reverse=True)

    word_distribution_ori = sorted(collections.Counter(line_ori_doublesplit_list).items(), key=lambda x: x[1],
                                  reverse=True)  # 获得文件中各个单词的分布

    ori_pxy = pxy_doubleseq(str_temp_ori, singlebase, word_distribution_ori)

    kl_ = KL(sc_pxy, ori_pxy)

    return kl_
def JSD(DICsc, DICori):
    """
    计算两个概率分布的 Jensen-Shannon 散度
    :param DICsc: 隐写序列的二核苷酸概率分布
    :param DICori: 原始序列的二核苷酸概率分布
    :return: JSD 散度
    """
    sc = []
    ori = []

    # 提取隐写序列和原始序列的概率分布
    for bases, Pxy in DICsc.items():
        sc.append(Pxy)

    for bases, Pxy in DICori.items():
        ori.append(Pxy)

    # 标准化分布
    ori = ori / np.sum(ori)
    sc = sc / np.sum(sc)

    # 中间分布 M 的计算
    M = 0.5 * (np.array(sc) + np.array(ori))

    # 分别计算 KL(P || M) 和 KL(Q || M)
    KL_P_M = scipy.stats.entropy(sc, M)  # KL(sc || M)
    KL_Q_M = scipy.stats.entropy(ori, M)  # KL(ori || M)

    # 计算 JSD
    JSD = 0.5 * KL_P_M + 0.5 * KL_Q_M

    return JSD


def JSDDoubleStrand(line_sc, line_ori):
    """
    计算 DNA 双链的 Jensen-Shannon 散度
    :param line_sc: 隐写序列
    :param line_ori: 原始序列
    :return: JSD 散度
    """
    # Step 1: 对隐写序列生成双链
    line_sc_complement, line_sc = SequenceComplement(line_sc)
    line_sc_doublesequence = line_sc + line_sc_complement

    temp = [line_sc_doublesequence]
    line_sc_doublesplit, _ = txt_process_sc(temp)
    line_sc_doublesplit_list = line_sc_doublesplit[0].split(' ')[:-1]

    str_temp_sc = list(_)

    # 计算隐写序列的单核苷酸分布和二核苷酸分布
    singlebase_sc = sorted(collections.Counter(str_temp_sc).items(), key=lambda x: x[1], reverse=True)
    word_distribution_sc = sorted(collections.Counter(line_sc_doublesplit_list).items(), key=lambda x: x[1], reverse=True)

    sc_pxy = pxy_doubleseq(str_temp_sc, singlebase_sc, word_distribution_sc)

    # Step 2: 对原始序列生成双链
    line_ori_complement, line_ori = SequenceComplement(line_ori)
    line_ori_doublesequence = line_ori + line_ori_complement

    temp = [line_ori_doublesequence]
    line_ori_doublesplit, __ = txt_process_sc(temp)
    line_ori_doublesplit_list = line_ori_doublesplit[0].split(' ')[:-1]

    str_temp_ori = list(__)

    # 计算原始序列的单核苷酸分布和二核苷酸分布
    singlebase_ori = sorted(collections.Counter(str_temp_ori).items(), key=lambda x: x[1], reverse=True)
    word_distribution_ori = sorted(collections.Counter(line_ori_doublesplit_list).items(), key=lambda x: x[1], reverse=True)

    ori_pxy = pxy_doubleseq(str_temp_ori, singlebase_ori, word_distribution_ori)

    # Step 3: 计算 JSD
    jsd_ = JSD(sc_pxy, ori_pxy)

    return jsd_
def split_data(path,dividenum):
    with open(path,"r") as f1:
        lines = f1.readlines()

    temp = ''
    all = ''
    for line in lines:
        for i in range(len(line)):
            if line[i] == 'A' or line[i] == 'T' or line[i] == 'C' or line[i] == 'G':
                temp += line[i]

    stop = len(temp) - len(temp) % dividenum

    temp = temp[ : stop]

    for i in range(0,len(temp),dividenum):
        all += temp[i : i + dividenum] + ' '

    temp = temp[ : 22400]

    l = list(temp)
    out = np.array(list(temp)).reshape((175,-1))

    return all,out

def get_nucleotide_composition(sequence, k=1):
    """计算k-mer核苷酸组成
    
    Args:
        sequence: DNA序列字符串
        k: k-mer长度(1,2或3)
    
    Returns:
        dict: k-mer组成字典
    """
    composition = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        composition[kmer] = composition.get(kmer, 0) + 1
    
    # 转换为频率
    total = sum(composition.values())
    for kmer in composition:
        composition[kmer] = composition[kmer] / total
    
    return composition

def calculate_composition_bias(ori_seq, gen_seq, k=1):
    """计算原始序列和生成序列之间的k-mer组成偏差
    
    Args:
        ori_seq: 原始DNA序列
        gen_seq: 生成的DNA序列
        k: k-mer长度(1,2或3)
    
    Returns:
        float: 相对偏差百分比
    """
    ori_comp = get_nucleotide_composition(ori_seq, k)
    gen_comp = get_nucleotide_composition(gen_seq, k)
    
    # 确保两个字典有相同的键
    all_kmers = set(ori_comp.keys()) | set(gen_comp.keys())
    for kmer in all_kmers:
        ori_comp.setdefault(kmer, 0)
        gen_comp.setdefault(kmer, 0)
    
    # 计算平均相对偏差
    relative_biases = []
    for kmer in all_kmers:
        if ori_comp[kmer] > 0:  # 避免除以零
            relative_bias = abs(ori_comp[kmer] - gen_comp[kmer]) / ori_comp[kmer]
            relative_biases.append(relative_bias)
    
    # 返回平均相对偏差的百分比
    return np.mean(relative_biases) * 100 if relative_biases else 0

def nucleotide_composition_bias(line_ori, line_sc, len_sc):
    """计算所有核苷酸组成偏差指标
    
    Args:
        line_ori: 原始序列
        line_sc: 生成序列
        len_sc: 序列长度
    
    Returns:
        tuple: (单核苷酸偏差%, 双核苷酸偏差%, 三核苷酸偏差%)
    """
    # 处理序列
    line_sc_, all_sc = txt_process_sc_FromKl(line_sc, len_sc, divied=1)
    line_ori_, all_ori = txt_process_sc_FromKl(line_ori, len_sc, divied=1)
    
    line_sc = ''.join(str_to_list(line_sc_))
    line_ori = ''.join(str_to_list(line_ori_))
    
    # 计算各种核苷酸组成偏差
    mono_bias = calculate_composition_bias(line_ori, line_sc, k=1)
    di_bias = calculate_composition_bias(line_ori, line_sc, k=2)
    tri_bias = calculate_composition_bias(line_ori, line_sc, k=3)
    
    return mono_bias, di_bias, tri_bias