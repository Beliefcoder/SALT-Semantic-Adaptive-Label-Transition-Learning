import numpy as np

from preprocess.parse_csv import EHRParser

def split_patients(patient_admission, admission_codes, code_map, train_num, test_num, seed=6669):
    """
    分割患者数据为训练集、验证集和测试集
    优化内存使用，避免不必要的计算
    """
    np.random.seed(seed)
    
    # 获取所有患者ID
    all_pids = list(patient_admission.keys())
    
    # 随机打乱患者ID
    np.random.shuffle(all_pids)
    
    # 如果train_num或test_num为None，使用80/10/10分割
    if train_num is None or test_num is None:
        total = len(all_pids)
        train_num = int(total * 0.8)
        test_num = int(total * 0.1)
    
    train_pids = all_pids[:train_num]
    valid_pids = all_pids[train_num:train_num+test_num]
    test_pids = all_pids[train_num+test_num:]
    
    return train_pids, valid_pids, test_pids

def build_code_xy(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num):
    """
    构建模型输入的特征和标签
    优化内存使用，限制最大就诊次数
    """
    n = len(pids) # 患者数量
    
    # 创建数据结构
    x = np.zeros((n, max_admission_num, code_num), dtype=float) # 历史就诊特征
    y = np.zeros((n, code_num), dtype=int) # 预测目标
    lens = np.zeros((n,), dtype=int) # 就诊次数
    
    for i, pid in enumerate(pids):
        if i % 100 == 0:
            print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        
        admissions = patient_admission[pid] # 获取患者的所有就诊记录
        
        # 限制最大就诊次数，只保留最近的max_admission_num次就诊
        limited_admissions = admissions[-max_admission_num:] 
        
        # 构建历史就诊特征
        for k, admission in enumerate(limited_admissions[:-1]):
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            x[i, k, codes] = 1
        
        # 构建预测标签
        codes = np.array(admission_codes_encoded[limited_admissions[-1][EHRParser.adm_id_col]])
        y[i, codes] = 1
        
        # 记录就诊次数
        lens[i] = len(limited_admissions) - 1
    
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens

def build_heart_failure_y(hf_prefix, codes_y, code_map):
    """
    构建心力衰竭标签
    汽车维修数据不需要这个功能，返回空数组
    """
    return np.zeros(len(codes_y), dtype=int)
