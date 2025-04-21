import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

class DataLoader:
    def __init__(self, config):
        self.window_size = config['window_size']
        self.pred_len = config['pred_len']
        self.scaler = StandardScaler()

    def load_from_txt(self, file_path):
        # 加载本地txt文件
        df = pd.read_csv(file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']}, 
                       dayfirst=True, dtype={'Global_active_power': 'float'})
        
        # 处理缺失值并筛选目标列
        df = df[['datetime', 'Global_active_power']].dropna(subset=['Global_active_power'])
        
        # 创建时间特征
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >=5 else 0)
        
        # 添加滑动窗口统计特征
        df['rolling_24h_mean'] = df['Global_active_power'].rolling(144).mean()
        
        return df.reset_index(drop=True)

    def preprocess(self, data):
        # 单变量数据标准化
        scaled_data = self.scaler.fit_transform(data[['Global_active_power']].values)
        
        # 改进的滑动窗口处理（支持长序列预测）
        total_len = self.window_size + self.pred_len
        sequences = []
        # 按分钟级时间步长滑动（原数据集采样间隔10分钟）
        step_size = 144  # 24小时/10分钟=144步
        for i in range(0, len(data) - total_len + 1, step_size):
            seq = data[i:i+total_len]
            # 确保输入输出维度匹配
            if seq.shape[0] == total_len:
                sequences.append(seq)
        return np.array(sequences)