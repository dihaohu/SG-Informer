import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, config):
        self.window_size = config['window_size']
        self.pred_len = config['pred_len']
        self.scaler = StandardScaler()

    def load_data(self):
        # 从UCI仓库获取数据集
        dataset = fetch_ucirepo(id=235)
        
        # 合并特征和目标值
        df = pd.concat([
            dataset.data.features.rename(columns=lambda x: x.replace(' ', '_')),
            dataset.data.targets.rename(columns=lambda x: x.replace(' ', '_'))
        ], axis=1)
        
        # 创建时间序列索引
        df['datetime'] = pd.date_range(start='2006-12-16', periods=len(df), freq='10T')
        
        # 添加时间特征
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >=5 else 0)
        
        # 创建滑动窗口统计特征
        df['Global_active_power'] = pd.to_numeric(df['Global_active_power_avg'], errors='coerce')
        df['rolling_24h_mean'] = df['Global_active_power'].rolling(144).mean()
        
        return df[['datetime', 'Global_active_power', 'hour', 
                 'dayofweek', 'is_weekend', 'rolling_24h_mean']].dropna().reset_index(drop=True)

    def preprocess(self, data):
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