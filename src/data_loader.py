import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

class DataLoader:
    def __init__(self, config):
        self.window_size = config['window_size']
        self.pred_len = config['pred_len']
        self.scaler = StandardScaler()
        self.sample_rate = config.get('sample_rate', 15)  # 分钟
        self.hist_years = config.get('hist_years', 3)  # 历史年份

    def load_from_txt(self, file_path):
        # 加载本地txt文件
        df = pd.read_csv(file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']}, 
                       dayfirst=True, dtype={'Global_active_power': 'float'})
        
        # 使用历史平均值填充缺失值
        # df = self._fill_missing_with_historical_mean(df)
        # 使用前一个有效值填充缺失值
        df = df.fillna(method='ffill')

        # 重采样为15分钟间隔
        df = df.set_index('datetime').resample('15T').asfreq().reset_index()
        
        # 筛选目标列
        df = df[['datetime', 'Global_active_power']]
        
        # 调整时间特征生成（适配15分钟粒度）
        df['period'] = df['datetime'].dt.hour * 4 + df['datetime'].dt.minute // 15
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >=5 else 0)
        df['month'] = df['datetime'].dt.month
        
        # 添加滑动窗口统计特征
        df['rolling_24h_mean'] = df['Global_active_power'].rolling(96).mean()
        
        return df.reset_index(drop=True)

    def preprocess(self, data):
        # 单变量数据标准化
        # 调整标准化维度适配新特征
        scaled_data = self.scaler.fit_transform(data[['Global_active_power']].values)
        
        # 改进的滑动窗口处理（支持长序列预测）
        total_len = self.window_size + self.pred_len
        sequences = []
        # 调整步长为15分钟间隔（96步/天）
        # 计算实际时间跨度对应的步长（窗口尺寸*15分钟为总时间跨度）
        step_size = self.pred_len  # 按预测长度（24小时）步长滑动
        for i in range(0, len(data) - total_len + 1, step_size):
            seq = data[i:i+total_len]
            # 确保输入输出维度匹配
            if seq.shape[0] == total_len:
                sequences.append(seq)
        return np.array(sequences)