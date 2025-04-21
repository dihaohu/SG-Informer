import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data_loader import DataLoader as DataProcessor
from src.models.informer import Informer
import numpy as np
import os

def main():
    # 加载配置
    with open('config/config.json') as f:
        config = json.load(f)
    
    # 初始化数据处理器
    data_processor = DataLoader(config['train'])
    df = data_processor.load_from_txt(config['data']['path'])
    
    # 数据预处理（包含标准化）
    sequences = data_processor.preprocess(df)
    
    # 划分训练集（80%）、验证集（10%）、测试集（10%）
    split1 = int(0.8 * len(sequences))
    split2 = int(0.9 * len(sequences))
    train_data = torch.FloatTensor(sequences[:split1])
    val_data = torch.FloatTensor(sequences[split1:split2])
    test_data = torch.FloatTensor(sequences[split2:])
    
    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(train_data[:, :config['train']['window_size']], 
                                          train_data[:, -config['train']['pred_len']:]),
                            batch_size=config['train']['batch_size'], shuffle=True)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Informer(config['model']).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config['train']['epochs']):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x.unsqueeze(-1), x.unsqueeze(-1))
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 改进的验证流程（多步滚动预测）
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # 初始化验证输入
            input_seq = val_data[:, :config['train']['window_size']].unsqueeze(-1).to(device)
            total_steps = config['train']['pred_len'] // 144  # 每24小时预测一次
            
            # 多步滚动预测
            predictions = []
            targets = []
            for step in range(total_steps):
                # 预测未来24小时（144个10分钟间隔）
                pred = model(input_seq, input_seq)
                predictions.append(pred.cpu())
                
                # 更新输入序列（保留后window_size个时间步）
                input_seq = torch.cat([input_seq[:, 144:], pred[:, :144]], dim=1)
                
                # 记录真实值
                start_idx = config['train']['window_size'] + step*144
                end_idx = start_idx + 144
                if end_idx <= val_data.shape[1]:
                    targets.append(val_data[:, start_idx:end_idx].to(device))
            
            # 计算整体损失
            predictions = torch.cat(predictions, dim=1)[:, :config['train']['pred_len']]
            targets = torch.cat(targets, dim=1)[:, :config['train']['pred_len']]
            val_loss = criterion(predictions.squeeze(), targets.squeeze())
        
        print(f'Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val Loss {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_informer.pth')
    
    print('Training completed. Best validation loss:', best_val_loss.item())

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    main()