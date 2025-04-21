import torch
import torch.nn as nn
from math import sqrt

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # 实现ProbSparse自注意力机制
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(2), K_sample.transpose(-2, -1)).squeeze()

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def forward(self, queries, keys, values):
        B, L, H, D = queries.shape
        queries = queries.view(B, H, -1, D)
        keys = keys.view(B, H, -1, D)
        values = values.view(B, H, -1, D)

        U_part = self.factor * torch.ceil(torch.log(torch.tensor(L))).int().item()
        u = self.factor * torch.ceil(torch.log(torch.tensor(L))).int().item()

        scores_top, index = self._prob_QK(queries, keys, sample_k=u, n_top=U_part)
        scale = self.scale or 1. / sqrt(D)
        scores_top = scores_top * scale
        
        if self.mask_flag:
            ones = torch.ones_like(scores_top)
            mask = torch.tril(ones)
            scores_top = scores_top.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores_top, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, values)
        return context, attn

class Informer(nn.Module):
    def __init__(self, config):
        super(Informer, self).__init__()
        # 增强编码器结构
        self.encoder = nn.Sequential(
            nn.Linear(config['enc_in'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )
        
        # 改进的解码器结构（支持长序列输出）
        self.decoder = nn.Sequential(
            nn.Linear(config['dec_in'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.TransformerDecoderLayer(
                d_model=config['d_model'],
                nhead=config['n_heads'],
                dim_feedforward=config['d_ff'],
                dropout=config['dropout']
            )
        )
        
        # 增强注意力机制
        self.prob_attn = ProbAttention(
            mask_flag=True,
            factor=5,
            scale=None,
            attention_dropout=config['dropout']
        )
        
        # 时间序列输出适配层
        self.time_series_adapter = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']*2),
            nn.GELU(),
            nn.Linear(config['d_model']*2, config['c_out'])
        )
        
    def forward(self, x_enc, x_dec):
        enc_out = self.encoder(x_enc)
        dec_out = self.decoder(x_dec)
        
        # 改进的注意力交互
        attn_out, _ = self.prob_attn(enc_out, dec_out, dec_out)
        
        # 多尺度特征融合
        combined = torch.cat([attn_out, dec_out], dim=-1)
        output = self.time_series_adapter(combined)
        return output