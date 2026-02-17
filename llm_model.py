import torch
from torch import nn
from torch.nn import functional as F
import math

class AudioTextTransformer(nn.Module):
    def __init__(self, 
                 n_text_vocab, 
                 n_audio_vocab, 
                 n_codebooks=8,
                 n_speakers=0, # 추가: 멀티 스피커 지원
                 d_model=512, 
                 nhead=8, 
                 num_layers=12, 
                 dim_feedforward=2048, 
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_codebooks = n_codebooks
        
        # 텍스트 및 오디오 토큰 임베딩
        self.text_emb = nn.Embedding(n_text_vocab, d_model)
        self.audio_embs = nn.ModuleList([nn.Embedding(n_audio_vocab + 1, d_model) for _ in range(n_codebooks)])
        
        # 추가: 스피커 임베딩
        if n_speakers > 0:
            self.spk_emb = nn.Embedding(n_speakers, d_model)
        else:
            self.spk_emb = None

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # GPT 스타일의 Decoder-only Transformer
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.audio_head = nn.Linear(d_model, n_audio_vocab)
        
    def forward(self, text_tokens, audio_tokens, sid=None):
        # text_tokens: [batch, text_len]
        # audio_tokens: [batch, n_codebooks, audio_len]
        # sid: [batch] (Speaker ID)
        
        t_emb = self.text_emb(text_tokens) 
        
        a_emb = 0
        for i in range(self.n_codebooks):
            a_emb += self.audio_embs[i](audio_tokens[:, i, :])
        
        # 입력 결합: [Text] + [Audio]
        x = torch.cat([t_emb, a_emb], dim=1)
        
        # 스피커 정보가 있다면 모든 토큰에 더해줌 (Global Conditioning)
        if self.spk_emb is not None and sid is not None:
            s_emb = self.spk_emb(sid).unsqueeze(1) # [batch, 1, d_model]
            x = x + s_emb
            
        x = self.pos_encoder(x)
        
        # Causal mask 생성
        sz = x.size(1)
        mask = torch.triu(torch.ones(sz, sz, device=x.device) * float('-inf'), diagonal=1)
        
        # Transformer 통과
        output = self.transformer(x, x, tgt_mask=mask)
        
        # 오디오 토큰 부분만 추출해서 예측
        audio_logits = self.audio_head(output[:, text_tokens.size(1):-1, :])
        
        return audio_logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)
