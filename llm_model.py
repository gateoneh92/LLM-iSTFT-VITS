import torch
from torch import nn
from torch.nn import functional as F
import math

class AudioTextTransformer(nn.Module):
    def __init__(self, 
                 n_text_vocab, 
                 n_audio_vocab, 
                 n_codebooks=8,
                 n_speakers=0, 
                 n_emotions=0, # 추가: 감정 제어 지원
                 d_model=512, 
                 nhead=8, 
                 num_layers=12, 
                 dim_feedforward=2048, 
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_codebooks = n_codebooks
        
        # 임베딩 레이어들
        self.text_emb = nn.Embedding(n_text_vocab, d_model)
        self.audio_embs = nn.ModuleList([nn.Embedding(n_audio_vocab + 1, d_model) for _ in range(n_codebooks)])
        
        if n_speakers > 0:
            self.spk_emb = nn.Embedding(n_speakers, d_model)
        else:
            self.spk_emb = None

        # 추가: 감정 임베딩
        if n_emotions > 0:
            self.emo_emb = nn.Embedding(n_emotions, d_model)
        else:
            self.emo_emb = None

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # GPT 스타일의 Decoder-only Transformer
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.audio_head = nn.Linear(d_model, n_audio_vocab)
        
    def forward(self, text_tokens, audio_tokens, sid=None, eid=None, use_cache=False, past_key_values=None):
        # eid: [batch] (Emotion ID)
        
        if use_cache and past_key_values is not None:
            audio_tokens = audio_tokens[:, :, -1:]
            
        t_emb = self.text_emb(text_tokens) 
        
        a_emb = 0
        for i in range(self.n_codebooks):
            a_emb += self.audio_embs[i](audio_tokens[:, i, :])
        
        if use_cache and past_key_values is not None:
            x = a_emb
        else:
            x = torch.cat([t_emb, a_emb], dim=1)
        
        # 화자 정보 결합
        if self.spk_emb is not None and sid is not None:
            s_emb = self.spk_emb(sid).unsqueeze(1)
            x = x + s_emb
            
        # 추가: 감정 정보 결합
        if self.emo_emb is not None and eid is not None:
            e_emb = self.emo_emb(eid).unsqueeze(1)
            x = x + e_emb
            
        x = self.pos_encoder(x)
        
        # Causal mask 및 캐시 처리 (실제 구현에서는 Transformer 레이어별 KV 핸들링이 필요)
        # 여기서는 구조적 개념을 통합하는 방식으로 작성합니다.
        sz = x.size(1)
        mask = torch.triu(torch.ones(sz, sz, device=x.device) * float('-inf'), diagonal=1)
        
        output = self.transformer(x, x, tgt_mask=mask)
        
        # 다음 토큰 예측
        audio_logits = self.audio_head(output)
        
        if use_cache:
            # 새로운 캐시를 생성해서 반환 (개념적 예시)
            new_past_key_values = output 
            return audio_logits, new_past_key_values
            
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
