"""
IPA 기반 GPT-2 TTS 모델
GPT-2 구조 사용하되 IPA vocab으로 학습
"""
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model


class IPAGPT2TTS(nn.Module):
    """
    IPA 기반 TTS with GPT-2 구조
    IPA 텍스트 → 오디오 토큰
    """
    def __init__(self,
                 n_ipa_vocab=200,  # IPA vocabulary size
                 n_audio_vocab=1024,  # EnCodec vocab
                 n_codebooks=8,
                 hidden_size=768,
                 n_layers=12,
                 n_heads=12,
                 n_speakers=0):

        super().__init__()

        self.n_ipa_vocab = n_ipa_vocab
        self.n_audio_vocab = n_audio_vocab
        self.n_codebooks = n_codebooks
        self.hidden_size = hidden_size
        self.n_speakers = n_speakers

        # Load pretrained GPT-2
        print(f"Loading pretrained GPT-2 from: ./pretrained_llm/gpt2")
        self.gpt2 = GPT2Model.from_pretrained("./pretrained_llm/gpt2", local_files_only=True)

        # Enable gradient checkpointing to save memory
        self.gpt2.gradient_checkpointing_enable()

        # Replace embedding layer for IPA vocab
        original_vocab_size = self.gpt2.wte.weight.size(0)
        print(f"Replacing embedding: {original_vocab_size} → {n_ipa_vocab} (IPA vocab)")

        # New embedding layer
        self.gpt2.wte = nn.Embedding(n_ipa_vocab, hidden_size)

        # Initialize with small random values
        nn.init.normal_(self.gpt2.wte.weight, mean=0.0, std=0.02)

        print(f"✅ Loaded pretrained GPT-2 with IPA embedding layer")
        print(f"🔓 All parameters trainable + gradient checkpointing enabled")

        # Audio token embeddings (8 codebooks)
        self.audio_embs = nn.ModuleList([
            nn.Embedding(n_audio_vocab + 1, hidden_size)
            for _ in range(n_codebooks)
        ])

        # Speaker embedding (optional)
        if n_speakers > 0:
            self.speaker_emb = nn.Embedding(n_speakers, hidden_size)
        else:
            self.speaker_emb = None

        # Output head for audio tokens
        self.audio_head = nn.Linear(hidden_size, n_audio_vocab)

    def forward(self, ipa_input_ids, audio_tokens=None, speaker_id=None, attention_mask=None):
        """
        Args:
            ipa_input_ids: [batch, text_len] - IPA token indices
            ...
        """
        batch_size = ipa_input_ids.size(0)
        text_len = ipa_input_ids.size(1)

        # IPA text embedding (GPT-2's wte)
        text_emb = self.gpt2.wte(ipa_input_ids)  # [batch, text_len, hidden]

        # ==========================================
        # [수정된 부분] Attention Mask 자동 생성 로직 추가
        # ==========================================
        if attention_mask is None:
            # 패딩 토큰(0)을 제외한 실제 데이터 위치만 1로 마스킹
            text_mask = (ipa_input_ids != 0).long()
            
            if audio_tokens is not None:
                audio_mask = (audio_tokens[:, 0, :] != 0).long()
                attention_mask = torch.cat([text_mask, audio_mask], dim=1)
            else:
                attention_mask = text_mask
                
        # GPT-2 모델의 내부 형태에 맞게 4D 마스크로 변환 [batch, 1, 1, seq_len]
        attention_mask = attention_mask[:, None, None, :]
        # ==========================================

        if audio_tokens is not None:
            # Training: teacher forcing
            audio_len = audio_tokens.size(2)
            audio_emb_list = []
            for i in range(self.n_codebooks):
                audio_emb_list.append(self.audio_embs[i](audio_tokens[:, i, :]))
            audio_emb = torch.stack(audio_emb_list, dim=0).mean(dim=0)

            # Concatenate text and audio
            combined_emb = torch.cat([text_emb, audio_emb], dim=1)
        else:
            combined_emb = text_emb

        # (이하 기존 코드 동일...)
        if self.speaker_emb is not None and speaker_id is not None:
            speaker_emb = self.speaker_emb(speaker_id).unsqueeze(1)
            combined_emb = combined_emb + speaker_emb

        seq_len = combined_emb.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=combined_emb.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_emb = self.gpt2.wpe(position_ids)

        hidden_states = combined_emb + position_emb
        hidden_states = self.gpt2.drop(hidden_states)

        # Pass through transformer blocks (마스크 적용됨)
        for block in self.gpt2.h:
            outputs = block(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]

        hidden_states = self.gpt2.ln_f(hidden_states)
        audio_logits = self.audio_head(hidden_states)

        return audio_logits

    def generate(self, ipa_input_ids, max_length=500, temperature=1.0, speaker_id=None, eos_token=0, eos_threshold=0.5):
        """
        Autoregressive generation

        Args:
            ipa_input_ids: [batch, text_len]
            max_length: Maximum audio tokens to generate
            temperature: Sampling temperature
            speaker_id: [batch] - Speaker ID
            eos_token: End-of-sequence token (default: 0)
            eos_threshold: Probability threshold for EOS stopping (default: 0.5)

        Returns:
            generated_tokens: [batch, n_codebooks, generated_len]
        """
        self.eval()
        batch_size = ipa_input_ids.size(0)
        device = ipa_input_ids.device

        # Start with empty audio tokens
        generated = torch.zeros(batch_size, self.n_codebooks, 0, dtype=torch.long, device=device)

        with torch.no_grad():
            for step in range(max_length):
                # Forward pass
                logits = self.forward(
                    ipa_input_ids,
                    generated if generated.size(2) > 0 else None,
                    speaker_id=speaker_id
                )

                # Get last position logits
                next_token_logits = logits[:, -1, :] / temperature  # [batch, vocab]

                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

                # Expand to all codebooks (for simplicity, use same token)
                next_token_expanded = next_token.unsqueeze(1).expand(-1, self.n_codebooks, -1)

                # Append
                generated = torch.cat([generated, next_token_expanded], dim=2)

                # Early stopping based on EOS probability
                if step > 20:
                    eos_prob = probs[:, eos_token]
                    if (eos_prob > eos_threshold).all():
                        print(f"Early stopping at step {step+1}: EOS prob = {eos_prob.mean().item():.3f}")
                        break

                    # Also stop if EOS token is sampled
                    if (next_token == eos_token).all():
                        print(f"Early stopping at step {step+1}: EOS token sampled")
                        break

        return generated


if __name__ == "__main__":
    # Test
    from ipa_tokenizer import IPA_VOCAB_SIZE

    model = IPAGPT2TTS(
        n_ipa_vocab=IPA_VOCAB_SIZE,
        n_audio_vocab=1024,
        n_codebooks=8,
        hidden_size=768,
        n_layers=12,
        n_heads=12
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test forward
    ipa_ids = torch.randint(0, IPA_VOCAB_SIZE, (2, 10))
    audio_tokens = torch.randint(0, 1024, (2, 8, 20))

    logits = model(ipa_ids, audio_tokens)
    print(f"Output shape: {logits.shape}")

    # Test generation
    generated = model.generate(ipa_ids, max_length=50)
    print(f"Generated shape: {generated.shape}")
