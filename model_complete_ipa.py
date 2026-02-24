"""
IPA 기반 완전한 TTS 모델
IPA → GPT-2 → Audio tokens → Decoder → Mel → iSTFT Vocoder → Waveform
"""
import torch
from torch import nn
from torch.nn import functional as F

from models import Multiband_iSTFT_Generator
from ipa_gpt2_model import IPAGPT2TTS


class CompleteTTS_IPA(nn.Module):
    """
    IPA 기반 완전한 TTS 시스템:
    IPA 텍스트 → [GPT-2 구조 LLM] → 오디오 토큰 → [Decoder] → Mel → [iSTFT Vocoder] → 파형
    """
    def __init__(self,
                 n_ipa_vocab=200,
                 n_audio_vocab=1024,
                 n_codebooks=8,
                 # LLM params
                 hidden_size=768,
                 n_layers=12,
                 n_heads=12,
                 # Vocoder params
                 inter_channels=192,
                 hidden_channels=192,
                 filter_channels=768,
                 resblock="1",
                 resblock_kernel_sizes=[3, 7, 11],
                 resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
                 upsample_rates=[4, 4],
                 upsample_initial_channel=512,
                 upsample_kernel_sizes=[16, 16],
                 gen_istft_n_fft=16,
                 gen_istft_hop_size=4,
                 subbands=4,
                 n_mel_channels=80,
                 n_speakers=0):

        super().__init__()

        # 1. LLM: IPA 텍스트 → 오디오 토큰
        self.llm = IPAGPT2TTS(
            n_ipa_vocab=n_ipa_vocab,
            n_audio_vocab=n_audio_vocab,
            n_codebooks=n_codebooks,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_speakers=n_speakers
        )

        # 2. Decoder: 오디오 토큰 → Mel
        self.decoder = nn.Sequential(
            nn.Linear(n_audio_vocab, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, n_mel_channels)
        )

        # 3. Vocoder: Mel → 파형 (Multiband iSTFT Generator)
        self.vocoder = Multiband_iSTFT_Generator(
            initial_channel=n_mel_channels,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gen_istft_n_fft=gen_istft_n_fft,
            gen_istft_hop_size=gen_istft_hop_size,
            subbands=subbands
        )

        self.n_mel_channels = n_mel_channels

    def forward(self, ipa_input_ids, audio_tokens=None, speaker_id=None):
        """
        Training forward pass

        Args:
            ipa_input_ids: [batch, text_len] - IPA 토큰
            audio_tokens: [batch, n_codebooks, audio_len] - 오디오 토큰 (teacher forcing)
            speaker_id: [batch] - 화자 ID

        Returns:
            audio_logits: [batch, seq_len, n_audio_vocab]
            predicted_mel: [batch, n_mel, time]
            generated_audio: [batch, 1, time]
            generated_audio_mb: [batch, subbands, time]
        """
        # 1. LLM: IPA → 오디오 토큰 logits
        audio_logits = self.llm(ipa_input_ids, audio_tokens, speaker_id=speaker_id)

        # 2. 토큰 → Mel (teacher forcing 시 실제 토큰 사용, inference 시 predicted 사용)
        if audio_tokens is not None:
            # Training: 실제 오디오 토큰 사용
            # [batch, n_codebooks, seq_len] → 첫 번째 codebook만 사용
            token_sequence = audio_tokens[:, 0, :]  # [batch, seq_len]

            # One-hot encoding
            token_onehot = F.one_hot(token_sequence.long(), num_classes=self.decoder[0].in_features)
            token_onehot = token_onehot.float()  # [batch, seq_len, n_audio_vocab]
        else:
            # Inference: predicted 토큰 사용
            token_sequence = torch.argmax(audio_logits, dim=-1)  # [batch, seq_len]
            token_onehot = F.one_hot(token_sequence.long(), num_classes=self.decoder[0].in_features)
            token_onehot = token_onehot.float()

        # Decoder: 토큰 → Mel features
        mel_features = self.decoder(token_onehot)  # [batch, seq_len, n_mel]
        mel_features = mel_features.transpose(1, 2)  # [batch, n_mel, seq_len]

        # 3. Vocoder: Mel → 파형
        generated_audio, generated_audio_mb = self.vocoder(mel_features)

        return audio_logits, mel_features, generated_audio, generated_audio_mb

    def inference(self, ipa_input_ids, speaker_id=None, max_audio_len=500, temperature=1.0, eos_threshold=0.5):
        """
        Inference 모드: IPA 텍스트 → 음성 생성

        Args:
            ipa_input_ids: [batch, text_len] - IPA token indices
            speaker_id: [batch] - Speaker ID (optional)
            max_audio_len: Maximum audio tokens to generate
            temperature: Sampling temperature (default: 1.0)
            eos_threshold: EOS probability threshold for early stopping (default: 0.5)

        Returns:
            audio: [batch, 1, time] - Generated waveform
        """
        self.eval()
        with torch.no_grad():
            # 1. LLM: IPA → audio tokens (autoregressive generation)
            audio_tokens = self.llm.generate(
                ipa_input_ids,
                max_length=max_audio_len,
                temperature=temperature,
                speaker_id=speaker_id,
                eos_threshold=eos_threshold
            )  # [batch, n_codebooks, audio_len]

            # 2. Decoder: audio tokens → Mel
            # Use first codebook
            token_sequence = audio_tokens[:, 0, :]  # [batch, audio_len]

            # One-hot encoding
            token_onehot = F.one_hot(token_sequence.long(), num_classes=self.decoder[0].in_features)
            token_onehot = token_onehot.float()  # [batch, audio_len, n_audio_vocab]

            mel_features = self.decoder(token_onehot)  # [batch, audio_len, n_mel]
            mel_features = mel_features.transpose(1, 2)  # [batch, n_mel, time]

            # 3. Vocoder: Mel → 파형
            audio, _ = self.vocoder(mel_features)

        return audio


if __name__ == "__main__":
    from ipa_tokenizer import IPA_VOCAB_SIZE

    model = CompleteTTS_IPA(
        n_ipa_vocab=IPA_VOCAB_SIZE,
        n_audio_vocab=1024,
        n_codebooks=8,
        n_mel_channels=80
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test
    ipa_ids = torch.randint(0, IPA_VOCAB_SIZE, (2, 10))
    audio_tokens = torch.randint(0, 1024, (2, 8, 20))

    audio_logits, mel, audio, audio_mb = model(ipa_ids, audio_tokens)
    print(f"Audio logits: {audio_logits.shape}")
    print(f"Mel: {mel.shape}")
    print(f"Audio: {audio.shape}")
    print(f"Audio MB: {audio_mb.shape}")
