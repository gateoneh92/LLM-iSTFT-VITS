"""
IPA 기반 데이터 로더
텍스트를 IPA로 변환하여 로드
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io.wavfile import read

from ipa_tokenizer import IPATokenizer, text_to_ipa
from audio_tokenizer import AudioTokenizer
from mel_processing import spectrogram_torch


class IPATextAudioDataset(Dataset):
    """
    IPA 기반 텍스트-오디오 데이터셋
    """
    def __init__(self, audiopaths_and_text, hparams, offline_mode=True):
        self.audiopaths_and_text = self.load_filepaths_and_text(audiopaths_and_text)
        self.hparams = hparams
        self.offline_mode = offline_mode

        # IPA tokenizer
        self.ipa_tokenizer = IPATokenizer()

        # Audio tokenizer (EnCodec)
        print("오프라인 모드: ./encodec_pretrained에서 모델 로드 중...")
        self.audio_tokenizer = AudioTokenizer(
            device='cpu',
            offline_mode=True,
            repository_path="./encodec_pretrained"
        )

    def load_filepaths_and_text(self, filename):
        with open(filename, encoding='utf-8') as f:
            filepaths_and_text = [line.strip().split('|') for line in f]
        return filepaths_and_text

    def get_audio(self, filename):
        audio, sampling_rate = self.load_wav_to_torch(filename)
        if sampling_rate != self.hparams.data.sampling_rate:
            raise ValueError(f"{sampling_rate} SR doesn't match target {self.hparams.data.sampling_rate} SR")
        audio_norm = audio / self.hparams.data.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm

    def load_wav_to_torch(self, full_path):
        sampling_rate, data = read(full_path)
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate

    def get_text(self, text):
        """영어 텍스트 → IPA → 토큰"""
        # 1. 영어 → IPA 변환
        ipa_text = text_to_ipa(text, language='en')

        # 2. IPA → 토큰
        tokens = self.ipa_tokenizer.encode(ipa_text, add_special_tokens=True)

        return torch.LongTensor(tokens)

    def __getitem__(self, index):
        row = self.audiopaths_and_text[index]

        # Support both 2-column and 3-column format
        if len(row) == 2:
            audiopath, ipa_text = row
        else:
            audiopath, text, ipa_text = row

        # IPA 토큰 (이미 IPA 형식이므로 직접 tokenize)
        ipa_tokens = torch.LongTensor(self.ipa_tokenizer.encode(ipa_text, add_special_tokens=True))

        # Audio tokens (EnCodec) - 파일 경로로 직접 로드
        audio_tokens = self.audio_tokenizer.encode(audiopath)  # [1, n_codebooks, time]
        audio_tokens = audio_tokens.squeeze(0)  # [n_codebooks, time]

        # 오디오 (mel spectrogram용)
        audio = self.get_audio(audiopath)

        # Mel spectrogram
        spec = spectrogram_torch(
            audio,
            self.hparams.data.filter_length,
            self.hparams.data.sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length,
            center=False
        )
        spec = torch.squeeze(spec, 0)

        return (
            ipa_tokens,  # [text_len]
            audio,  # [1, time]
            spec,  # [n_mel, time]
            audio_tokens  # [n_codebooks, time]
        )

    def __len__(self):
        return len(self.audiopaths_and_text)


class IPATextAudioCollate:
    """IPA 데이터 collate function"""

    def __init__(self):
        pass

    def __call__(self, batch):
        # Separate batch
        ipa_tokens_batch = [x[0] for x in batch]
        audio_batch = [x[1] for x in batch]
        spec_batch = [x[2] for x in batch]
        audio_tokens_batch = [x[3] for x in batch]

        # Lengths
        ipa_lengths = torch.LongTensor([len(x) for x in ipa_tokens_batch])
        audio_lengths = torch.LongTensor([x.size(1) for x in audio_batch])
        spec_lengths = torch.LongTensor([x.size(1) for x in spec_batch])
        audio_token_lengths = torch.LongTensor([x.size(1) for x in audio_tokens_batch])

        # Pad sequences
        ipa_tokens_padded = torch.nn.utils.rnn.pad_sequence(
            ipa_tokens_batch, batch_first=True, padding_value=0
        )

        max_audio_len = max([x.size(1) for x in audio_batch])
        audio_padded = torch.zeros(len(batch), 1, max_audio_len)
        for i, audio in enumerate(audio_batch):
            audio_padded[i, :, :audio.size(1)] = audio

        max_spec_len = max([x.size(1) for x in spec_batch])
        n_mel = spec_batch[0].size(0)
        spec_padded = torch.zeros(len(batch), n_mel, max_spec_len)
        for i, spec in enumerate(spec_batch):
            spec_padded[i, :, :spec.size(1)] = spec

        max_audio_token_len = max([x.size(1) for x in audio_tokens_batch])
        n_codebooks = audio_tokens_batch[0].size(0)
        audio_tokens_padded = torch.zeros(len(batch), n_codebooks, max_audio_token_len, dtype=torch.long)
        for i, tokens in enumerate(audio_tokens_batch):
            audio_tokens_padded[i, :, :tokens.size(1)] = tokens

        return (
            ipa_tokens_padded,  # [batch, max_text_len]
            ipa_lengths,  # [batch]
            spec_padded,  # [batch, n_mel, max_spec_len]
            spec_lengths,  # [batch]
            audio_padded,  # [batch, 1, max_audio_len]
            audio_lengths,  # [batch]
            audio_tokens_padded,  # [batch, n_codebooks, max_token_len]
            audio_token_lengths  # [batch]
        )


if __name__ == "__main__":
    # Test
    import utils

    hps = utils.get_hparams_from_file("configs/complete_tts.json")

    dataset = IPATextAudioDataset(
        "filelists/ljs_audio_text_train_filelist.txt.cleaned",
        hps
    )

    print(f"Dataset size: {len(dataset)}")

    # Test one sample
    sample = dataset[0]
    print(f"IPA tokens shape: {sample[0].shape}")
    print(f"Audio shape: {sample[1].shape}")
    print(f"Spec shape: {sample[2].shape}")
    print(f"Audio tokens shape: {sample[3].shape}")

    # Test dataloader
    from torch.utils.data import DataLoader

    collate_fn = IPATextAudioCollate()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

    batch = next(iter(loader))
    print(f"\nBatch:")
    for i, item in enumerate(batch):
        print(f"  {i}: {item.shape}")
