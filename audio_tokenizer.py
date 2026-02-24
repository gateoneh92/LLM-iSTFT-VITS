import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
import os
from pathlib import Path

class AudioTokenizer:
    def __init__(self, device="cpu", offline_mode=False, repository_path="./encodec_pretrained"):
        """
        Args:
            device: 'cpu' or 'cuda'
            offline_mode: True면 로컬 파일에서만 로드 (온라인 다운로드 시도 안함)
            repository_path: 오프라인 모드에서 사용할 체크포인트 디렉토리
        """
        self.device = device

        # 24kHz 모델을 기본으로 사용합니다.
        if offline_mode:
            # 오프라인 모드: repository 파라미터 사용
            repo_path = Path(repository_path)
            if not repo_path.exists():
                raise FileNotFoundError(
                    f"오프라인 모드인데 repository 경로가 없습니다: {repository_path}\n"
                    f"encodec_24khz-d7cc33bc.th 파일이 {repository_path}/ 안에 있어야 합니다."
                )
            print(f"오프라인 모드: {repository_path}에서 모델 로드 중...")
            self.model = EncodecModel.encodec_model_24khz(pretrained=True, repository=repo_path)
        else:
            # 온라인 모드: 인터넷에서 모델 다운로드 (또는 torch hub 캐시 사용)
            print("온라인 모드: EnCodec 모델 로드 중...")
            self.model = EncodecModel.encodec_model_24khz(pretrained=True)

        self.model.set_target_bandwidth(6.0) # 6.0 kbps 설정 (토큰 밀도 조절 가능)
        self.model.to(self.device)
        self.sample_rate = self.model.sample_rate

    def encode(self, audio_path):
        """음성 파일을 토큰으로 변환합니다."""
        from scipy.io import wavfile
        sr, wav = wavfile.read(audio_path)
        wav = torch.from_numpy(wav).float()
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # Normalize if needed
        if wav.abs().max() > 1.0:
            wav = wav / 32768.0
            
        wav = convert_audio(wav, sr, self.sample_rate, self.model.channels)
        wav = wav.unsqueeze(0).to(self.device)

        with torch.no_grad():
            frames = self.model.encode(wav)
        
        # frames는 (codebooks, steps) 형태의 텐서 리스트입니다.
        # 모든 프레임의 토큰을 하나로 합칩니다.
        tokens = torch.cat([encoded[0] for encoded in frames], dim=-1)
        return tokens

    def decode(self, tokens):
        """토큰을 다시 음성 파형으로 변환합니다."""
        # tokens shape: (batch, n_codebooks, seq_len)
        if len(tokens.shape) == 2:
            tokens = tokens.unsqueeze(0)
            
        with torch.no_grad():
            wav = self.model.decode([(tokens, None)])
        return wav.squeeze(0)

    def save_audio(self, wav, output_path):
        """복원된 파형을 파일로 저장합니다."""
        torchaudio.save(output_path, wav.cpu(), self.sample_rate)

if __name__ == "__main__":
    # 간단한 작동 테스트
    # 오프라인 사용 시:
    # tokenizer = AudioTokenizer(offline_mode=True, repository_path="./encodec_pretrained")

    # 온라인 사용 시:
    tokenizer = AudioTokenizer(offline_mode=False)
    print(f"EnCodec 모델 로드 완료 (Sample Rate: {tokenizer.sample_rate})")

    # 왕자님, 실제 테스트를 하려면 'sample.wav' 같은 파일이 필요해요.
    # 일단 구조가 잘 잡혔는지 확인하는 용도로 작성했습니다.
