import torch
import sys
import os

# MB-iSTFT-VITS 경로 추가
sys.path.append(os.path.join(os.getcwd(), "MB-iSTFT-VITS"))

from models import LLMSynthesizer
from text.symbols import symbols
import utils

def generate_voice_test(text, output_path="generated_voice.wav"):
    print(f"--- 🌸 '{text}' 생성 시도 중 ---")
    
    device = "cpu"
    
    # 1. 모델 설정 (config에서 불러오는 것이 원칙이나 테스트를 위해 직접 설정)
    # (주의: 실제 고음질을 위해서는 학습된 가중치 파일(.pth)이 필요합니다)
    n_text_vocab = len(symbols)
    n_audio_vocab = 1024
    n_codebooks = 8
    
    model_params = {
        "inter_channels": 192,
        "resblock": "1",
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
        "upsample_rates": [4,4],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16,16],
        "gen_istft_n_fft": 16,
        "gen_istft_hop_size": 4,
        "subbands": 4,
        "gin_channels": 0
    }
    
    # 2. 모델 초기화
    net_g = LLMSynthesizer(n_text_vocab, n_audio_vocab, n_codebooks, **model_params).to(device)
    
    # 3. 텍스트 토큰화 (간단히 처리)
    # 실제로는 text_to_sequence를 써야 하지만, 구조 확인을 위해 랜덤하게 생성해 봅니다.
    x = torch.randint(0, n_text_vocab, (1, len(text))).to(device)
    
    # 4. 음성 생성 (Inference)
    # 현재는 학습된 데이터가 없으므로 모델 내부의 랜덤한 '초기 가중치'에 의해 소리가 만들어집니다.
    print("소리 파형을 계산하고 있습니다...")
    with torch.no_grad():
        # 임의의 시작 오디오 토큰 (Reference 없이 생성하는 예시)
        o, o_mb = net_g.infer(x)
    
    # 5. 파일 저장
    # scipy를 이용해 wav 저장
    from scipy.io import wavfile
    import numpy as np
    
    # 파형 데이터를 16비트 정수형으로 변환
    audio_data = o.squeeze().cpu().numpy()
    audio_data = (audio_data * 32767).astype(np.int16)
    
    wavfile.write(output_path, 22050, audio_data)
    print(f"✅ 생성 완료! 저장 위치: {os.path.abspath(output_path)}")
    print("⚠️  주의: 아직 학습되지 않은 모델이라 노이즈나 기계음이 들릴 수 있습니다.")

if __name__ == "__main__":
    generate_voice_test("Hello, Prince Seongwoong!")
