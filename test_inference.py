import torch
import commons
import utils
from models import LLMSynthesizer
from text.symbols import symbols
from text import text_to_sequence
import scipy.io.wavfile as wavfile

# 설정 로드
hps = utils.get_hparams_from_file("configs/ljs_mb_istft_vits.json")

# 모델 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_g = LLMSynthesizer(
    len(symbols),
    n_audio_vocab=1024,
    n_codebooks=8,
    inter_channels=hps.model.inter_channels,
    resblock=hps.model.resblock,
    resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
    resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
    upsample_rates=hps.model.upsample_rates,
    upsample_initial_channel=hps.model.upsample_initial_channel,
    upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
    gen_istft_n_fft=hps.model.gen_istft_n_fft,
    gen_istft_hop_size=hps.model.gen_istft_hop_size,
    subbands=hps.model.subbands,
    gin_channels=hps.model.gin_channels
).to(device)

# 체크포인트 로드
utils.load_checkpoint("logs/ljspeech_test/G_0.pth", net_g, None)
net_g.eval()

# 텍스트를 음성으로 변환
text = "Hello world, this is a test."
text_norm = text_to_sequence(text, hps.data.text_cleaners)
text_tensor = torch.LongTensor(text_norm).unsqueeze(0).to(device)

print(f"Generating audio for: {text}")
print(f"Text sequence length: {len(text_norm)}")

with torch.no_grad():
    # LLM으로 오디오 토큰 생성 (max 50 토큰 - 짧게)
    audio_tensor, audio_mb = net_g.infer(text_tensor, max_len=50)

print(f"Generated audio shape: {audio_tensor.shape}")

# 오디오 저장
audio = audio_tensor.squeeze().cpu().numpy()
wavfile.write("output_test.wav", hps.data.sampling_rate, (audio * 32768.0).astype('int16'))
print("Audio saved to output_test.wav")
