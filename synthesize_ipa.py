"""
IPA TTS 합성 스크립트
자동 IPA 변환 지원
"""
import torch
import soundfile as sf
from model_complete_ipa import CompleteTTS_IPA
from ipa_tokenizer import IPATokenizer, IPA_VOCAB_SIZE
import utils

try:
    from phonemizer import phonemize
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    print("⚠️  phonemizer not installed. Auto IPA conversion disabled.")
    print("   Install: pip install phonemizer")

def synthesize(text_ipa, checkpoint_path, output_path="output.wav"):
    """
    IPA 텍스트로 음성 합성

    Args:
        text_ipa: IPA 형식의 텍스트 (예: "həˈloʊ wɜːld")
        checkpoint_path: 체크포인트 경로
        output_path: 출력 wav 파일 경로
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    hps = utils.get_hparams_from_file("configs/ipa_tts.json")

    # Tokenizer
    tokenizer = IPATokenizer()

    # Model
    print("Loading model...")
    model = CompleteTTS_IPA(
        n_ipa_vocab=IPA_VOCAB_SIZE,
        n_audio_vocab=1024,
        n_codebooks=8,
        n_mel_channels=hps.data.n_mel_channels,
        **hps.model
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print("✅ Model loaded")

    # Encode text
    print(f"Input IPA: {text_ipa}")
    ipa_tokens = tokenizer.encode(text_ipa, add_special_tokens=True)
    ipa_tensor = torch.LongTensor(ipa_tokens).unsqueeze(0).to(device)  # [1, len]
    print(f"IPA tokens: {ipa_tokens} (length: {len(ipa_tokens)})")

    # Synthesize
    print("Synthesizing...")
    with torch.no_grad():
        audio = model.inference(ipa_tensor)  # [1, 1, time]

    # Save
    audio_np = audio.squeeze().cpu().numpy()
    sf.write(output_path, audio_np, hps.data.sampling_rate)

    print(f"✅ Audio saved to: {output_path}")
    print(f"   Duration: {len(audio_np) / hps.data.sampling_rate:.2f}s")
    print(f"   Sample rate: {hps.data.sampling_rate} Hz")

    return audio_np


def synthesize_text(text, language='en-us', checkpoint_path=None, output_path="output.wav"):
    """
    일반 텍스트를 자동으로 IPA 변환 후 음성 합성

    Args:
        text: 일반 텍스트 (예: "안녕하세요", "Hello world")
        language: 언어 코드
            - 'ko': 한국어
            - 'en-us': 영어 (미국)
            - 'cmn': 중국어 (만다린)
            - 'ja': 일본어
            - 'es': 스페인어
            - 'fr': 프랑스어
            - 'de': 독일어
        checkpoint_path: 체크포인트 경로 (None이면 최신 체크포인트 자동 선택)
        output_path: 출력 wav 파일 경로

    Returns:
        audio_np: 생성된 오디오 numpy array
    """
    if not PHONEMIZER_AVAILABLE:
        raise ImportError(
            "phonemizer is required for auto IPA conversion.\n"
            "Install: pip install phonemizer"
        )

    # 자동으로 최신 체크포인트 찾기
    if checkpoint_path is None:
        import glob
        checkpoints = sorted(glob.glob("ipa_tts/G_step*.pth"), reverse=True)
        if not checkpoints:
            checkpoints = sorted(glob.glob("ipa_tts/G_*.pth"), reverse=True)
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found in ipa_tts/")
        checkpoint_path = checkpoints[0]
        print(f"📂 Using checkpoint: {checkpoint_path}")

    # 텍스트 → IPA 변환
    print(f"\n📝 Original text ({language}): {text}")
    try:
        text_ipa = phonemize(text, language=language, backend='espeak')
        print(f"🔤 IPA conversion: {text_ipa}")
    except Exception as e:
        print(f"❌ IPA conversion failed: {e}")
        print(f"   Make sure espeak is installed and language '{language}' is supported")
        raise

    # IPA 텍스트로 합성
    return synthesize(text_ipa, checkpoint_path, output_path)


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("IPA-Based Multilingual TTS - Synthesis Demo")
    print("=" * 70)

    # 테스트할 텍스트 (한국어 + 영어)
    test_cases = [
        # (텍스트, 언어 코드, 설명)
        ("안녕하세요", "ko", "Korean"),
        ("만나서 반갑습니다", "ko", "Korean"),
        ("Hello world", "en-us", "English"),
        ("How are you today", "en-us", "English"),
    ]

    # 1. IPA 직접 입력 테스트
    print("\n" + "=" * 70)
    print("Test 1: Direct IPA Input")
    print("=" * 70)

    ipa_tests = [
        "həˈloʊ wɜːld",  # "Hello world"
        "ðɪs ɪz ə tɛst",  # "This is a test"
    ]

    import glob
    checkpoints = sorted(glob.glob("ipa_tts/G_step*.pth"), reverse=True)
    if not checkpoints:
        checkpoints = sorted(glob.glob("ipa_tts/G_*.pth"), reverse=True)
    if not checkpoints:
        print("❌ No checkpoints found!")
        sys.exit(1)

    checkpoint_path = checkpoints[0]

    for i, text_ipa in enumerate(ipa_tests):
        output_path = f"output_direct_ipa_{i+1}.wav"
        print(f"\n📢 Synthesizing: {text_ipa}")
        try:
            synthesize(text_ipa, checkpoint_path, output_path)
        except Exception as e:
            print(f"❌ Error: {e}")

    # 2. 자동 IPA 변환 테스트
    if PHONEMIZER_AVAILABLE:
        print("\n" + "=" * 70)
        print("Test 2: Auto IPA Conversion (Multilingual)")
        print("=" * 70)

        for i, (text, lang, desc) in enumerate(test_cases):
            output_path = f"output_auto_{lang}_{i+1}.wav"
            print(f"\n{'='*70}")
            print(f"Language: {desc} ({lang})")
            print(f"{'='*70}")
            try:
                synthesize_text(text, language=lang, output_path=output_path)
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    else:
        print("\n" + "=" * 70)
        print("⚠️  Auto IPA conversion skipped (phonemizer not installed)")
        print("   Install: pip install phonemizer")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("✅ All synthesis tests completed!")
    print("=" * 70)
