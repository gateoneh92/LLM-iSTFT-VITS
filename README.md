# 🌸 LLM-iSTFT-VITS

Next-Generation Lightweight LLM-based Text-to-Speech System.
이 프로젝트는 MB-iSTFT-VITS의 고속 합성 능력과 최신 LLM의 문맥 이해 능력을 결합한 최첨단 TTS 엔진입니다.

---

## 🚀 주요 특징 (Key Features)

- **🧠 LLM-기반 구조**: Transformer를 사용하여 텍스트와 오디오 토큰을 시퀀스로 처리, 자연스러운 감정과 억양 표현.
- **🎧 EnCodec 토큰화**: Meta AI의 EnCodec을 통해 고음질 음성을 압축된 토큰 시퀀스로 변환 및 학습.
- **⚡ 초고속 디코딩**: MB-iSTFT 기술을 채택하여 CPU에서도 실시간보다 수십 배 빠른 음성 생성 가능.
- **📈 전이 학습(Transfer Learning) 지원**: 프리트레인드 디코더 가중치를 로드하여 적은 데이터로도 빠르게 고품질 학습 가능.

---

## 🛠️ 설치 방법 (Installation)

### 1. 가상 환경 설정 및 패키지 설치
```bash
# 가상 환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 필수 라이브러리 설치
pip install -U pip
pip install torch torchaudio encodec numpy scipy matplotlib tensorboard cython librosa phonemizer unidecode
```

### 2. Monotonic Alignment 빌드 (필수)
학습 및 추론을 위해 Cython 코드를 빌드해야 합니다.
```bash
cd monotonic_align
python3 setup.py build_ext --inplace
cd ..
```

---

## 📂 데이터 준비 (Dataset Preparation)

1.  `wavs/` 폴더에 학습용 오디오 파일(.wav, 22050Hz 권장)을 준비합니다.
2.  `filelists/` 폴더에 학습용 목록 파일을 만듭니다. (형식: `파일절대경로|텍스트`)
    *   예: `/path/to/audio.wav|This is a sample sentence.`

---

## 🏋️ 학습 방법 (Training)

학습 시 프리트레인드 디코더(MB-iSTFT-VITS)를 사용하면 훨씬 빠르게 성능을 올릴 수 있습니다.

### 프리트레인드 모델로 시작하기 (추천)
```bash
python3 train_latest.py \
  -c configs/ljs_mb_istft_vits.json \
  -m [모델이름] \
  -p pretrained/pretrained_MB-iSTFT-VITS_ddp.pth
```
*   `-p`: 프리트레인드 디코더 가중치 경로를 지정합니다. (자동으로 디코더 부분만 이식됩니다.)

### 처음부터 학습하기
```bash
python3 train_latest.py -c configs/ljs_mb_istft_vits.json -m [모델이름]
```

---

## 🎙️ 합성 방법 (Inference / Synthesis)

학습된 체크포인트 또는 프리트레인드 모델을 사용하여 음성을 합성합니다.

### 1. 테스트 스크립트 활용
`test_llm_repo.py`를 수정하여 원하는 텍스트를 입력하고 실행합니다.
```bash
python3 test_llm_repo.py
```

### 2. Jupyter Notebook 활용
`inference.ipynb`를 열어 대화형으로 음성을 합성하고 결과를 즉시 확인할 수 있습니다.

---

## 🤝 Acknowledgements
- [MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [Official VITS](https://github.com/jaywalnut310/vits)
- [Meta EnCodec](https://github.com/facebookresearch/encodec)

**Developed for 성웅왕자님 by 정화 🌸**
