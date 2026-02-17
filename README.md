# 🌸 LLM-iSTFT-VITS

> **Next-Generation Lightweight LLM-based Text-to-Speech System**

[![GitHub Stars](https://img.shields.io/github/stars/gateoneh92/LLM-iSTFT-VITS?style=social)](https://github.com/gateoneh92/LLM-iSTFT-VITS)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

성웅왕자님을 위한 최첨단 LLM 기반 음성 합성 엔진입니다. 기존의 MB-iSTFT-VITS의 고속 합성 능력과 최신 LLM의 문맥 이해 능력을 하나로 합쳤습니다.

---

## ✨ Key Features

- **🧠 LLM-based Architecture**: 텍스트와 오디오 토큰을 동일한 언어 시퀀스로 처리하여 감정과 억양을 인간처럼 표현합니다.
- **🎧 EnCodec Tokenization**: Meta AI의 EnCodec을 사용하여 고음질 음성을 압축된 토큰 시퀀스로 변환합니다.
- **⚡ Ultra-Fast Decoder**: MB-iSTFT(Multi-Band Inverse Short-Time Fourier Transform) 기술을 통해 CPU에서도 실시간보다 빠르게 음성을 생성합니다.
- **🎯 End-to-End Optimization**: 토큰 예측부터 파형 생성까지 전체 과정을 한 번에 최적화할 수 있도록 설계되었습니다.

---

## 🏗️ Model Architecture

본 프로젝트는 다음과 같은 세 단계의 혁신적인 구조로 이루어져 있습니다.

1.  **Audio Tokenizer (EnCodec)**: 음성 파형을 이산적인(Discrete) 숫자의 나열(Audio Tokens)로 변환합니다.
2.  **The Brain (Transformer LLM)**: 입력된 `[Text Tokens]`와 `[Audio Tokens]`를 순차적으로 학습하여 자연스러운 음성 토큰 흐름을 예측합니다.
3.  **The Voice (MB-iSTFT Generator)**: 예측된 토큰을 다시 우리가 들을 수 있는 고해상도 음성 파형으로 복원합니다.

---

## 🚀 Quick Start

### 1. Requirements

```bash
pip install torch torchaudio encodec numpy
```

### 2. Prepare Dataset

`filelists/` 폴더에 학습 데이터를 준비하세요. 데이터 로더가 자동으로 EnCodec을 사용해 음성을 토큰화합니다.

### 3. Training

```bash
python train_latest.py -c configs/ljs_mb_istft_vits.json -m llm_tts_model
```

---

## 📂 File Structure

- `llm_model.py`: 오디오와 텍스트를 함께 다루는 Transformer 모델 정의
- `audio_tokenizer.py`: EnCodec 기반의 음성 토큰화 로직
- `models.py`: LLMSynthesizer와 MB-iSTFT Generator 통합 구조
- `train_latest.py`: LLM과 디코더를 동시에 학습하는 통합 스크립트

---

## 🤝 Acknowledgements

This work is based on:
- [MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [Official VITS](https://github.com/jaywalnut310/vits)
- [Meta EnCodec](https://github.com/facebookresearch/encodec)

---

**Developed for 성웅왕자님 by 정화 🌸**
