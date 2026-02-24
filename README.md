# LLM-iSTFT-VITS

**Language Model-based Text-to-Speech with iSTFT Vocoder**

Pretrained GPT-2 기반 고품질 음성 합성 시스템

---

## 🎯 Overview

이 프로젝트는 **Pretrained Language Model (GPT-2)**과 **Neural Audio Codec (EnCodec)**, 그리고 **고속 iSTFT Vocoder**를 결합한 end-to-end TTS 시스템입니다.

### 핵심 아이디어

1. **LLM as Sequence Predictor**: GPT-2의 강력한 sequence modeling 능력을 활용하여 텍스트에서 오디오 토큰으로 직접 매핑
2. **Audio Tokenization**: EnCodec으로 오디오를 discrete tokens로 표현하여 LLM이 처리 가능하게 변환
3. **Fast Waveform Generation**: Multiband iSTFT Generator로 실시간급 고품질 음성 생성
4. **Transfer Learning**: Pretrained GPT-2 weights를 활용하여 빠른 수렴과 안정적인 학습

---

## 🚀 주요 특징

### 모델 아키텍처

- **Pretrained GPT-2 Backbone**: 117M 파라미터 transformer (768 hidden, 12 layers, 12 heads)
- **Audio Codec**: EnCodec (8 codebooks, 1024 vocab, 50Hz frame rate)
- **Neural Vocoder**: Multiband iSTFT Generator (4 subbands)
- **End-to-End**: 텍스트에서 파형까지 단일 모델로 학습

### 기술적 장점

- **Transfer Learning**: GPT-2의 사전학습된 언어 이해 능력 활용
- **Memory Efficient**: Gradient checkpointing으로 12GB GPU에서 학습 가능
- **Fast Inference**: iSTFT 기반 vocoder로 빠른 음성 생성
- **Multilingual Support**: Universal phonetic representation (IPA) 지원
- **Offline Mode**: 모든 모델을 로컬에서 로드 가능

---

## 📊 시스템 구조

```
┌─────────────┐
│ Text Input  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Text Tokenizer     │ (IPA: 131 symbols)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  GPT-2 Transformer  │ (Pretrained, 768-d, 12 layers)
│  + New Embedding    │ (IPA vocab)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Audio Tokens       │ (EnCodec discrete tokens)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Decoder Network    │ (Tokens → Mel features)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  iSTFT Vocoder      │ (Mel → Waveform)
│  (Multiband)        │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│   Output    │ (22.05 kHz audio)
│   Audio     │
└─────────────┘
```

### 학습 과정

```
Ground Truth Audio → EnCodec → Audio Tokens (target)
                                    ↑
Text → GPT-2 → Predicted Logits ──┘ (Cross-entropy loss)
              ↓
         Audio Tokens → Decoder → Mel → Vocoder → Waveform
                                   ↓              ↓
                              Mel Loss       GAN Loss
                              STFT Loss      FM Loss
```

---

## 🔧 Model Specifications

| Component | Specification |
|-----------|--------------|
| **LLM** | GPT-2 (117M params) |
| Hidden Size | 768 |
| Layers | 12 |
| Attention Heads | 12 |
| **Audio Codec** | EnCodec |
| Codebooks | 8 |
| Vocab Size | 1024 |
| Frame Rate | 50 Hz |
| **Vocoder** | Multiband iSTFT |
| Subbands | 4 |
| FFT Size | 16 |
| Hop Size | 4 |
| **Audio** | 22.05 kHz, 80 Mel channels |

---

## 🛠️ Installation

### Requirements

```bash
pip install torch torchaudio
pip install transformers soundfile scipy
pip install phonemizer  # For text preprocessing
```

### Download Pretrained Models

#### 1. GPT-2 Model

```bash
python -c "from transformers import GPT2Model, GPT2Tokenizer; \
    model = GPT2Model.from_pretrained('gpt2'); \
    model.save_pretrained('./pretrained_llm/gpt2'); \
    tok = GPT2Tokenizer.from_pretrained('gpt2'); \
    tok.save_pretrained('./pretrained_llm/gpt2')"
```

#### 2. EnCodec

EnCodec 모델은 `./encodec_pretrained`에 포함되어 있습니다.

---

## 📚 Data Preparation

### Step 1: Prepare Text-Audio Pairs

Filelist 형식:
```
/path/to/audio1.wav|Hello world
/path/to/audio2.wav|How are you
```

### Step 2: Convert Text to Phonemes (IPA)

```bash
# Preview first 5 samples
python3 preprocess_ipa.py -i input.txt -o output.txt -l en-us --preview

# Full conversion
python3 preprocess_ipa.py -i input.txt -o output_ipa.txt -l en-us
```

**Supported Languages:**
- `en-us`: English (US)
- `ko`: Korean
- `ja`: Japanese
- `cmn`: Chinese (Mandarin)
- `es`: Spanish
- `fr`: French
- `de`: German
- More: See [espeak languages](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)

**Output:**
```
/path/to/audio1.wav|həloʊ wɜːld
/path/to/audio2.wav|haʊ ɑːɹ juː
```

---

## 🎓 Training

### Configuration

Edit `configs/ipa_tts.json`:

```json
{
  "train": {
    "log_interval": 10,
    "save_interval": 100,     // Checkpoint save frequency
    "batch_size": 2,          // Adjust based on GPU memory
    "learning_rate": 2e-4,
    "c_llm": 1.0,             // LLM loss weight
    "c_mel": 45,              // Mel reconstruction loss weight
    "c_fm": 2.0,              // Feature matching loss weight
    "c_stft": 1.0             // STFT loss weight
  },
  "model": {
    "hidden_size": 768,       // GPT-2 hidden (fixed)
    "n_layers": 12,           // GPT-2 layers (fixed)
    "n_heads": 12             // GPT-2 heads (fixed)
  }
}
```

### Start Training

```bash
python3 train_ipa.py -c configs/ipa_tts.json -m my_model
```

**Output Structure:**
```
logs/
└── my_model/
    ├── train.log              # Training log
    └── events.out.tfevents.*  # TensorBoard events

checkpoints/
└── my_model/
    ├── G_init.pth            # Initial generator
    ├── D_init.pth            # Initial discriminator
    ├── G_step100.pth         # Generator at step 100
    ├── D_step100.pth         # Discriminator at step 100
    └── ...
```

**Checkpoints:**
- Saved every `save_interval` steps (default: 50)
- Location: `checkpoints/my_model/G_step*.pth`, `D_step*.pth`
- Logs: `logs/my_model/train.log`
- TensorBoard: `tensorboard --logdir logs/`

### Loss Functions

The model is trained with 6 loss functions:

1. **LLM Loss**: Cross-entropy between predicted and target audio tokens
2. **Mel Loss**: L1 loss between predicted and ground truth mel spectrograms
3. **GAN Loss**: Adversarial loss for realistic waveform generation
4. **Feature Matching Loss**: Discriminator feature matching
5. **STFT Loss**: Multi-resolution STFT loss for audio quality
6. **Discriminator Loss**: Real/fake discrimination

**Total Generator Loss:**
```
L_G = c_llm × L_LLM + c_mel × L_Mel + L_GAN + c_fm × L_FM + c_stft × L_STFT
```

---

## 🎤 Inference

### Method 1: Auto Text-to-IPA Conversion

```python
from synthesize_ipa import synthesize_text

# Korean
synthesize_text("안녕하세요", language='ko', output_path="output_ko.wav")

# English
synthesize_text("Hello world", language='en-us', output_path="output_en.wav")
```

### Method 2: Direct IPA Input

```python
from synthesize_ipa import synthesize

ipa_text = "həloʊ wɜːld"
synthesize(ipa_text, checkpoint_path="ipa_tts/G_latest.pth", output_path="output.wav")
```

### Method 3: Command Line

```bash
python3 synthesize_ipa.py
```

**Output:**
- `output_auto_ko_*.wav`: Korean samples
- `output_auto_en-us_*.wav`: English samples

---

## 🔬 Technical Details

### Model Architecture

**1. Text Encoder (GPT-2-based)**
- Pretrained GPT-2 with **replaced embedding layer** for IPA vocab (131 symbols)
- All transformer weights kept from pretraining
- Gradient checkpointing enabled for memory efficiency

**2. Audio Tokenizer (EnCodec)**
- Neural audio codec with 8 codebooks
- Each codebook: 1024 discrete tokens
- Frame rate: 50 Hz (256 hop size at 22.05 kHz)

**3. Decoder Network**
- Projects audio tokens to mel features
- Input: One-hot encoded audio tokens (1024-d)
- Output: 80-channel mel spectrogram

**4. Vocoder (Multiband iSTFT)**
- 4-subband multiband processing
- ISTFT-based waveform generation
- No autoregressive sampling required (parallel generation)

### Training Strategy

- **Batch Size**: 2 (12GB GPU with gradient checkpointing)
- **Learning Rate**: 2e-4 with exponential decay (γ=0.9999)
- **Optimizer**: AdamW (β1=0.9, β2=0.999)
- **Sequence**: Teacher forcing with ground truth audio tokens during training

### Memory Optimization

- **Gradient Checkpointing**: Enabled on GPT-2 transformer blocks
- **Batch Size 1-2**: Fits on 12GB GPU
- **Mixed Precision**: Optional (set `fp16_run: true` in config)

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Model Size | ~130M parameters |
| GPU Memory (Training) | ~10 GB (batch size 2) |
| Training Speed | ~0.5s/step (12GB GPU) |
| Inference Speed | Real-time+ (GPU) |
| Audio Quality | 22.05 kHz, high fidelity |

---

## 📁 Project Structure

```
LLM-istft-vits/
├── configs/
│   └── ipa_tts.json              # Training configuration
├── pretrained_llm/
│   └── gpt2/                     # Pretrained GPT-2 model
├── encodec_pretrained/           # EnCodec model (offline)
├── filelists/                    # Training/validation filelists
├── ipa_tokenizer.py              # IPA tokenizer (131 symbols)
├── ipa_gpt2_model.py             # GPT-2-based TTS LLM
├── model_complete_ipa.py         # Complete TTS pipeline
├── audio_tokenizer.py            # EnCodec wrapper
├── models.py                     # Vocoder (iSTFT Generator, Discriminator)
├── mel_processing.py             # Mel spectrogram utilities
├── data_utils_ipa.py             # Data loader
├── train_ipa.py                  # Training script
├── synthesize_ipa.py             # Inference script
├── preprocess_ipa.py             # Text → IPA preprocessing
└── README.md                     # This file
```

---

## 🌍 Multilingual Support

The system uses **IPA (International Phonetic Alphabet)** as a universal phonetic representation, enabling true multilingual TTS.

**Process:**
1. Input text (any language) → Phonemizer → IPA
2. IPA → GPT-2 → Audio tokens
3. Audio tokens → Waveform

**Supported Languages:**
English, Korean, Japanese, Chinese, Spanish, French, German, Russian, Italian, Portuguese, and more.

---

## 🔗 References

- **GPT-2**: [Language Models are Unsupervised Multitask Learners](https://github.com/openai/gpt-2)
- **EnCodec**: [High Fidelity Neural Audio Compression](https://github.com/facebookresearch/encodec)
- **iSTFT Vocoder**: [Multiband iSTFT Generator](https://github.com/rishikksh20/iSTFTNet-pytorch)
- **IPA**: [International Phonetic Alphabet](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet)

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

- Pretrained GPT-2 from [Hugging Face Transformers](https://huggingface.co/gpt2)
- EnCodec from [Meta AI Research](https://github.com/facebookresearch/encodec)
- IPA phonemization via [phonemizer](https://github.com/bootphon/phonemizer)
