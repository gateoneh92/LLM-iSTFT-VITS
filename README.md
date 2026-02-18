# 🌸 LLM-iSTFT-VITS

Next-Generation Lightweight LLM-based Text-to-Speech System.

This project combines the fast synthesis capabilities of MB-iSTFT-VITS with the advanced contextual understanding of modern LLMs to create a cutting-edge TTS engine.

---

## 🚀 Key Features

- **🧠 LLM-Based Architecture**: Uses Transformer to process text and audio tokens as sequences, enabling natural emotion and prosody expression.
- **🎧 EnCodec Tokenization**: Leverages Meta AI's EnCodec to convert high-quality speech into compressed token sequences for efficient learning.
- **⚡ Ultra-Fast Decoding**: Adopts MB-iSTFT technology to achieve speech generation dozens of times faster than real-time, even on CPU.
- **📈 Transfer Learning Support**: Load pretrained decoder weights to achieve high-quality results quickly with minimal data.

---

## 🆚 Differences from MB-iSTFT-VITS

While the original **MB-iSTFT-VITS** is a fast and lightweight TTS model, it has limitations in handling long contexts and expressing subtle emotions.
**LLM-iSTFT-VITS** addresses these limitations by adopting an **LLM (Large Language Model) approach**.

| Aspect | Original MB-iSTFT-VITS | 🌸 LLM-iSTFT-VITS (This Repo) |
| :--- | :--- | :--- |
| **Core Architecture** | VITS (VAE + Flow + GAN) | **Transformer (GPT-style LLM) + VITS Decoder** |
| **Input Processing** | Phoneme → Waveform (End-to-End) | **Text Tokens + Audio Tokens (Sequence-to-Sequence)** |
| **Audio Representation** | Linear / Mel Spectrogram | **EnCodec Discrete Tokens (Neural Codec)** |
| **Context Understanding** | Sentence-level (Local Context) | **Long Sequences & Context-based (Global Context)** |
| **Strengths** | Very fast, lightweight | **More natural prosody, emotion expression, In-Context Learning potential** |
| **Training Method** | Monotonic Alignment Search (MAS) | **Next Token Prediction (Auto-regressive) + GAN Finetuning** |

> **Summary**: This project combines the **superior contextual understanding of LLMs** with the **ultra-fast synthesis capabilities of MB-iSTFT-VITS** in a hybrid next-generation model.

---

## 🛠️ Installation

### 1. Virtual Environment Setup & Package Installation
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -U pip
pip install torch torchaudio encodec numpy scipy matplotlib tensorboard cython librosa phonemizer unidecode
```

### 2. Build Monotonic Alignment (Required)
You need to build the Cython code for training and inference.
```bash
cd monotonic_align
python3 setup.py build_ext --inplace
cd ..
```

---

## 📂 Dataset Preparation

1. Prepare training audio files (.wav, 22050Hz recommended) in the `wavs/` folder.
2. Create a file list in the `filelists/` folder. (Format: `absolute_file_path|text`)
   - Example: `/path/to/audio.wav|This is a sample sentence.`

---

## 🏋️ Training

Using a pretrained decoder (MB-iSTFT-VITS) during training can significantly speed up performance improvement.

### Starting with a Pretrained Model (Recommended)
```bash
python3 train_latest.py \
  -c configs/ljs_mb_istft_vits.json \
  -m [model_name] \
  -p pretrained/pretrained_MB-iSTFT-VITS_ddp.pth
```
* `-p`: Specify the path to the pretrained decoder weights. (The decoder part will be automatically transferred.)

### Training from Scratch
```bash
python3 train_latest.py -c configs/ljs_mb_istft_vits.json -m [model_name]
```

---

## 🎙️ Inference / Synthesis

Use trained checkpoints or pretrained models to synthesize speech.

### 1. Using Test Scripts
Modify `test_llm_repo.py` with your desired text and run:
```bash
python3 test_llm_repo.py
```

### 2. Using Jupyter Notebook
Open `inference.ipynb` for interactive speech synthesis and real-time result verification.

---

## 🤝 Acknowledgements
- [MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [Official VITS](https://github.com/jaywalnut310/vits)
- [Meta EnCodec](https://github.com/facebookresearch/encodec)

