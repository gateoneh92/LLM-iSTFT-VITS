# Docker Setup for LLM-iSTFT-VITS

## Prerequisites

- Docker (>= 20.10)
- Docker Compose (>= 1.29)
- NVIDIA Docker Runtime (nvidia-docker2)
- NVIDIA GPU with CUDA support

### Install NVIDIA Docker Runtime

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Quick Start

### 1. Build Docker Image

```bash
docker build -t llm-istft-vits:latest .
```

Or using docker-compose:

```bash
docker-compose build
```

### 2. Download Pretrained Models

Before running, download the required pretrained models:

```bash
# GPT-2
python3 -c "from transformers import GPT2Model, GPT2Tokenizer; \
    model = GPT2Model.from_pretrained('gpt2'); \
    model.save_pretrained('./pretrained_llm/gpt2'); \
    tok = GPT2Tokenizer.from_pretrained('gpt2'); \
    tok.save_pretrained('./pretrained_llm/gpt2')"
```

EnCodec is already included in `encodec_pretrained/`.

### 3. Run Container

#### Using docker-compose (Recommended)

```bash
# Start container
docker-compose up -d

# Enter container
docker-compose exec llm-istft-vits bash

# Stop container
docker-compose down
```

#### Using docker run

```bash
docker run --gpus all -it \
  --name llm-istft-vits \
  --shm-size 8g \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/pretrained_llm:/workspace/pretrained_llm \
  -v $(pwd)/encodec_pretrained:/workspace/encodec_pretrained \
  -v $(pwd)/filelists:/workspace/filelists \
  -v $(pwd)/configs:/workspace/configs \
  -p 6006:6006 \
  llm-istft-vits:latest
```

## Usage

### Training

Inside the container:

```bash
# Prepare IPA filelist
python3 preprocess_ipa.py -i filelists/ljs_audio_text_train_filelist.txt.cleaned \
                           -o filelists/train_ipa.txt -l en-us

# Start training
python3 train_ipa.py -c configs/ipa_tts.json -m my_model
```

### Monitor Training with TensorBoard

```bash
# Inside container
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# Access from host browser
# http://localhost:6006
```

### Synthesis

```bash
# Inside container
python3 synthesize_ipa.py
```

Or use Python API:

```python
from synthesize_ipa import synthesize_text

# Korean
synthesize_text("안녕하세요", language='ko', output_path="output_ko.wav")

# English
synthesize_text("Hello world", language='en-us', output_path="output_en.wav")
```

## Volume Mounts

The docker-compose setup mounts the following directories:

- `./logs` → Container logs and TensorBoard events
- `./checkpoints` → Model checkpoints
- `./pretrained_llm` → Pretrained GPT-2 models
- `./encodec_pretrained` → EnCodec models
- `./filelists` → Training/validation filelists
- `./configs` → Configuration files

This allows you to:
- Access checkpoints and logs from the host
- Modify configs without rebuilding
- Persist training progress

## GPU Configuration

To use specific GPUs:

```bash
# Use GPU 0 only
CUDA_VISIBLE_DEVICES=0 docker-compose up

# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 docker-compose up
```

Or modify `docker-compose.yml`:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in `configs/ipa_tts.json`:

```json
{
  "train": {
    "batch_size": 1
  }
}
```

### Permission Errors

If you encounter permission issues with mounted volumes:

```bash
# On host
sudo chown -R $USER:$USER logs checkpoints
```

### Container Exits Immediately

Check logs:

```bash
docker-compose logs
```

## Building for Different CUDA Versions

Edit `Dockerfile` to change CUDA version:

```dockerfile
# For CUDA 12.1
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# For CUDA 11.3
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
```

Then rebuild:

```bash
docker-compose build --no-cache
```

## Clean Up

```bash
# Remove container and volumes
docker-compose down -v

# Remove image
docker rmi llm-istft-vits:latest

# Remove all unused images and cache
docker system prune -a
```
