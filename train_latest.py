import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from pqmf import PQMF

import commons
import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  LLMSynthesizer,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
  subband_stft_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
global_step = 0

# LLM 손실 함수 정의 (Cross Entropy)
def llm_loss(logits, targets):
  # logits: [batch, seq_len, n_audio_vocab]
  # targets: [batch, seq_len]
  return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

def main():
  """Assume Single Node Multi GPUs Training Only"""
  # assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '65520'
#   n_gpus = 1

  hps = utils.get_hparams()
  if n_gpus > 1:
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
  else:
    run(0, n_gpus, hps)


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

  train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioCollate()
  train_loader = DataLoader(train_dataset, num_workers=0, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  # LLMSynthesizer 사용하도록 변경
  net_g = LLMSynthesizer(
      len(symbols),
      n_audio_vocab=1024, # EnCodec 기본 어휘 사전 크기
      n_codebooks=8,      # EnCodec 기본 코드북 수
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
  
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  
  if n_gpus > 1:
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0
    if hps.pretrained_gen != "":
      utils.load_pretrained_generator(hps.pretrained_gen, net_g)

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval], device, n_gpus)
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None, device, n_gpus)
    scheduler_g.step()
    scheduler_d.step()



def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, device, n_gpus):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, audio_tokens, audio_token_lengths) in enumerate(train_loader):
    x, x_lengths = x.to(device, non_blocking=True), x_lengths.to(device, non_blocking=True)
    spec, spec_lengths = spec.to(device, non_blocking=True), spec_lengths.to(device, non_blocking=True)
    y, y_lengths = y.to(device, non_blocking=True), y_lengths.to(device, non_blocking=True)
    audio_tokens = audio_tokens.to(device, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      # LLMSynthesizer forward: 오디오 토큰 예측
      audio_logits = net_g(x, audio_tokens)
      
      # LLM Loss 계산 (텍스트 이후의 오디오 토큰 예측 부분만 추출)
      # logits: [batch, text_len + audio_len, n_audio_vocab]
      # targets: [batch, n_codebooks, audio_len] -> [batch, audio_len-1] (코드북 0번 기준)
      text_len = x.size(1)
      # 오디오 토큰 시퀀스에 대응하는 로짓만 슬라이싱
      # 텍스트의 마지막 토큰 위치가 첫 번째 오디오 토큰을 예측하는 지점임
      audio_logits_for_loss = audio_logits[:, text_len-1:-1, :]
      audio_targets = audio_tokens[:, 0, :]
      
      # VITS 디코더 부분 학습을 위한 오디오 합성 (Teacher Forcing 스타일)
      # LLM이 예측한 토큰이 아니라, 실제 Ground Truth 오디오 토큰의 임베딩을 디코더에 넣습니다.
      if n_gpus > 1:
        model = net_g.module
      else:
        model = net_g

      # LLMSynthesizer 내부의 디코더를 직접 호출하거나, 토큰 기반으로 강제 합성
      x_emb = 0
      for i in range(model.llm.n_codebooks):
          x_emb += model.llm.audio_embs[i](audio_tokens[:, i, :])
      z = model.token_proj(x_emb).transpose(1, 2)
      y_hat, y_hat_mb = model.dec(z)

      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      
      # 로깅용 y_mel 정의
      y_mel = mel 
      
      # 슬라이싱 대신 전체 길이로 예시 (실제로는 최적화 필요)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      # Discriminator 학습
      # 길이 맞추기
      min_wav_len = min(y.size(-1), y_hat.size(-1))
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y[:,:,:min_wav_len], y_hat[:,:,:min_wav_len].detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator & LLM 학습
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y[:,:,:min_wav_len], y_hat[:,:,:min_wav_len])
      with autocast(enabled=False):
        # 1. LLM Loss (어휘 예측)
        loss_llm_val = llm_loss(audio_logits_for_loss, audio_targets) # 다음 토큰 예측
        
        # 2. VITS Decoder Losses
        # 길이 맞추기 (STFT 패딩 차이로 인해 발생 가능)
        min_len = min(mel.size(-1), y_hat_mel.size(-1))
        loss_mel = F.l1_loss(mel[:,:,:min_len], y_hat_mel[:,:,:min_len]) * hps.train.c_mel
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        
        if hps.model.mb_istft_vits == True:
          pqmf = PQMF(y.device)
          y_mb = pqmf.analysis(y)
          loss_subband = subband_stft_loss(hps, y_mb, y_hat_mb)
        else:
          loss_subband = torch.tensor(0.0)

        # 전체 손실 합산
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_subband + loss_llm_val * 10.0 # LLM 비중 조절

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_llm_val, loss_subband]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/llm": loss_llm_val, "loss/g/subband": loss_subband})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            # "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval, device)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1

  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))
  
    

 
def evaluate(hps, generator, eval_loader, writer_eval, device):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, audio_tokens, audio_token_lengths) in enumerate(eval_loader):
        x, x_lengths = x.to(device), x_lengths.to(device)
        spec, spec_lengths = spec.to(device), spec_lengths.to(device)
        y, y_lengths = y.to(device), y_lengths.to(device)
        audio_tokens = audio_tokens.to(device)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        audio_tokens = audio_tokens[:1]
        break
      
      if isinstance(generator, (DDP, nn.DataParallel)):
        model = generator.module
      else:
        model = generator
        
      if isinstance(model, LLMSynthesizer):
        y_hat, y_hat_mb = model.infer(x, max_len=100)
        y_hat_lengths = torch.tensor([y_hat.size(-1)], device=device)
      else:
        y_hat, y_hat_mb, attn, mask, *_ = model.infer(x, x_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
  main()
