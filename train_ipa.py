"""
IPA 기반 TTS 학습 스크립트
모든 언어 학습 가능
Multi-GPU DDP 지원
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import argparse
import logging

import utils
from data_utils_ipa import IPATextAudioDataset, IPATextAudioCollate
from model_complete_ipa import CompleteTTS_IPA
from models import MultiPeriodDiscriminator
from losses import discriminator_loss, generator_loss, feature_loss
from mel_processing import mel_spectrogram_torch
from stft_loss import MultiResolutionSTFTLoss
from ipa_tokenizer import IPA_VOCAB_SIZE
from torch.nn import functional as F

global_step = 0


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='JSON config file')
    parser.add_argument('-m', '--model_dir', type=str, required=True, help='Model directory')
    args = parser.parse_args()

    # Load config
    hps = utils.get_hparams_from_file(args.config)
    hps.model_dir = args.model_dir

    # Check available GPUs
    n_gpus = torch.cuda.device_count()

    if n_gpus > 1:
        print(f"🚀 Found {n_gpus} GPUs, launching multi-GPU training with DDP")
        mp.spawn(run, args=(n_gpus, hps), nprocs=n_gpus, join=True)
    else:
        print(f"🚀 Found {n_gpus} GPU, launching single-GPU training")
        run(0, 1, hps)


def run(rank, world_size, hps):
    """Main training function for single or multi-GPU"""
    # Setup distributed if multi-GPU
    is_distributed = world_size > 1
    if is_distributed:
        setup_distributed(rank, world_size)

    # Set device
    device = torch.device(f'cuda:{rank}')
    is_main_process = rank == 0

    # Create logs directory structure (only on main process)
    if is_main_process:
        log_dir = os.path.join('logs', hps.model_dir)
        checkpoint_dir = os.path.join('checkpoints', hps.model_dir)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Update model_dir to checkpoint_dir for saving models
        hps.checkpoint_dir = checkpoint_dir
        hps.log_dir = log_dir

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(log_dir)
        logger.info(f'Using {world_size} GPU(s)')
        logger.info(f'Rank {rank} using device: {device}')

        # Tensorboard
        writer = SummaryWriter(log_dir=log_dir)
    else:
        hps.checkpoint_dir = os.path.join('checkpoints', hps.model_dir)
        hps.log_dir = os.path.join('logs', hps.model_dir)
        logger = None
        writer = None

    # Synchronize all processes
    if is_distributed:
        dist.barrier()

    # Dataset
    if is_main_process:
        print("📊 Loading dataset...", flush=True)

    train_dataset = IPATextAudioDataset(hps.data.training_files, hps)

    if is_main_process:
        print(f"✅ Dataset size: {len(train_dataset)}", flush=True)

    collate_fn = IPATextAudioCollate()

    # Use DistributedSampler for multi-GPU
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4 if world_size > 1 else 0,
        pin_memory=True
    )

    # Model
    if is_main_process:
        print("🏗️  Creating models...", flush=True)

    net_g = CompleteTTS_IPA(
        n_ipa_vocab=IPA_VOCAB_SIZE,
        n_audio_vocab=1024,
        n_codebooks=8,
        n_mel_channels=hps.data.n_mel_channels,
        **hps.model
    ).to(device)

    net_d = MultiPeriodDiscriminator().to(device)

    # Wrap with DDP for multi-GPU
    if is_distributed:
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank])

    if is_main_process:
        print("✅ Generator created", flush=True)
        print("✅ Discriminator created", flush=True)

    # Optimizers
    if is_main_process:
        print("⚙️  Creating optimizers...", flush=True)

    optim_g = optim.AdamW(
        net_g.parameters(),
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )
    optim_d = optim.AdamW(
        net_d.parameters(),
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )

    if is_main_process:
        print("✅ Optimizers created", flush=True)

    # Load checkpoint (only on main process initially, then broadcast)
    if is_main_process:
        print("📂 Checking for checkpoints...", flush=True)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.checkpoint_dir, "G_*.pth"),
            net_g, optim_g
        )
        _, _, _, _ = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.checkpoint_dir, "D_*.pth"),
            net_d, optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
        if is_main_process and logger:
            logger.info(f"Loaded checkpoint from epoch {epoch_str}")
    except:
        epoch_str = 1
        global_step = 0
        if is_main_process and logger:
            logger.info("Starting from scratch")

    # Synchronize all processes
    if is_distributed:
        dist.barrier()

    # Schedulers
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    # STFT Loss
    stft_criterion = MultiResolutionSTFTLoss()

    # Save initial checkpoint (only on main process)
    if epoch_str == 1 and is_main_process:
        print("💾 Saving initial checkpoint...", flush=True)
        logger.info("Saving initial checkpoint")
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, 0, os.path.join(hps.checkpoint_dir, "G_init.pth"))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, 0, os.path.join(hps.checkpoint_dir, "D_init.pth"))
        print("✅ Initial checkpoint saved", flush=True)

    # Synchronize all processes
    if is_distributed:
        dist.barrier()

    # Training loop
    if is_main_process:
        print("🚀 Starting training loop...", flush=True)

    try:
        for epoch in range(epoch_str, hps.train.epochs + 1):
            if is_main_process:
                print(f"\n{'='*50}", flush=True)
                print(f"Epoch {epoch}/{hps.train.epochs}", flush=True)
                print(f"{'='*50}", flush=True)

            # Set epoch for DistributedSampler
            if is_distributed:
                train_loader.sampler.set_epoch(epoch)

            train_epoch(
                rank, epoch, hps, net_g, net_d, optim_g, optim_d,
                scheduler_g, scheduler_d, scaler, train_loader,
                stft_criterion, logger, writer, device, is_distributed, is_main_process
            )
    except KeyboardInterrupt:
        if is_main_process and logger:
            logger.info("Training interrupted by user")
    finally:
        # Save checkpoint on exit (only on main process)
        if is_main_process:
            logger.info("Saving final checkpoint")
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.checkpoint_dir, f"G_latest.pth"))
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.checkpoint_dir, f"D_latest.pth"))
            logger.info(f"Saved checkpoint to {hps.checkpoint_dir}/G_latest.pth and D_latest.pth")

    # Cleanup distributed
    if is_distributed:
        cleanup_distributed()


def train_epoch(rank, epoch, hps, net_g, net_d, optim_g, optim_d,
                scheduler_g, scheduler_d, scaler, train_loader,
                stft_criterion, logger, writer, device, is_distributed, is_main_process):
    global global_step
    net_g.train()
    net_d.train()

    for batch_idx, batch in enumerate(train_loader):
        ipa_tokens, ipa_lengths, spec, spec_lengths, wav, wav_lengths, audio_tokens, audio_token_lengths = batch

        ipa_tokens = ipa_tokens.to(device)
        spec = spec.to(device)
        wav = wav.to(device)
        audio_tokens = audio_tokens.to(device)

        # Generator forward
        with autocast(enabled=hps.train.fp16_run):
            # Forward pass
            audio_logits, pred_mel, y_hat, y_hat_mb = net_g(ipa_tokens, audio_tokens)

            # Loss 1: LLM Loss (cross entropy)
            # Only compute loss on audio positions (skip text positions)
            text_len = ipa_tokens.size(1)
            audio_logits_only = audio_logits[:, text_len:, :]  # [batch, audio_len, vocab]

            loss_llm = F.cross_entropy(
                audio_logits_only.reshape(-1, audio_logits_only.size(-1)),
                audio_tokens[:, 0, :].reshape(-1).long()  # First codebook
            )

            # Loss 2: Mel Loss
            mel_gt = mel_spectrogram_torch(
                wav.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            min_len = min(pred_mel.size(2), mel_gt.size(2))
            loss_mel = F.l1_loss(pred_mel[:, :, :min_len], mel_gt[:, :, :min_len])

            # Discriminator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wav, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

        # Update discriminator
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.step(optim_d)

        # Generator adversarial forward
        with autocast(enabled=hps.train.fp16_run):
            # GAN loss
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wav, y_hat)
            with autocast(enabled=False):
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_fm = feature_loss(fmap_r, fmap_g)

            # STFT loss
            sc_loss, mag_loss = stft_criterion(y_hat.squeeze(1), wav.squeeze(1))
            loss_stft = sc_loss + mag_loss

            # Total generator loss
            loss_gen_all = (
                hps.train.c_llm * loss_llm +
                hps.train.c_mel * loss_mel +
                loss_gen +
                hps.train.c_fm * loss_fm +
                hps.train.c_stft * loss_stft
            )

        # Update generator
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.step(optim_g)
        scaler.update()

        # Logging (only on main process)
        if global_step % hps.train.log_interval == 0 and is_main_process:
            logger.info(
                f"Epoch: {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Total: {loss_gen_all.item():.3f} | "
                f"LLM: {loss_llm.item():.3f} | "
                f"Mel: {loss_mel.item():.3f} | "
                f"GAN: {loss_gen.item():.3f} | "
                f"FM: {loss_fm.item():.3f} | "
                f"STFT: {loss_stft.item():.3f} | "
                f"Disc: {loss_disc.item():.3f}"
            )

        global_step += 1

        # Save checkpoint at specified interval (only on main process)
        if global_step % hps.train.save_interval == 0 and is_main_process:
            logger.info(f"Saving checkpoint at step {global_step}")
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.checkpoint_dir, f"G_step{global_step}.pth"))
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.checkpoint_dir, f"D_step{global_step}.pth"))

    # Save checkpoint at end of epoch (only on main process)
    if (epoch % 10 == 0 or epoch == 1) and is_main_process:
        logger.info(f"Saving checkpoint at epoch {epoch}")
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.checkpoint_dir, f"G_{epoch}.pth"))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.checkpoint_dir, f"D_{epoch}.pth"))

    scheduler_g.step()
    scheduler_d.step()


if __name__ == "__main__":
    main()
