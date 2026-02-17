import os
import sys
import torch
import numpy as np
from scipy.io.wavfile import write

# Ensure imports work
sys.path.append(os.getcwd())

import utils
import commons
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)

def main():
    # Use the config from LLM-iSTFT-VITS
    config_path = "./configs/ljs_mb_istft_vits.json"
    checkpoint_path = "../MB-iSTFT-VITS/pretrained/pretrained_MB-iSTFT-VITS_ddp.pth"
    text = "Hello, this is a test of the integrated LLM repo decoder."
    out_path = "../llm_repo_test.wav"

    hps = utils.get_hparams_from_file(config_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # The SynthesizerTrn in LLM-iSTFT-VITS matches the MB-iSTFT structure
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    net_g.eval()

    # Load weights
    utils.load_checkpoint(checkpoint_path, net_g, None)

    # Clean text (fall back to basic if needed)
    try:
        stn_tst = get_text(text, hps)
    except:
        print("Phonemizer failed, falling back to basic_cleaners")
        text_norm = text_to_sequence(text, ["basic_cleaners"])
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        stn_tst = torch.LongTensor(text_norm)

    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        # Using the standard VITS-like infer method in SynthesizerTrn
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0)[0][0,0].data.cpu().float().numpy()

    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    write(out_path, hps.data.sampling_rate, (audio * 32767).astype(np.int16))
    print(f"Successfully synthesized to {out_path}")

if __name__ == "__main__":
    main()
