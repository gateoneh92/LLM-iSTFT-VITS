"""Test dataset loading"""
import utils
from data_utils_ipa import IPATextAudioDataset

hps = utils.get_hparams_from_file("configs/ipa_tts.json")

dataset = IPATextAudioDataset(hps.data.training_files, hps)

print(f"Dataset size: {len(dataset)}")

# Test one sample
sample = dataset[0]
print(f"\nSample 0:")
print(f"  IPA tokens: {sample[0].shape}")
print(f"  Audio: {sample[1].shape}")
print(f"  Spec: {sample[2].shape}")
print(f"  Audio tokens: {sample[3].shape}")
