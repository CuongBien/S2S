import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from datasets import load_dataset, Audio
from transformers import MimiModel, AutoFeatureExtractor
import torch
import soundfile as sf
import numpy as np
import io

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang chạy trên: {device}")

librispeech_dummy = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    "clean",
    split="validation"
).cast_column("audio", Audio(decode=False))

print("Đang kết nối tới Hugging Face để tải Weights (khoảng 1GB)...")
model = MimiModel.from_pretrained("kyutai/mimi").to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
target_sr = feature_extractor.sampling_rate  # 24000
print(f"Tải hoàn tất! Target sampling rate: {target_sr} Hz")

# Đọc audio từ bytes
audio_entry = librispeech_dummy[0]["audio"]
audio_sample, sampling_rate = sf.read(io.BytesIO(audio_entry["bytes"]))
print(f"Audio gốc: {sampling_rate} Hz, shape: {audio_sample.shape}")

# ✅ Resample nếu cần
if sampling_rate != target_sr:
    import scipy.signal as signal
    num_samples = int(len(audio_sample) * target_sr / sampling_rate)
    audio_sample = signal.resample(audio_sample, num_samples)
    print(f"Đã resample: {sampling_rate} Hz → {target_sr} Hz")

# Đảm bảo mono (1D)
if audio_sample.ndim > 1:
    audio_sample = audio_sample.mean(axis=1)

inputs = feature_extractor(
    raw_audio=audio_sample,
    sampling_rate=target_sr,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    encoder_outputs = model.encode(inputs["input_values"])
    audio_values = model.decode(encoder_outputs.audio_codes)[0]

out_audio = audio_values.squeeze().cpu().numpy()
sf.write("output_mimi.wav", out_audio, target_sr)
print("Đã lưu file output_mimi.wav")