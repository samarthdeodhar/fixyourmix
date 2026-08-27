import os
import librosa
import numpy as np
import pandas as pd

# Folders containing reference audio
folders = ["raw_audio/balanced", "raw_audio/industry_reference"]
dataset = []
real_lows, real_mids, real_highs = [], [], []

print("🎛️ Scanning reference tracks (personal + industry)...")

for folder in folders:
    if os.path.exists(folder):
        for file_name in os.listdir(folder):
            if file_name.lower().endswith((".wav", ".mp3")):
                file_path = os.path.join(folder, file_name)
                try:
                    y, sr = librosa.load(file_path, sr=22050)
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    S_db = librosa.power_to_db(S, ref=np.max)
                    freq_averages = np.mean(S_db, axis=1)

                    low = float(np.mean(freq_averages[0:16]))
                    mid = float(np.mean(freq_averages[16:80]))
                    high = float(np.mean(freq_averages[80:128]))

                    dataset.append([low, mid, high, 0])
                    real_lows.append(low)
                    real_mids.append(mid)
                    real_highs.append(high)
                    print(f"✅ Extracted Baseline: {file_name}")
                except Exception as e:
                    print(f"❌ Error reading {file_name}: {e}")

if len(dataset) == 0:
    print("⚠️ No audio files found. Ensure .wav or .mp3 files exist in raw_audio subfolders.")
else:
    avg_low = np.mean(real_lows)
    avg_mid = np.mean(real_mids)
    avg_high = np.mean(real_highs)

    # Save target baseline so app.py can load real empirical standards
    np.save("baseline_metrics.npy", np.array([avg_low, avg_mid, avg_high]))
    print(f"\n📊 Empirical Baselines: Low={avg_low:.2f}dB, Mid={avg_mid:.2f}dB, High={avg_high:.2f}dB")

    # Generate synthetic edge cases for training multi-class classifier
    print("🧪 Synthesizing labeled mix flaws for training...")
    for _ in range(30):
        # 0: Balanced (minor natural variance)
        dataset.append([avg_low + np.random.uniform(-1.5, 1.5),
                        avg_mid + np.random.uniform(-1.5, 1.5),
                        avg_high + np.random.uniform(-1.5, 1.5), 0])
        # 1: Too Muddy (Sub/Bass boosted significantly)
        dataset.append([avg_low + np.random.uniform(6.0, 14.0), avg_mid, avg_high, 1])
        # 2: Too Harsh (Highs boosted significantly)
        dataset.append([avg_low, avg_mid, avg_high + np.random.uniform(6.0, 14.0), 2])
        # 3: Weak Mids (Scooped midrange)
        dataset.append([avg_low, avg_mid - np.random.uniform(7.0, 15.0), avg_high, 3])

    df = pd.DataFrame(dataset, columns=["low_energy", "mid_energy", "high_energy", "label"])
    df.to_csv("mix_dataset.csv", index=False)
    print("🎉 SUCCESS: Generated 'mix_dataset.csv' & 'baseline_metrics.npy'.")