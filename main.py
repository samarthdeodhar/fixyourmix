import librosa
import numpy as np

# 1. Load the user's uploaded beat
# sr=22050 forces a consistent sampling rate (Max Frequency analyzed is 11,025 Hz)
file_name = "Beat_4.wav"
y, sr = librosa.load(file_name, sr=22050)

# 2. Generate the Mel Spectrogram (using 128 frequency bands mapped to human hearing)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# Convert raw power values to a logarithmic scale (Decibels) relative to peak volume
S_db = librosa.power_to_db(S, ref=np.max)

# 3. Reduce the 2D matrix into a 1D array by averaging all time columns (axis=1)
frequency_averages = np.mean(S_db, axis=1)

# 4. Feature Engineering: Separate indices into scientifically mapped mixing zones
# Row 0 to 15 isolate the Sub-bass, 808s, and Kick thud (0 Hz to ~250 Hz)
low_freq_energy = np.mean(frequency_averages[0:16])

# Row 16 to 79 isolate Snares, Melodies, Vocals, and Synths (~250 Hz to ~3500 Hz)
mid_freq_energy = np.mean(frequency_averages[16:80])

# Row 80 to 127 isolate Hi-Hats, Cymbals, and Crisp "Air" (~3500 Hz to ~11025 Hz)
high_freq_energy = np.mean(frequency_averages[80:128])

# 5. Print the numerical breakdown (The core data features your AI will learn from)
print("=== AUDIO FEATURE EXTRACTION ===")
print(f"Low Frequency Energy (Bass): {low_freq_energy:.2f} dB")
print(f"Mid Frequency Energy (Mids): {mid_freq_energy:.2f} dB")
print(f"High Frequency Energy (Highs): {high_freq_energy:.2f} dB")
print("================================")

# 6. Baseline Analytical Logic (Will be upgraded to Scikit-Learn next week)
print("\n=== INITIAL MIX ANALYSIS REPORT ===")
balance_ratio = low_freq_energy - high_freq_energy

if balance_ratio > 15:
    print("Prediction: Too Muddy")
    print("Suggestion: Your bass frequencies dominate the mix. Try lowering your kick or 808.")
elif balance_ratio < 2:
    print("Prediction: Too Harsh")
    print("Suggestion: Your high-end is piercing. Lower the volume of your hi-hats or percussion.")
else:
    print("Prediction: Good Balance")
    print("Suggestion: Excellent spectral balance across your instruments.")
print("=== END OF REPORT ===")
