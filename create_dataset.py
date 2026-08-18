import os
import librosa
import numpy as np
import pandas as pd

# Path directly to the folder where your 8 beats live
source_folder = "raw_audio/balanced"
industry_reference_data = []

print("🎛️ Initializing local scanner. Analyzing your 8 reference beats...")

if not os.path.exists(source_folder):
    print(f"❌ Error: Could not find the folder '{source_folder}'. Make sure it exists!")
else:
    # Loop through the files already sitting in your local folder
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(source_folder, file_name)

            try:
                # Process the real track spectrum coefficients (Same as your main.py)
                y, sr = librosa.load(file_path, sr=22050)
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_db = librosa.power_to_db(S, ref=np.max)
                freq_averages = np.mean(S_db, axis=1)

                # Map the true commercial frequency distributions from your tracks
                low_energy = np.mean(freq_averages[0:16])     # Kick & Bass
                mid_energy = np.mean(freq_averages[16:80])    # Snares & Melodies
                high_energy = np.mean(freq_averages[80:128])  # Hi-Hats & Air

                # Label 0 means this is your target "Gold Standard" balanced baseline
                industry_reference_data.append([low_energy, mid_energy, high_energy, 0])
                print(f"📊 Gold Standard Baseline Extracted from: {file_name}")

            except Exception as e:
                print(f"❌ Error analyzing local file {file_name}: {e}")

# Build your pristine empirical reference catalog spreadsheet
if len(industry_reference_data) > 0:
    columns = ["low_energy", "mid_energy", "high_energy", "label"]
    df = pd.DataFrame(industry_reference_data, columns=columns)
    df.to_csv("mix_dataset.csv", index=False)
    print("\n🎉 SUCCESS! 'mix_dataset.csv' created from your local 8 beats.")
else:
    print("\n⚠️ No data saved. Ensure your 8 WAV files are inside 'raw_audio/balanced'.")
