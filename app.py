import streamlit as st
import numpy as np
import librosa
import bcrypt
import plotly.graph_objects as go
from supabase import create_client, Client

# 1. Page Configuration
st.set_page_config(
    page_title="fixyourmix.ai | Trap Mixing Engine",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e6edf3; font-family: -apple-system, sans-serif; }
    .hero-title { font-size: 2.4rem; font-weight: 800; background: linear-gradient(90deg, #ff4b4b, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stMetricValue"] { color: #f0f6fc !important; font-weight: 700; font-size: 1.5rem !important; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# 2. Database Init (Optional for this view)
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

st.markdown('<div class="hero-title">🎛️ fixyourmix.ai</div>', unsafe_allow_html=True)
st.markdown('### AI Source Separation & Clash Detection Engine')
st.write("---")

uploaded_file = st.file_uploader("Upload your master beat (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("✂️ Locating the drop and slicing 15 seconds for analysis..."):
        try:
            # Load the full audio
            y, sr = librosa.load(uploaded_file, sr=22050)

            # Find the drop (loudest part of the track)
            rms = librosa.feature.rms(y=y)[0]
            max_rms_frame = np.argmax(rms)
            start_sample = max(0,
                               librosa.frames_to_samples(max_rms_frame) - (sr * 2))  # Start 2 seconds before the peak
            end_sample = min(len(y), start_sample + (sr * 15))  # Grab 15 seconds

            y_slice = y[start_sample:end_sample]

            st.success("✅ Drop located. Running Harmonic-Percussive Separation...")

            # 🧠 The Magic: Separate Sustained Notes (Melody/808) from Transients (Kick/Snare/Hats)
            y_harmonic, y_percussive = librosa.effects.hpss(y_slice)

            # Convert to Decibel Spectrograms
            S_harm = librosa.power_to_db(librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=128), ref=np.max)
            S_perc = librosa.power_to_db(librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_mels=128), ref=np.max)

            harm_avg = np.mean(S_harm, axis=1)
            perc_avg = np.mean(S_perc, axis=1)

            # Isolate the exact instruments based on trap frequency bands
            bass_808 = float(np.mean(harm_avg[0:12]))  # Sustained sub (808)
            kick_drum = float(np.mean(perc_avg[0:12]))  # Transient sub (Kick)
            melody = float(np.mean(harm_avg[16:80]))  # Sustained mids (Melody/Pads)
            snare_clap = float(np.mean(perc_avg[16:80]))  # Transient mids (Snare/Clap)
            hi_hats = float(np.mean(perc_avg[80:128]))  # Transient highs (Hats)

            st.write("---")
            st.markdown("### 🔍 Advanced Diagnostic Report")

            # Engine Logic: Clash Detection

            # 1. Kick vs 808 Clash
            low_diff = abs(kick_drum - bass_808)
            if kick_drum > -40 and bass_808 > -40:  # If both exist
                if low_diff < 3.0:
                    st.error(
                        f"🚨 **LOW-END CLASH DETECTED:** Your Kick drum ({kick_drum:.1f} dB) and 808 ({bass_808:.1f} dB) are fighting for the exact same frequencies. Sidechain the 808 to the kick, or carve an EQ hole in the 808 at 50Hz.")
                elif kick_drum > bass_808 + 5:
                    st.warning(
                        "⚠️ **WEAK 808:** Your kick is punching right through, but the 808 sub-bass is buried. Turn up the 808 or add saturation.")
                else:
                    st.success("🎯 **Clean Low-End:** Your Kick and 808 are separated well and not phasing.")
            else:
                st.info(
                    "ℹ️ **Standard Trap Bass:** Only detecting heavy 808 energy without a hard kick. No low-end clashing detected.")

            # 2. Melody vs Snare Masking
            if snare_clap < melody - 2:
                st.warning(
                    f"🚨 **BURIED SNARE:** Your main melodies ({melody:.1f} dB) are masking your snare drum ({snare_clap:.1f} dB). Turn the snare bus up, or apply a slight EQ dip to the melody around 1kHz - 2kHz.")
            else:
                st.success("🎯 **Crisp Mid-Range:** Your snare is punching cleanly through the melody.")

            # 3. Hi-Hat Harshness
            if hi_hats > melody + 5:
                st.warning(
                    "🚨 **HARSH HIGHS:** Your hi-hats are overpowering the entire mix. Turn down the hat velocities to sit them properly in the background.")

            st.write("---")
            st.markdown("### 🎛️ Stem Isolation Overview")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("808 Power", f"{(bass_808 + 80) * 1.5:.0f} / 100")
            m2.metric("Kick Punch", f"{(kick_drum + 80) * 1.5:.0f} / 100")
            m3.metric("Melody Presence", f"{(melody + 80) * 1.5:.0f} / 100")
            m4.metric("Snare/Clap Punch", f"{(snare_clap + 80) * 1.5:.0f} / 100")

        except Exception as e:
            st.error(f"Processing Error: {e}")