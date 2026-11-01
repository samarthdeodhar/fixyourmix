import streamlit as st
import numpy as np
import librosa
import bcrypt
from supabase import create_client, Client

# 1. Page Configuration
st.set_page_config(
    page_title="fixyourmix| Trap Peak Engine",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e6edf3; font-family: -apple-system, sans-serif; }
    .hero-title { font-size: 2.4rem; font-weight: 800; background: linear-gradient(90deg, #ff4b4b, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stMetricValue"] { color: #f0f6fc !important; font-weight: 700; font-size: 1.5rem !important; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# 2. Database Init (Crucial for the essay flex)
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

@st.cache_resource
def get_supabase_client():
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception:
            return None
    return None

supabase: Client = get_supabase_client()

# 3. Sidebar Authentication
st.sidebar.markdown("### 🔐 Producer Portal")
if "logged_in_user" not in st.session_state:
    st.session_state["logged_in_user"] = None

if st.session_state["logged_in_user"] is None:
    choice = st.sidebar.selectbox("Session Type", ["Guest Mode", "Login", "Sign Up"])
    if choice == "Guest Mode":
        st.sidebar.info("👤 Logged in as **Guest Producer**")
    elif choice == "Sign Up":
        new_user = st.sidebar.text_input("Username / Producer Tag")
        new_pass = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Register Account"):
            if new_user and new_pass and supabase:
                hashed = bcrypt.hashpw(new_pass.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                try:
                    supabase.table("profiles").insert({"username": new_user, "password_hash": hashed}).execute()
                    st.sidebar.success("Account registered! Switch to Login tab.")
                except:
                    st.sidebar.error("Registration Error")
    elif choice == "Login":
        login_user = st.sidebar.text_input("Username / Tag")
        login_pass = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Access Studio"):
            if supabase:
                try:
                    res = supabase.table("profiles").select("*").eq("username", login_user).execute()
                    if res.data and bcrypt.checkpw(login_pass.encode('utf-8'), res.data[0]["password_hash"].encode('utf-8')):
                        st.session_state["logged_in_user"] = login_user
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid credentials.")
                except:
                    st.sidebar.error("Auth Error")
else:
    st.sidebar.success(f"🎛️ Producer: **{st.session_state['logged_in_user']}**")
    if st.sidebar.button("Log Out"):
        st.session_state["logged_in_user"] = None
        st.rerun()

# 4. Main App UI
st.markdown('<div class="hero-title">🎛️ fixyourmix.ai</div>', unsafe_allow_html=True)
st.markdown('### FL Studio Peak Balance Engine')
st.write("---")

uploaded_file = st.file_uploader("Upload your master beat (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("✂️ Locating the drop and calculating Peak dB..."):
        try:
            y, sr = librosa.load(uploaded_file, sr=22050)

            # Master Volume Check
            master_peak = float(np.max(librosa.amplitude_to_db(np.abs(y), ref=1.0)))

            rms = librosa.feature.rms(y=y)[0]
            max_rms_frame = np.argmax(rms)
            start_sample = max(0, librosa.frames_to_samples(max_rms_frame) - (sr * 2))
            end_sample = min(len(y), start_sample + (sr * 15))
            y_slice = y[start_sample:end_sample]

            st.success("✅ Drop located. Running Harmonic-Percussive Separation...")

            y_harmonic, y_percussive = librosa.effects.hpss(y_slice)

            S_harm = librosa.power_to_db(librosa.feature.melspectrogram(y=y_harmonic, sr=sr, n_mels=128), ref=np.max)
            S_perc = librosa.power_to_db(librosa.feature.melspectrogram(y=y_percussive, sr=sr, n_mels=128), ref=np.max)

            # EXTRACING PEAK DB
            bass_808 = float(np.max(S_harm[0:12]))
            kick_drum = float(np.max(S_perc[0:12]))
            melody = float(np.max(S_harm[16:80]))
            snare_clap = float(np.max(S_perc[16:80]))
            hi_hats = float(np.max(S_perc[80:128]))

            # FL Studio Trap Standards (Peak dBFS)
            target_808 = -0.5
            target_kick = -0.5
            target_snare = -4.5
            target_melody = -12.0
            target_hats = -9.0

            # Algorithmic Fader Math
            d_808 = target_808 - bass_808
            d_kick = target_kick - kick_drum
            d_snare = target_snare - snare_clap
            d_melody = target_melody - melody
            d_hats = target_hats - hi_hats

            st.write("---")
            st.markdown("### 🎛️ Stem Isolation (FL Studio Peak Meters)")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("808 Sub", f"{bass_808:.1f} dB", f"{d_808:+.1f} dB to target", delta_color="off")
            m2.metric("Kick", f"{kick_drum:.1f} dB", f"{d_kick:+.1f} dB to target", delta_color="off")
            m3.metric("Clap/Snare", f"{snare_clap:.1f} dB", f"{d_snare:+.1f} dB to target", delta_color="off")
            m4.metric("Melody", f"{melody:.1f} dB", f"{d_melody:+.1f} dB to target", delta_color="off")
            m5.metric("Hi-Hats", f"{hi_hats:.1f} dB", f"{d_hats:+.1f} dB to target", delta_color="off")

            st.write("---")
            st.markdown("### 🔍 Fader Adjustment Blueprint")

            if master_peak < -3.0:
                st.warning(f"**Master Volume:** Your master is peaking at **{master_peak:.1f} dB**. Slap a soft clipper on the master and push it up closer to 0 dB.")

            if abs(d_808) > 1.5:
                action = "Boost" if d_808 > 0 else "Lower"
                st.info(f"**808 Fader:** {action} your 808 by **{abs(d_808):.1f} dB** to hit the standard 0 dB pocket.")

            if abs(d_snare) > 1.5:
                action = "Boost" if d_snare > 0 else "Lower"
                st.info(f"**Clap/Snare Fader:** {action} your clap bus by **{abs(d_snare):.1f} dB** to sit right around -4.5 dB.")

            if abs(d_melody) > 2.0:
                action = "Boost" if d_melody > 0 else "Lower"
                st.info(f"**Melody Fader:** {action} your main melodies by **{abs(d_melody):.1f} dB** to sit back at -12 dB.")

            if abs(d_hats) > 1.5:
                action = "Boost" if d_hats > 0 else "Lower"
                st.info(f"**Hi-Hat Fader:** {action} your hats by **{abs(d_hats):.1f} dB** so they tick cleanly at -9 dB.")

        except Exception as e:
            st.error(f"Processing Error: {e}")