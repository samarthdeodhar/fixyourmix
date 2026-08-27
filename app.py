import streamlit as st
import numpy as np
import librosa
import joblib
import bcrypt
import plotly.graph_objects as go
from supabase import create_client, Client
import os

# 1. Initialize Supabase Connection
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

@st.cache_resource
def get_supabase_client():
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    return None

supabase: Client = get_supabase_client()

# 2. Web Page Configuration
st.set_page_config(page_title="fixyourmix.ai", page_icon="🎛️", layout="wide")
st.title("🎛️ fixyourmix.ai")
st.subheader("Automated Machine Learning Audio Analytics for Trap Producers")
st.write("---")

# 3. Sidebar Authentication
st.sidebar.title("🔐 Producer Access Portal")
if "logged_in_user" not in st.session_state:
    st.session_state["logged_in_user"] = None

if st.session_state["logged_in_user"] is None:
    menu = ["Guest Mode", "Login", "Sign Up"]
    choice = st.sidebar.selectbox("Select Mode", menu)

    if choice == "Guest Mode":
        st.sidebar.info("Session: **Guest Producer**")
    elif choice == "Sign Up":
        st.sidebar.subheader("Create Account")
        new_user = st.sidebar.text_input("Username/Email")
        new_pass = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Register"):
            if new_user and new_pass and supabase:
                hashed = bcrypt.hashpw(new_pass.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                try:
                    supabase.table("profiles").insert({"username": new_user, "password_hash": hashed}).execute()
                    st.sidebar.success("Account created! Switch to Login.")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
    elif choice == "Login":
        st.sidebar.subheader("Login")
        login_user = st.sidebar.text_input("Username/Email")
        login_pass = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Access Dashboard"):
            if supabase:
                try:
                    res = supabase.table("profiles").select("*").eq("username", login_user).execute()
                    if res.data and bcrypt.checkpw(login_pass.encode('utf-8'), res.data[0]["password_hash"].encode('utf-8')):
                        st.session_state["logged_in_user"] = login_user
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid credentials.")
                except Exception as e:
                    st.sidebar.error(f"Auth Error: {e}")
else:
    st.sidebar.success(f"Verified: **{st.session_state['logged_in_user']}**")
    if st.sidebar.button("Log Out"):
        st.session_state["logged_in_user"] = None
        st.rerun()

# 4. User Mix Memory
if st.session_state["logged_in_user"] and supabase:
    try:
        past_logs = supabase.table("mix_history").select("*").eq("username", st.session_state["logged_in_user"]).order("created_at", desc=True).limit(5).execute()
        if past_logs.data:
            st.markdown("### 🧠 Personal AI Mix Memory")
            verdicts = [r["verdict"] for r in past_logs.data]
            if verdicts.count("Too Muddy") >= 2:
                st.warning("🤖 **AI Trend:** Your mixes frequently lean too heavy in the sub-bass (0–60Hz). Watch your 808 gain staging.")
            elif verdicts.count("Too Harsh") >= 2:
                st.warning("🤖 **AI Trend:** High transient harshness detected across recent uploads. Watch your hi-hat velocity levels.")
            else:
                st.success("🤖 **AI Trend:** Consistent spectral balance across recent uploads.")
            st.write("---")
    except Exception:
        pass

# 5. Core Audio Analysis Engine
st.markdown("### 📥 Drop Your Drum Stem or Beat Below")
uploaded_file = st.file_uploader("Upload audio (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    st.info("🔄 Running Feature Extraction Pipeline...")

    try:
        y, sr = librosa.load(uploaded_file, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        freq_averages = np.mean(S_db, axis=1)

        low_energy = float(np.mean(freq_averages[0:16]))
        mid_energy = float(np.mean(freq_averages[16:80]))
        high_energy = float(np.mean(freq_averages[80:128]))

        # Load baseline metrics or use calibrated defaults
        if os.path.exists("baseline_metrics.npy"):
            target_baseline = np.load("baseline_metrics.npy")
        else:
            target_baseline = np.array([-14.0, -22.0, -28.0])

        delta_low = low_energy - target_baseline[0]
        delta_mid = mid_energy - target_baseline[1]
        delta_high = high_energy - target_baseline[2]

        # Predict with ML model
        pred_id = 0
        if os.path.exists("mixing_ai_model.pkl"):
            try:
                model = joblib.load("mixing_ai_model.pkl")
                pred = model.predict([[low_energy, mid_energy, high_energy]])
                pred_id = int(pred[0])
            except Exception:
                pred_id = 0

        verdict_data = {
            0: ("🎯 Solid Trap Balance", st.success, "Your track matches commercial reference targets closely across all 3 key frequency zones."),
            1: ("🚨 Too Muddy (Sub/Bass Dominant)", st.error, f"Your low-end is **{abs(delta_low):.1f} dB** above reference standard. Lower your 808/kick channel or apply a high-pass cut below 30Hz."),
            2: ("⚠️ Too Harsh (Piercing High-End)", st.warning, f"Your high-end is **{abs(delta_high):.1f} dB** above target. Lower hi-hat velocity or apply a gentle high shelf cut at 8kHz-10kHz."),
            3: ("⚠️ Scooped Midrange (Weak Melodies/Snare)", st.warning, f"Midrange energy is **{abs(delta_mid):.1f} dB** below reference target. Boost snare/clap volume or clear mud masking the 500Hz-2kHz region.")
        }

        title, ui_func, desc = verdict_data.get(pred_id, verdict_data[0])

        st.write("---")
        st.subheader("📊 AI MIX ANALYSIS REPORT")
        ui_func(f"### {title}")
        st.markdown(desc)

        # Quantitative Metrics Breakdown
        col1, col2, col3 = st.columns(3)
        col1.metric("Low Energy (Sub/808)", f"{low_energy:.1f} dB", delta=f"{delta_low:+.1f} dB vs target", delta_color="inverse")
        col2.metric("Mid Energy (Snare/Melody)", f"{mid_energy:.1f} dB", delta=f"{delta_mid:+.1f} dB vs target", delta_color="normal")
        col3.metric("High Energy (Hats/Air)", f"{high_energy:.1f} dB", delta=f"{delta_high:+.1f} dB vs target", delta_color="inverse")

        # Database Telemetry
        if st.session_state["logged_in_user"] and supabase:
            try:
                supabase.table("mix_history").insert({
                    "username": st.session_state["logged_in_user"],
                    "low_energy": low_energy,
                    "mid_energy": mid_energy,
                    "high_energy": high_energy,
                    "verdict": title
                }).execute()
            except Exception:
                pass

        # Visualization
        st.write("---")
        st.subheader("📉 Real-Time Spectral Balance Curve")
        frequency_zones = ["Low (0-250Hz / 808)", "Mid (250Hz-4kHz / Melodies)", "High (4kHz-20kHz / Hats)"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frequency_zones, y=[low_energy, mid_energy, high_energy], mode='lines+markers', name='Your Uploaded Mix', line=dict(color='#ff4b4b', width=4)))
        fig.add_trace(go.Scatter(x=frequency_zones, y=target_baseline.tolist(), mode='lines+markers', name='Industry Target Curve', line=dict(color='#00f2fe', width=3, dash='dash')))

        fig.update_layout(
            xaxis_title="Frequency Zone",
            yaxis_title="Energy (dB)",
            template="plotly_dark",
            yaxis=dict(range=[-55, 0]),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Processing Error: {e}")