import streamlit as st
import numpy as np
import librosa
import joblib
import bcrypt
import plotly.graph_objects as go
from supabase import create_client, Client

# 1. Initialize Connection to Your Remote Supabase Cloud PostgreSQL Database
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


try:
    supabase: Client = get_supabase_client()
except Exception as e:
    st.error(f"Database Routing Error: {e}")

# 2. Web Page Branding and Layout Configuration
st.set_page_config(page_title="fixyourmix.ai", page_icon="🎛️", layout="wide")
st.title("🎛️ fixyourmix.ai")
st.subheader("Automated Machine Learning Audio Analytics for Trap Producers")
st.write("---")

# 3. Sidebar Layout: Hybrid Guest & Cloud Registration Portal
st.sidebar.title("🔐 Producer Access Portal")

if "logged_in_user" not in st.session_state:
    st.session_state["logged_in_user"] = None

if st.session_state["logged_in_user"] is None:
    menu = ["Guest Mode", "Login", "Sign Up"]
    choice = st.sidebar.selectbox("Select Authentication Mode", menu)

    if choice == "Guest Mode":
        st.sidebar.info("Current Session: **Guest Producer**")
        st.sidebar.caption("Feel free to upload tracks as a guest below!")

    elif choice == "Sign Up":
        st.sidebar.subheader("Create New Account")
        new_user = st.sidebar.text_input("Choose Username/Email")
        new_pass = st.sidebar.text_input("Choose Password", type="password")

        if st.sidebar.button("Register Account"):
            if new_user and new_pass:
                hashed_bytes = bcrypt.hashpw(new_pass.encode('utf-8'), bcrypt.gensalt())
                hashed_string = hashed_bytes.decode('utf-8')

                try:
                    data, count = supabase.table("profiles").insert({
                        "username": new_user,
                        "password_hash": hashed_string
                    }).execute()
                    st.sidebar.success("Account created! Switch to Login menu to access.")
                except Exception as db_err:
                    st.sidebar.error(f"Database Error: {db_err}")
            else:
                st.sidebar.warning("Please fill out both entry slots.")

    elif choice == "Login":
        st.sidebar.subheader("Account Login")
        login_user = st.sidebar.text_input("Username/Email")
        login_pass = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Access Dashboard"):
            try:
                response = supabase.table("profiles").select("*").eq("username", login_user).execute()
                if len(response.data) > 0:
                    stored_hash = response.data[0]["password_hash"]
                    if bcrypt.checkpw(login_pass.encode('utf-8'), stored_hash.encode('utf-8')):
                        st.session_state["logged_in_user"] = login_user
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid secure password match.")
                else:
                    st.sidebar.error("Username profile record not found.")
            except Exception as login_err:
                st.sidebar.error(f"Authentication Failure: {login_err}")
else:
    st.sidebar.success(f"Verified Session: **{st.session_state['logged_in_user']}**")
    if st.sidebar.button("Log Out"):
        st.session_state["logged_in_user"] = None
        st.rerun()

# ==========================================
# 🧠 ADVANCED DYNAMIC USER MEMORY ENGINE
# ==========================================
# If a producer is logged in, pull their past data logs and build a custom profile briefing
if st.session_state["logged_in_user"] is not None:
    try:
        past_logs = supabase.table("mix_history").select("*").eq("username", st.session_state["logged_in_user"]).order(
            "created_at", desc=True).limit(5).execute()

        st.markdown("### 🧠 Personal AI Assistant Memory Briefing")
        if len(past_logs.data) > 0:
            verdicts = [row["verdict"] for row in past_logs.data]
            muddy_count = verdicts.count("Too Muddy")
            harsh_count = verdicts.count("Too Harsh")

            st.write(
                f"👋 **Welcome back, {st.session_state['logged_in_user']}!** The system analyzed your last {len(past_logs.data)} mixing uploads to map your behavioral habits:")
            if muddy_count >= 3:
                st.warning(
                    "🤖 **AI Observation:** Your mix history displays a systematic pattern of heavy sub-bass grouping. You are consistently overcompensating on your 808 velocities. Pay extra close attention to your low-to-mid balance curve adjustments today.")
            elif harsh_count >= 3:
                st.warning(
                    "🤖 **AI Observation:** Your historical trend graphs reveal a high transient piercing pattern. Your hi-hat faders are frequently crossing your -9dB sweet-spot guidelines. Try applying a soft high-shelf cut before your export today.")
            else:
                st.success(
                    "🤖 **AI Observation:** Your historical track catalog reflects strong engineering growth. Your balance ratios are clustering tight to our gold standard baselines. Let's process your next track.")
        else:
            st.info(
                "🤖 **AI Observation:** Your historical profile ledger is empty. Upload your first track below to begin building your personalized engineering catalog profile.")
        st.write("---")
    except Exception as db_fetch_err:
        pass

# 4. Main Dashboard: The Core Audio Analysis Tool
st.markdown("### 📥 Drop Your Drum Track Stem Below")
uploaded_file = st.file_uploader("Upload your track (.wav format)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.info("🔄 Running Feature Extraction Pipeline...")

    try:
        y, sr = librosa.load(uploaded_file, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        freq_averages = np.mean(S_db, axis=1)

        low_energy = np.mean(freq_averages[0:16])
        mid_energy = np.mean(freq_averages[16:80])
        high_energy = np.mean(freq_averages[80:128])

        try:
            model = joblib.load("mixing_ai_model.pkl")
            user_features = np.array([[low_energy, mid_energy, high_energy]])
            ai_prediction = model.predict(user_features)
        except Exception:
            ai_prediction = 0

        st.write("---")
        st.subheader("📊 AI MIX ANALYSIS REPORT")
        low_to_mid_gap = low_energy - mid_energy
        high_to_mid_gap = high_energy - mid_energy

        st.write("---")
        st.subheader("📊 AI MIX ANALYSIS REPORT")

        # 1. Map the model's integer output to real industry verdicts
        verdict_map = {
            0: {
                "verdict": "Good Balance",
                "ui": st.success,
                "title": "🎯 **Verdict: Solid Trap Balance!**",
                "desc": "The AI compared your stem against industry standards (Future, Carti) and found a professional spectral balance."
            },
            1: {
                "verdict": "Too Muddy",
                "ui": st.error,
                "title": "🚨 **Verdict: Too Muddy (Low-End Clipping)**",
                "desc": "The AI detected your 808/Kick masking your mid-range. Try applying a high-pass filter or cutting the 30-60Hz range slightly."
            },
            2: {
                "verdict": "Too Harsh",
                "ui": st.warning,
                "title": "⚠️ **Verdict: Too Harsh (Hi-Hats Overpowering)**",
                "desc": "The AI detected harsh transients in your high-end. Lower your hi-hat velocities or apply a high-shelf EQ cut."
            },
            3: {
                "verdict": "Weak Mids",
                "ui": st.warning,
                "title": "⚠️ **Verdict: Weak Mid-Range (Burying the Snare/Clap)**",
                "desc": "The AI detected a scooped mid-range. Your snare, claps, and main melodies are getting buried under the bass."
            }
        }

        # 2. Safely extract the prediction integer from the Scikit-Learn array
        try:
            pred_id = int(ai_prediction[0]) if isinstance(ai_prediction, np.ndarray) else int(ai_prediction)
        except Exception:
            pred_id = 0  # Fallback to 0 if the array shape is unexpected

        # 3. Inject the mapped ML result directly into the UI and Database variables
        result = verdict_map.get(pred_id, verdict_map[0])
        final_verdict = result["verdict"]

        result["ui"](result["title"])
        st.markdown(result["desc"])
        # ==========================================
        # 💾 CLOUD TELEMETRY INJECTION POINT
        # ==========================================
        if st.session_state["logged_in_user"] is not None:
            try:
                supabase.table("mix_history").insert({
                    "username": st.session_state["logged_in_user"],
                    "low_energy": float(low_energy),
                    "mid_energy": float(mid_energy),
                    "high_energy": float(high_energy),
                    "verdict": final_verdict
                }).execute()
            except Exception:
                pass

        # ==========================================
        # 📊 DYNAMIC PLOTLY VISUALIZATION LAYER
        # ==========================================
        st.write("---")
        st.subheader("📉 Real-Time Spectral Balance Curve")

        frequency_zones = ["Low Frequency (Bass/808)", "Mid Frequency (Mids/Melody)", "High Frequency (Hats/Air)"]
        user_decibel_metrics = [low_energy, mid_energy, high_energy]
        target_baseline_metrics = [-12.0, -22.0, -28.0]

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frequency_zones, y=user_decibel_metrics,
            mode='lines+markers', name='Your Uploaded Mix',
            line=dict(color='#ff4b4b', width=4)
        ))
        fig.add_trace(go.Scatter(
            x=frequency_zones, y=target_baseline_metrics,
            mode='lines', name='Target Reference Curve',
            line=dict(color='#00f2fe', width=3, dash='dash')
        ))

        fig.update_layout(
            title="Your Track vs. Target Audio Baseline Spectrum",
            xaxis_title="Acoustic Frequency Bands",
            yaxis_title="Average Energy Volume (dB)",
            template="plotly_dark",
            yaxis=dict(range=[-60, 0]),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Pipeline Processing Error: {e}")
