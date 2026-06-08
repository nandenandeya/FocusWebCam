"""
FocusWebCam — Streamlit Edition with Premium UI (Proyek 3 - Final)
===================================================================
- Landing page dengan tombol navigasi
- Background gambar dari assets/bg.png
- Logika identik Proyek 2
- UI semirip Proyek 1
"""

import streamlit as st
import cv2
import numpy as np
import time
import queue
from collections import deque
from datetime import datetime
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FocusWebCam | Ethical AI Focus Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# IMPOR MEDIAPIPE
# ============================================================
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# ============================================================
# LANDMARK INDICES (IDENTIK PROYEK 2)
# ============================================================
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
NOSE_TIP, FACE_LEFT, FACE_RIGHT = 1, 234, 454

# ============================================================
# MODEL PARAMETERS (HARDCODE dari training_report.txt)
# ============================================================
MODEL_COEF = {"ear": 1.0494, "head_pose": -2.6625, "mouth_ratio": 2.0005}
MODEL_INTERCEPT = -0.5234
MODEL_SCALER = {
    "ear":         {"mean": 0.214, "std": 0.098},
    "head_pose":   {"mean": 0.178, "std": 0.245},
    "mouth_ratio": {"mean": 0.068, "std": 0.082},
}
ALERT_THRESHOLD  = 40
EAR_OPEN         = 0.25
EAR_CLOSED       = 0.15
SMOOTHING_WINDOW = 3
MOUTH_MAX_REALISTIC = 0.12

# ============================================================
# FUNGSI PERHITUNGAN FITUR (IDENTIK PROYEK 2)
# ============================================================
def calc_ear(lm, indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    A = np.hypot(pts[1][0]-pts[5][0], pts[1][1]-pts[5][1])
    B = np.hypot(pts[2][0]-pts[4][0], pts[2][1]-pts[4][1])
    C = np.hypot(pts[0][0]-pts[3][0], pts[0][1]-pts[3][1])
    return (A + B) / (2.0 * C) if C else 0.0

def calc_head_pose(lm, w, h):
    nose  = lm[NOSE_TIP]
    left  = lm[FACE_LEFT]
    right = lm[FACE_RIGHT]
    face_center = (left.x + right.x) / 2
    face_width  = abs(right.x - left.x)
    return abs(nose.x - face_center) / face_width if face_width else 0.0

def calc_mouth(lm, w, h):
    top    = lm[MOUTH_TOP]
    bottom = lm[MOUTH_BOTTOM]
    left   = lm[MOUTH_LEFT]
    right  = lm[MOUTH_RIGHT]
    vertical   = abs((top.y - bottom.y) * h)
    horizontal = abs((left.x - right.x) * w)
    ratio = vertical / horizontal if horizontal else 0.0
    return min(ratio, MOUTH_MAX_REALISTIC)

def standardize(v, mean, std):
    return (v - mean) / std if std else 0.0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict_probability(ear, head_pose, mouth):
    ear_s   = standardize(ear,       **MODEL_SCALER["ear"])
    head_s  = standardize(head_pose, **MODEL_SCALER["head_pose"])
    mouth_s = standardize(mouth,     **MODEL_SCALER["mouth_ratio"])
    logit   = (MODEL_COEF["ear"]        * ear_s  +
               MODEL_COEF["head_pose"]  * head_s +
               MODEL_COEF["mouth_ratio"]* mouth_s +
               MODEL_INTERCEPT)
    return float(sigmoid(logit))

def get_color(score):
    if score >= 65:  return (0, 255, 136)
    if score >= 40:  return (0, 200, 255)
    return (80, 80, 255)

def explain_score(ear, head, mouth, score):
    neg = []
    if ear   < 0.20: neg.append("mata tertutup/berkedip")
    if head  > 0.15: neg.append("kepala menoleh")
    if mouth > 0.08: neg.append("mulut terbuka")
    if score >= 65:
        return f"✅ Fokus baik ({score}/100)"
    elif score >= 40:
        isu = ", ".join(neg) if neg else "pertahankan kondisi"
        return f"⚡ Perhatian ({score}/100) — {isu}"
    else:
        isu = ", ".join(neg) if neg else "kondisi tidak optimal"
        return f"⚠️ Tidak fokus ({score}/100) — {isu}"

# ============================================================
# QUEUE DAN SESSION STATE
# ============================================================
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=5)
result_queue = st.session_state.result_queue

def init_state():
    defaults = {
        "session_active":   False,
        "session_start":    None,
        "score_history":    [],
        "alert_count":      0,
        "low_score_count":  0,
        "last_alert_time":  0,
        "log_entries":      ["— Sistem siap —"],
        "consent_given":    False,
        "consent_asked":    False,
        "show_landing":     True,
        "show_exit_popup":  False,
        "show_face_warning": False,
        "face_warning_triggered": False,
        "show_session_complete": False,
        "disp_score":       None,
        "disp_ear":         None,
        "disp_head":        None,
        "disp_mouth":       None,
        "disp_expl":        "",
        "disp_face":        False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ============================================================
# VIDEO PROCESSOR (IDENTIK PROYEK 2)
# ============================================================
class FocusVideoProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._smooth = deque(maxlen=SMOOTHING_WINDOW)
        self.no_face_counter = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res  = self.face_mesh.process(rgb)

        if res.multi_face_landmarks:
            self.no_face_counter = 0
            if st.session_state.show_face_warning:
                st.session_state.show_face_warning = False
                st.session_state.face_warning_triggered = False
            lm    = res.multi_face_landmarks[0].landmark
            ear_l = calc_ear(lm, LEFT_EYE,  w, h)
            ear_r = calc_ear(lm, RIGHT_EYE, w, h)
            ear   = (ear_l + ear_r) / 2.0
            head  = calc_head_pose(lm, w, h)
            mouth = calc_mouth(lm, w, h)

            prob  = predict_probability(ear, head, mouth)
            self._smooth.append(prob * 100)
            score = int(np.clip(round(np.mean(self._smooth)), 0, 100))
            color = get_color(score)
            expl  = explain_score(ear, head, mouth, score)

            data = {
                "face":  True,
                "score": score,
                "ear":   round(ear,   4),
                "head":  round(head,  4),
                "mouth": round(mouth, 4),
                "expl":  expl,
            }
            try:
                result_queue.put_nowait(data)
            except queue.Full:
                try:    result_queue.get_nowait()
                except: pass
                try:    result_queue.put_nowait(data)
                except: pass

            # Draw overlay
            for idx in LEFT_EYE + RIGHT_EYE:
                pt = lm[idx]
                cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 2, color, -1)

            fl = lm[FACE_LEFT]; fr = lm[FACE_RIGHT]
            ft = lm[10];        fb = lm[152]
            cv2.rectangle(img,
                (int(fl.x*w), int(ft.y*h)),
                (int(fr.x*w), int(fb.y*h)),
                color, 1)

            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (210, 105), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
            cv2.putText(img, f"FOCUS: {score}", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(img, f"EAR:{ear:.3f}  HEAD:{head:.3f}", (18, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
            cv2.putText(img, f"MOUTH:{mouth:.3f}", (18, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
            bar_w = int((score/100)*182)
            cv2.rectangle(img, (18, 88), (200, 97), (40,40,40), -1)
            cv2.rectangle(img, (18, 88), (18+bar_w, 97), color, -1)
        else:
            self.no_face_counter += 1
            if self.no_face_counter > 30 and not st.session_state.face_warning_triggered:
                st.session_state.show_face_warning = True
                st.session_state.face_warning_triggered = True
            cv2.putText(img, "Tidak ada wajah terdeteksi", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,80,80), 1)
            data = {"face": False, "score": 0,
                    "ear": None, "head": None, "mouth": None, "expl": ""}
            try:
                result_queue.put_nowait(data)
            except queue.Full:
                try:    result_queue.get_nowait()
                except: pass
                try:    result_queue.put_nowait(data)
                except: pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================================
# DRAIN QUEUE
# ============================================================
def drain_queue():
    latest = None
    while True:
        try:
            latest = result_queue.get_nowait()
        except queue.Empty:
            break
    if latest is None:
        return
    st.session_state.disp_score = latest["score"]
    st.session_state.disp_ear   = latest["ear"]
    st.session_state.disp_head  = latest["head"]
    st.session_state.disp_mouth = latest["mouth"]
    st.session_state.disp_expl  = latest["expl"]
    st.session_state.disp_face  = latest["face"]

    if st.session_state.session_active and latest["face"]:
        score = latest["score"]
        st.session_state.score_history.append(score)

        if score < ALERT_THRESHOLD:
            st.session_state.low_score_count += 1
        else:
            st.session_state.low_score_count = 0

        now = time.time()
        if (st.session_state.low_score_count >= 5 and
                now - st.session_state.last_alert_time >= 30):
            st.session_state.alert_count += 1
            st.session_state.last_alert_time = now
            st.session_state.low_score_count = 0
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.log_entries.insert(
                0, f"⚠️ [{ts}] Alert #{st.session_state.alert_count} — skor {score}")
            st.warning(f"⚠️ Skor fokus rendah ({score}) selama 5 detik!")

# ============================================================
# CSS PREMIUM (meniru Proyek 1 dengan background gambar)
# ============================================================
def load_css():
    # Cek apakah file bg.png ada di folder assets
    try:
        with open("assets/bg.png", "rb"):
            bg_exists = True
    except:
        bg_exists = False
    
    bg_style = ""
    if bg_exists:
        bg_style = """
        background: url('assets/bg.png') no-repeat center center fixed;
        background-size: cover;
        """
    else:
        bg_style = """
        background: linear-gradient(145deg, #dde8f0 0%, #c8d8e8 40%, #b8cfe0 100%);
        """
    
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lusitana:wght@400;700&family=Kameron:wght@400;600;700&family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    .stApp {{
        font-family: 'Syne', sans-serif;
        {bg_style}
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {{
        display: none !important;
    }}
    [data-testid="stSidebar"] {{
        display: none !important;
    }}
    /* Landing page */
    .landing-container {{
        position: fixed;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding: 80px;
        z-index: 10;
        background: transparent;
    }}
    .landing-content {{
        max-width: 700px;
        margin-left: 8%;
    }}
    .landing-welcome {{
        font-family: 'Lusitana', serif;
        font-size: 5rem;
        font-weight: 400;
        color: #1e2d40;
        line-height: 1;
    }}
    .landing-title {{
        font-family: 'Kameron', serif;
        font-size: 5rem;
        font-weight: 700;
        color: #1a2433;
        display: flex;
        align-items: center;
        gap: 24px;
    }}
    .dot-pulse {{
        display: inline-block;
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: #3a8c52;
        animation: dotBlink 2.2s ease-in-out infinite;
    }}
    @keyframes dotBlink {{
        0%,100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(58,140,82,0.5); }}
        50% {{ opacity: 0.35; box-shadow: 0 0 0 8px rgba(58,140,82,0); }}
    }}
    .landing-subtitle {{
        font-family: 'Lusitana', serif;
        font-size: 1.6rem;
        color: #4a6075;
        margin-top: 16px;
    }}
    .app-wrapper {{
        padding: 16px 24px;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(30,45,64,0.15);
        padding-bottom: 10px;
        margin-bottom: 12px;
    }}
    .logo {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .logo-dot {{
        width: 20px;
        height: 20px;
        background: #3a8c52;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(58,140,82,0.6);
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%,100% {{ opacity: 1; box-shadow: 0 0 10px #3a8c52; }}
        50% {{ opacity: 0.45; box-shadow: 0 0 4px #3a8c52; }}
    }}
    .logo-text {{
        font-family: 'Kameron', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #1a2433;
    }}
    .header-status {{
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        color: #6a7e92;
        letter-spacing: 0.1em;
    }}
    .main-layout {{
        display: grid;
        grid-template-columns: 1fr 340px;
        gap: 16px;
        flex: 1;
        overflow: hidden;
    }}
    .camera-section {{
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}
    .camera-frame {{
        position: relative;
        flex: 1;
        background: rgba(255,255,255,0.25);
        border: 1px solid rgba(30,45,64,0.18);
        border-radius: 6px;
        overflow: hidden;
        backdrop-filter: blur(4px);
    }}
    .corner {{
        position: absolute;
        width: 16px;
        height: 16px;
        border-color: #3a8c52;
        border-style: solid;
        z-index: 10;
        transition: border-color 0.3s;
    }}
    .tl {{ top: 10px; left: 10px; border-width: 2px 0 0 2px; }}
    .tr {{ top: 10px; right: 10px; border-width: 2px 2px 0 0; }}
    .bl {{ bottom: 10px; left: 10px; border-width: 0 0 2px 2px; }}
    .br {{ bottom: 10px; right: 10px; border-width: 0 2px 2px 0; }}
    .face-status {{
        position: absolute;
        bottom: 12px;
        left: 12px;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        background: rgba(255,255,255,0.7);
        padding: 4px 10px;
        border-radius: 2px;
        z-index: 10;
    }}
    .info-section {{
        display: flex;
        flex-direction: column;
        gap: 10px;
        overflow-y: auto;
        scrollbar-width: none;
    }}
    .score-card, .features-grid, .stats-card, .log-card {{
        background: rgba(255,255,255,0.55);
        border: 1px solid rgba(30,45,64,0.12);
        border-radius: 6px;
        backdrop-filter: blur(6px);
        padding: 14px 16px;
    }}
    .score-label {{
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #6a7e92;
        letter-spacing: 0.12em;
    }}
    .score-number {{
        font-family: 'Kameron', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e2d40;
        line-height: 1;
    }}
    .score-unit {{
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        color: #8a9eb0;
    }}
    .score-bar-track {{
        height: 8px;
        background: #dce8f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }}
    .score-bar-fill {{
        height: 100%;
        width: 0%;
        background: #3a8c52;
        transition: width 0.4s;
    }}
    .features-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
        padding: 10px 8px;
    }}
    .feature-card {{
        background: rgba(255,255,255,0.5);
        border: 1px solid rgba(30,45,64,0.1);
        border-radius: 4px;
        padding: 8px 4px;
        text-align: center;
    }}
    .feature-name {{
        font-family: 'Space Mono', monospace;
        font-size: 0.5rem;
        color: #6a7e92;
        text-transform: uppercase;
    }}
    .feature-value {{
        font-family: 'Kameron', serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: #1e2d40;
        margin: 4px 0;
    }}
    .feature-bar-track {{
        height: 3px;
        background: #dce8f0;
        border-radius: 2px;
        overflow: hidden;
    }}
    .feature-bar-fill {{
        height: 100%;
        width: 0%;
        background: #3a8c52;
        transition: width 0.4s;
    }}
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 6px;
        text-align: center;
    }}
    .stat-value {{
        font-family: 'Kameron', serif;
        font-size: 1rem;
        font-weight: 600;
        color: #c0392b;
    }}
    .stat-label {{
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        color: #8a9eb0;
    }}
    .log-list {{
        max-height: 180px;
        overflow-y: auto;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #6a7e92;
    }}
    .log-item {{
        padding: 3px 0;
        border-bottom: 1px solid rgba(30,45,64,0.07);
    }}
    .log-alert {{ color: #c0392b !important; }}
    .log-focus {{ color: #3a8c52 !important; }}
    .privacy-note {{
        font-family: 'Space Mono', monospace;
        font-size: 0.5rem;
        color: #8a9eb0;
        text-align: center;
        margin-top: 8px;
    }}
    .stButton > button {{
        background: transparent !important;
        border: 1.5px solid #3a8c52 !important;
        color: #3a8c52 !important;
        font-family: 'Space Mono', monospace !important;
        width: 100%;
        border-radius: 6px !important;
        transition: all 0.25s;
    }}
    .stButton > button:hover {{
        background: #3a8c52 !important;
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# DIALOG POPUP (MENGGUNAKAN DECORATOR @st.dialog)
# ============================================================
if not st.session_state.consent_asked:
    @st.dialog("📋 Privacy Agreement")
    def _consent_dialog():
        st.markdown("""
        **To help you track your focus levels accurately, FocusWebCam needs to analyze your facial data through your camera. But don't worry, your privacy is our number one priority!**

        - ✅ **100% Local Processing** – All facial analysis happens directly on your device.
        - ✅ **No Video Streams Sent Anywhere** – We do not upload or save your video.
        - ✅ **Only Session Scores Saved** – Aggregate scores for your progress.
        """)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Allow", use_container_width=True):
                st.session_state.consent_given = True
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, "✅ Privacy consent granted.")
                st.rerun()
        with col2:
            if st.button("❌ Deny", use_container_width=True):
                st.session_state.consent_given = False
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, "❌ Privacy denied. Limited mode.")
                st.rerun()
    _consent_dialog()

def session_complete_dialog():
    if st.session_state.get("show_session_complete", False):
        @st.dialog("🎉 Session Complete!")
        def _complete_dialog():
            hist = st.session_state.score_history
            avg = round(sum(hist)/len(hist)) if hist else 0
            alert = st.session_state.alert_count
            duration = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
            mm, ss = duration//60, duration%60
            st.markdown(f"""
            **Amazing job!** You made it to the end of your session.
            - **Duration:** {mm} menit {ss} detik
            - **Average Focus:** {avg}%
            - **Alerts:** {alert} kali
            """)
            if st.button("Start New Session", use_container_width=True):
                st.session_state.session_active = False
                st.session_state.show_session_complete = False
                st.session_state.score_history = []
                st.session_state.alert_count = 0
                st.session_state.low_score_count = 0
                st.session_state.last_alert_time = 0
                st.session_state.session_start = None
                st.rerun()
        _complete_dialog()

def exit_confirmation():
    if st.session_state.get("show_exit_popup", False):
        @st.dialog("Leave Focus Session?")
        def _exit_dialog():
            st.markdown("Your current focus monitoring session will be closed. Are you sure you want to return to the home page?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Stay Here", use_container_width=True):
                    st.session_state.show_exit_popup = False
                    st.rerun()
            with col2:
                if st.button("Leave", use_container_width=True):
                    st.session_state.show_exit_popup = False
                    st.session_state.show_landing = True
                    st.session_state.session_active = False
                    if "webrtc_ctx" in st.session_state:
                        st.session_state.webrtc_ctx = None
                    st.rerun()
        _exit_dialog()

def face_warning_dialog():
    if st.session_state.get("show_face_warning", False):
        @st.dialog("⚠️ Face Not Detected")
        def _face_dialog():
            st.markdown("We are unable to detect your face. Please make sure your face is visible and properly positioned.")
            if st.button("Return to Camera", use_container_width=True):
                st.session_state.show_face_warning = False
                st.rerun()
        _face_dialog()

# ============================================================
# LANDING PAGE
# ============================================================
def show_landing_page():
    st.markdown("""
    <div class="landing-container">
        <div class="landing-content">
            <p class="landing-welcome">Welcome to</p>
            <h1 class="landing-title">
                <span class="dot-pulse"></span>
                FocusWebCam
            </h1>
            <p class="landing-subtitle">Your Personal AI Companion for Unstoppable Focus.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tombol di tengah bawah agar pasti terlihat dan berfungsi
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Let's get started →", key="landing_btn", use_container_width=True):
            st.session_state.show_landing = False
            st.rerun()

# ============================================================
# MAIN APP PAGE
# ============================================================
def show_app_page():
    # Header
    col_logo, col_status = st.columns([3,1])
    with col_logo:
        st.markdown('<div class="logo"><span class="logo-dot"></span><span class="logo-text">FocusWebCam</span></div>', unsafe_allow_html=True)
    with col_status:
        if st.session_state.session_active:
            status = "SESI AKTIF"
        else:
            status = "SIAP — Model LR"
        st.markdown(f'<div class="header-status">{status}</div>', unsafe_allow_html=True)

    cam_col, info_col = st.columns([3,2])

    with cam_col:
        if not st.session_state.session_active:
            if st.button("▶ MULAI SESI", key="start_btn"):
                st.session_state.session_active = True
                st.session_state.session_start = time.time()
                st.session_state.score_history = []
                st.session_state.alert_count = 0
                st.session_state.low_score_count = 0
                st.session_state.last_alert_time = 0
                ts = datetime.now().strftime("%H:%M:%S")
                st.session_state.log_entries.insert(0, f"🎯 [{ts}] Sesi dimulai")
                st.rerun()
        else:
            if st.button("⏹ HENTIKAN SESI", key="stop_btn"):
                hist = st.session_state.score_history
                if hist:
                    avg = round(sum(hist)/len(hist))
                    pct = round(sum(1 for s in hist if s >= ALERT_THRESHOLD)/len(hist)*100)
                    ts = datetime.now().strftime("%H:%M:%S")
                    st.session_state.log_entries.insert(0, f"📊 [{ts}] Selesai — avg {avg}, fokus {pct}%, {st.session_state.alert_count} alert")
                st.session_state.session_active = False
                st.session_state.show_session_complete = True
                st.rerun()

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        ctx = webrtc_streamer(
            key="focus-cam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=FocusVideoProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
        st.session_state.webrtc_ctx = ctx

        st.markdown("""
        <div class="camera-frame" style="position:relative; min-height:360px;">
            <div class="corner tl"></div><div class="corner tr"></div>
            <div class="corner bl"></div><div class="corner br"></div>
            <div class="face-status">Wajah terdeteksi</div>
        </div>
        """, unsafe_allow_html=True)

    with info_col:
        drain_queue()
        score = st.session_state.disp_score
        ear   = st.session_state.disp_ear
        head  = st.session_state.disp_head
        mouth = st.session_state.disp_mouth
        expl  = st.session_state.disp_expl

        if score is not None:
            color_hex = "#00ff88" if score >= 65 else ("#ffcc00" if score >= 40 else "#ff4444")
            state_txt = "FOKUS" if score >= 65 else ("PERHATIAN" if score >= 40 else "TIDAK FOKUS")
        else:
            color_hex, state_txt, score = "#555555", "—", 0
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">FOCUS SCORE</div>
            <div><span class="score-number" style="color:{color_hex}">{score if score is not None else '--'}</span><span class="score-unit">/100</span></div>
            <div class="score-bar-track"><div class="score-bar-fill" style="width:{score if score else 0}%; background:{color_hex}"></div></div>
            <div class="score-state" style="color:{color_hex}">{state_txt}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(score/100 if score else 0)

        f1, f2, f3 = st.columns(3)
        ear_disp = f"{ear:.3f}" if ear is not None else "—"
        head_disp = f"{head:.3f}" if head is not None else "—"
        mouth_disp = f"{mouth:.3f}" if mouth is not None else "—"
        with f1:
            st.markdown(f'<div class="feature-card"><div class="feature-name">EAR (MATA)</div><div class="feature-value">{ear_disp}</div><div class="feature-bar-track"><div class="feature-bar-fill" style="width:{min(ear*400 if ear else 0,100)}%"></div></div></div>', unsafe_allow_html=True)
        with f2:
            st.markdown(f'<div class="feature-card"><div class="feature-name">HEAD POSE</div><div class="feature-value">{head_disp}</div><div class="feature-bar-track"><div class="feature-bar-fill" style="width:{min((1-head)*333 if head else 0,100)}%"></div></div></div>', unsafe_allow_html=True)
        with f3:
            st.markdown(f'<div class="feature-card"><div class="feature-name">MOUTH RATIO</div><div class="feature-value">{mouth_disp}</div><div class="feature-bar-track"><div class="feature-bar-fill" style="width:{min((1-mouth/0.12)*100 if mouth else 0,100)}%"></div></div></div>', unsafe_allow_html=True)

        if expl:
            st.markdown(f'<div class="score-card" style="margin-top:8px"><div class="score-label">📊 EXPLANATION</div><div style="font-size:0.75rem">{expl}</div></div>', unsafe_allow_html=True)

        hist = st.session_state.score_history
        avg_s = round(sum(hist)/len(hist)) if hist else 0
        fpct = round(sum(1 for s in hist if s >= ALERT_THRESHOLD)/len(hist)*100) if hist else 0
        elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
        mm, ss = elapsed//60, elapsed%60
        st.markdown(f"""
        <div class="stats-card">
            <div class="score-label">SESI INI</div>
            <div class="stats-grid">
                <div><div class="stat-value">{mm:02d}:{ss:02d}</div><div class="stat-label">Durasi</div></div>
                <div><div class="stat-value">{avg_s if hist else '--'}</div><div class="stat-label">Rata-rata</div></div>
                <div><div class="stat-value">{fpct if hist else '--'}%</div><div class="stat-label">Fokus</div></div>
                <div><div class="stat-value">{st.session_state.alert_count}</div><div class="stat-label">Alert</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        logs_html = ""
        for entry in st.session_state.log_entries[:20]:
            cls = "log-alert" if "⚠️" in entry else ("log-focus" if "🎯" in entry or "✅" in entry else "")
            logs_html += f'<div class="log-item {cls}">{entry}</div>'
        st.markdown(f'<div class="log-card"><div class="score-label">LOG AKTIVITAS</div><div class="log-list">{logs_html}</div></div>', unsafe_allow_html=True)

        if st.button("← Back", key="back_btn"):
            st.session_state.show_exit_popup = True
            st.rerun()

    st.markdown('<div class="privacy-note">🔒 Data diproses lokal — tidak dikirim ke server</div>', unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================
def main():
    load_css()
    session_complete_dialog()
    exit_confirmation()
    face_warning_dialog()

    if st.session_state.show_landing:
        show_landing_page()
    else:
        show_app_page()
        if st.session_state.get("webrtc_ctx") and st.session_state.webrtc_ctx.state.playing:
            time.sleep(0.5)
            st.rerun()

if __name__ == "__main__":
    main()
