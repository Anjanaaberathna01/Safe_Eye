import time
from collections import deque

import cv2  # type: ignore
import numpy as np  # type: ignore
import streamlit as st  # type: ignore
import streamlit.components.v1 as components  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="DriveGuard AI | Driver drowsiness dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FACE_MODEL_PATH = "training_results_class/best_4class_drowsiness_model.keras"
FACE_CLASS_NAMES = ["drowsy", "non_drowsy", "no_yawn", "yawn"]
FACE_ALERT_CLASSES = {"drowsy", "yawn"}

EYE_MODEL_PATH = "training_results_eye/best_eye_model.keras"
EYE_CLASS_NAMES = ["eye_close", "eye_open"]
EYE_ALERT_CLASSES = {"eye_close"}

IMG_SIZE = (224, 224)
PERCLOS_WINDOW_SEC = 60
HISTORY_POINTS = 240

LEVEL_COLORS = {
    "idle": "#64748B",
    "safe": "#34D399",
    "watch": "#FBBF24",
    "alert": "#F43F5E",
}
LEVEL_BGR = {  # for OpenCV overlays (RGB frame, so same order as hex)
    "idle": (100, 116, 139),
    "safe": (52, 211, 153),
    "watch": (251, 191, 36),
    "alert": (244, 63, 94),
}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #0B121B;
  --panel: #131D2A;
  --line: #213044;
  --text: #E6EDF5;
  --muted: #8A9BB0;
  --accent: #38BDF8;
}

html, body, [class*="css"], .stApp {
  font-family: 'Manrope', system-ui, sans-serif;
  color: var(--text);
}
.stApp { background: var(--bg); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }

[data-testid="stSidebar"] { background: #0F1722; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 0.95rem; color: var(--text); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: var(--muted); }

[data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--line); }

.dg-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
.dg-title { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.dg-sub { color: var(--muted); font-size: 0.92rem; margin-top: 4px; }
.dg-pill { display: inline-flex; align-items: center; gap: 8px; padding: 7px 14px; border-radius: 999px;
           border: 1px solid var(--line); background: var(--panel); font-weight: 600; font-size: 0.85rem; }
.dg-dot { width: 9px; height: 9px; border-radius: 50%; }
.dg-dot.pulse { animation: dgpulse 1.4s ease-out infinite; }
@keyframes dgpulse { 0% { box-shadow: 0 0 0 0 rgba(244,63,94,.6); } 100% { box-shadow: 0 0 0 12px rgba(244,63,94,0); } }
@media (prefers-reduced-motion: reduce) { .dg-dot.pulse { animation: none; } }

.dg-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; }
.dg-panel h4 { margin: 0 0 12px 0; font-size: 0.95rem; font-weight: 700; color: var(--text); }

.dg-status { display: flex; gap: 18px; align-items: center; border-left-width: 6px; border-left-style: solid; }
.dg-status .t { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.01em; }
.dg-status .d { color: var(--muted); font-size: 0.9rem; margin-top: 4px; line-height: 1.4; }

.dg-tile { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.dg-tile .l { color: var(--muted); font-size: 0.8rem; font-weight: 600; }
.dg-tile .v { font-size: 1.7rem; font-weight: 800; font-variant-numeric: tabular-nums; margin-top: 2px; letter-spacing: -0.02em; }
.dg-tile .s { color: var(--muted); font-size: 0.78rem; margin-top: 2px; }

.dg-row { display: grid; grid-template-columns: 92px 1fr 48px; align-items: center; gap: 10px; margin: 9px 0; font-size: 0.85rem; }
.dg-row .n { color: var(--muted); font-weight: 600; }
.dg-track { height: 8px; border-radius: 99px; background: #1B2A3D; overflow: hidden; }
.dg-fill { height: 100%; border-radius: 99px; }
.dg-row .p { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
.dg-group { color: var(--muted); font-size: 0.78rem; font-weight: 700; margin: 14px 0 2px 0; }
.dg-group:first-of-type { margin-top: 0; }

.dg-log { display: flex; flex-direction: column; gap: 8px; }
.dg-log .it { display: flex; gap: 10px; align-items: baseline; padding: 9px 12px; border-radius: 8px; background: #1A1B2A;
              border: 1px solid #3A2030; font-size: 0.85rem; }
.dg-log .tm { color: var(--muted); font-variant-numeric: tabular-nums; font-weight: 600; }
.dg-empty { color: var(--muted); font-size: 0.88rem; padding: 6px 0; }

.dg-idle { height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
           border: 1px dashed var(--line); border-radius: 14px; background: #0F1722; color: var(--muted); gap: 6px; }
.dg-idle b { color: var(--text); font-size: 1.1rem; }
</style>
"""

SOUND_HTML = """
<script>
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  function beep() {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = 'sine'; o.frequency.value = 880; g.gain.value = 0.15;
    o.connect(g); g.connect(ctx.destination);
    o.start(); o.stop(ctx.currentTime + 0.35);
  }
  beep(); setInterval(beep, 900);
</script>
"""


def html(s: str) -> str:
    """Collapse indentation/newlines so Markdown never treats the HTML as a code block."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())


# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def load_keras_model(path: str):
    return tf.keras.models.load_model(path)


def run_classifier(model, class_names, rgb_img):
    resized = cv2.resize(rgb_img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_array = np.expand_dims(resized.astype(np.float32), axis=0)
    processed = preprocess_input(img_array)
    probs = model.predict(processed, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return class_names[top_idx], float(probs[top_idx]), probs

# ----------------------------------------------------------------------------
# Image processing  
# ----------------------------------------------------------------------------

def extract_eye_region_crop(rgb_frame, cx_pct=0.43, cy_pct=0.38, box_pct=0.14):
    h, w, _ = rgb_frame.shape
    half = int(w * box_pct * 0.5)
    cx = int(w * cx_pct)
    cy = int(h * cy_pct)
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    if x2 > x1 and y2 > y1:
        return rgb_frame[y1:y2, x1:x2], (x1, y1, x2, y2)
    return None, None


# ----------------------------------------------------------------------------
# UI builders
# ----------------------------------------------------------------------------
def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def header_html(level: str, running: bool) -> str:
    label = {
        "idle": "Camera off",
        "safe": "Monitoring",
        "watch": "Monitoring",
        "alert": "Alert active",
    }[level]
    pulse = " pulse" if level == "alert" else ""
    return html(f"""
        <div class="dg-top">
          <div>
            <div class="dg-title">DriveGuard AI</div>
            <div class="dg-sub">Live driver drowsiness monitoring with a face model and an eye model</div>
          </div>
          <div class="dg-pill"><span class="dg-dot{pulse}" style="background:{LEVEL_COLORS[level]}"></span>{label}</div>
        </div>
    """)


def gauge_svg(score: float, color: str) -> str:
    circumference = 2 * 3.14159 * 54
    dash = circumference * max(0.0, min(score, 100.0)) / 100
    return html(f"""
        <svg viewBox="0 0 140 140" width="132" height="132" role="img" aria-label="Risk score {score:.0f} out of 100">
          <circle cx="70" cy="70" r="54" fill="none" stroke="#213044" stroke-width="12"/>
          <circle cx="70" cy="70" r="54" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"
                  stroke-dasharray="{dash:.1f} {circumference:.1f}" transform="rotate(-90 70 70)"/>
          <text x="70" y="76" text-anchor="middle" fill="#E6EDF5" font-size="34" font-weight="800"
                font-family="Manrope, sans-serif">{score:.0f}</text>
          <text x="70" y="96" text-anchor="middle" fill="#8A9BB0" font-size="11"
                font-family="Manrope, sans-serif">risk score</text>
        </svg>
    """)


def status_html(level, score, risk_summary, streak, threshold) -> str:
    color = LEVEL_COLORS[level]
    if level == "alert":
        title = "Drowsiness alert"
        detail = f"Detected {risk_summary}. Pull over safely and take a break."
    elif level == "watch":
        title = "Signs of fatigue"
        detail = f"Watching {risk_summary}. Alert triggers at {streak} of {threshold} checks."
    elif level == "safe":
        title = "Driver alert"
        detail = "No signs of drowsiness."
    else:
        title = "Monitoring is off"
        detail = "Turn on the camera switch to start."
    return html(f"""
        <div class="dg-panel dg-status" style="border-left-color:{color}">
          {gauge_svg(score, color)}
          <div><div class="t" style="color:{color}">{title}</div><div class="d">{detail}</div></div>
        </div>
    """)


def tile_html(label, value, sub="") -> str:
    return html(f"""
        <div class="dg-tile"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>
    """)


def bars_html(face_probs, eye_probs, eye_enabled) -> str:
    def rows(names, probs, alert_set):
        out = ""
        for n, p in zip(names, probs):
            color = "#F43F5E" if n in alert_set else "#38BDF8"
            out += (
                f'<div class="dg-row"><span class="n">{n.replace("_", " ")}</span>'
                f'<div class="dg-track"><div class="dg-fill" style="width:{p * 100:.0f}%;background:{color}"></div></div>'
                f'<span class="p">{p * 100:.0f}%</span></div>'
            )
        return out

    body = '<div class="dg-group">Face model</div>' + rows(FACE_CLASS_NAMES, face_probs, FACE_ALERT_CLASSES)
    if eye_enabled:
        body += '<div class="dg-group">Eye model</div>' + rows(EYE_CLASS_NAMES, eye_probs, EYE_ALERT_CLASSES)
    return html(f'<div class="dg-panel"><h4>Model confidence</h4>{body}</div>')


def log_html(events) -> str:
    if events:
        items = "".join(
            f'<div class="it"><span class="tm">{t}</span><span>{reason}</span></div>' for t, reason in events
        )
        body = f'<div class="dg-log">{items}</div>'
    else:
        body = '<div class="dg-empty">No alerts this session.</div>'
    return html(f'<div class="dg-panel"><h4>Recent alerts</h4>{body}</div>')


def timeline_html(history, threshold_line, color) -> str:
    w, h = 600, 150
    if len(history) < 2:
        body = '<div class="dg-empty">The risk timeline appears once monitoring starts.</div>'
    else:
        n = len(history)
        pts = [(i / (n - 1) * w, h - (v / 100) * h) for i, v in enumerate(history)]
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        area = f"0,{h} " + line + f" {w},{h}"
        ty = h - (threshold_line / 100) * h
        body = f"""
            <svg viewBox="0 0 {w} {h}" width="100%" height="170" preserveAspectRatio="none" role="img"
                 aria-label="Risk score over the last few minutes">
              <line x1="0" y1="{ty:.1f}" x2="{w}" y2="{ty:.1f}" stroke="#F43F5E" stroke-width="1" stroke-dasharray="5 5" opacity="0.6"/>
              <polygon points="{area}" fill="{color}" opacity="0.14"/>
              <polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"
                        vector-effect="non-scaling-stroke"/>
            </svg>
            <div class="dg-sub" style="margin-top:6px">Dashed line marks the high-risk level.</div>
        """
    return html(f'<div class="dg-panel"><h4>Risk over time</h4>{body}</div>')


def idle_video_html() -> str:
    return html("""
        <div class="dg-idle"><b>Camera is off</b>
        <span>Turn on the switch in the top right to start monitoring.</span></div>
    """)


def show_image(placeholder, img, **kwargs):
    try:
        placeholder.image(img, channels="RGB", width="stretch", **kwargs)
    except Exception:
        placeholder.image(img, channels="RGB", use_container_width=True, **kwargs)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def init_state():
    defaults = {
        "alert_count": 0,
        "events_log": [],
        "history": [],
        "elapsed": 0.0,
        "last_score": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    # ---------------- Sidebar ----------------
    st.sidebar.subheader("Camera")
    camera_index = st.sidebar.number_input("Webcam index", min_value=0, max_value=5, value=0, step=1)
    mirror = st.sidebar.toggle("Mirror view", value=True)
    show_eye_crop = st.sidebar.toggle("Show eye crop", value=True)

    st.sidebar.subheader("Eye crop box")
    eye_x = st.sidebar.slider("Eye box X (%)", 20, 80, 43) / 100.0
    eye_y = st.sidebar.slider("Eye box Y (%)", 15, 70, 38) / 100.0
    eye_size = st.sidebar.slider("Eye box size (%)", 8, 30, 14) / 100.0

    st.sidebar.subheader("Detection")
    frame_skip = st.sidebar.slider(
        "Predict every N frames", 1, 5, 2,
        help="A 30 FPS webcam predicting every 2 frames gives about 15 checks per second.",
    )
    alert_streak_threshold = st.sidebar.slider(
        "Consecutive risky checks", 3, 25, 8,
        help="How many risky checks in a row trigger an alert.",
    )
    confidence_threshold = st.sidebar.slider("Minimum confidence (%)", 30, 95, 60, step=5) / 100.0

    st.sidebar.subheader("Alerts")
    sound_on = st.sidebar.toggle("Play alert sound", value=True)
    if st.sidebar.button("Reset session", use_container_width=True):
        for k in ["alert_count", "events_log", "history", "elapsed", "last_score"]:
            st.session_state.pop(k, None)
        init_state()

    # ---------------- Models ----------------
    try:
        face_model = load_keras_model(FACE_MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load the face model at `{FACE_MODEL_PATH}`.\n\n{e}")
        return

    eye_model = None
    try:
        eye_model = load_keras_model(EYE_MODEL_PATH)
        st.sidebar.success("Eye model loaded")
    except Exception:
        st.sidebar.warning("Eye model not found. Eye detection is off.")

    # ---------------- Layout ----------------
    header_ph = st.empty()
    _, toggle_col = st.columns([5, 1])
    with toggle_col:
        run_cam = st.toggle("Camera", value=False)

    sound_ph = st.empty()

    col_video, col_side = st.columns([2.1, 1], gap="large")
    with col_video:
        video_ph = st.empty()
    with col_side:
        status_ph = st.empty()
        eye_ph = st.empty()

    k1, k2, k3, k4 = st.columns(4)
    kpi_phs = [k1.empty(), k2.empty(), k3.empty(), k4.empty()]

    c_chart, c_bars, c_log = st.columns([1.5, 1, 1], gap="large")
    chart_ph = c_chart.empty()
    bars_ph = c_bars.empty()
    log_ph = c_log.empty()

    def render_panels(level, score, risk_summary, streak, fps, perclos, face_probs, eye_probs):
        header_ph.markdown(header_html(level, run_cam), unsafe_allow_html=True)
        status_ph.markdown(
            status_html(level, score, risk_summary, streak, alert_streak_threshold), unsafe_allow_html=True
        )
        kpi_phs[0].markdown(tile_html("Session time", fmt_time(st.session_state.elapsed)), unsafe_allow_html=True)
        kpi_phs[1].markdown(
            tile_html("Alerts", st.session_state.alert_count, "this session"), unsafe_allow_html=True
        )
        kpi_phs[2].markdown(
            tile_html("Eyes closed", f"{perclos:.0f}%", f"last {PERCLOS_WINDOW_SEC} seconds"),
            unsafe_allow_html=True,
        )
        kpi_phs[3].markdown(tile_html("Speed", f"{fps:.0f} FPS", "video feed"), unsafe_allow_html=True)
        chart_ph.markdown(
            timeline_html(st.session_state.history, 60, "#38BDF8"),
            unsafe_allow_html=True,
        )
        bars_ph.markdown(bars_html(face_probs, eye_probs, eye_model is not None), unsafe_allow_html=True)
        log_ph.markdown(log_html(st.session_state.events_log), unsafe_allow_html=True)

    zero_face = np.zeros(len(FACE_CLASS_NAMES))
    zero_eye = np.zeros(len(EYE_CLASS_NAMES))

    # ---------------- Idle state ----------------
    if not run_cam:
        video_ph.markdown(idle_video_html(), unsafe_allow_html=True)
        render_panels("idle", st.session_state.last_score, "", 0, 0.0, 0.0, zero_face, zero_eye)
        return

    # ---------------- Live monitoring ----------------
    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        video_ph.markdown(
            html("""<div class="dg-idle"><b>Cannot open the webcam</b>
            <span>Check the webcam index in the sidebar and make sure no other app is using the camera.</span></div>"""),
            unsafe_allow_html=True,
        )
        render_panels("idle", 0.0, "", 0, 0.0, 0.0, zero_face, zero_eye)
        return

    session_start = time.time() - st.session_state.elapsed
    frame_idx = 0
    streak = 0
    was_triggered = False
    score = st.session_state.last_score
    fps = 0.0
    t_prev = time.time()
    closed_window = deque()  # (timestamp, 0/1)
    perclos = 0.0

    last_face_class, last_face_conf, last_face_probs = "non_drowsy", 0.0, zero_face
    last_eye_class, last_eye_conf, last_eye_probs = None, 0.0, zero_eye
    last_eye_box, last_eye_crop = None, None
    is_risky, risk_summary, level = False, "", "safe"

    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                video_ph.markdown(
                    html("""<div class="dg-idle"><b>Lost the camera feed</b>
                    <span>Turn the camera switch off and on to reconnect.</span></div>"""),
                    unsafe_allow_html=True,
                )
                break

            now = time.time()
            dt = max(now - t_prev, 1e-3)
            t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            st.session_state.elapsed = now - session_start

            frame_idx += 1
            if mirror:
                bgr_frame = cv2.flip(bgr_frame, 1)
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            predicted = frame_idx % frame_skip == 0
            if predicted:
                last_face_class, last_face_conf, last_face_probs = run_classifier(
                    face_model, FACE_CLASS_NAMES, rgb_frame
                )
                if eye_model is not None:
                    # [CHANGED] Slider values සමඟ single-eye crop call කළා
                    eye_crop, eye_box = extract_eye_region_crop(rgb_frame, eye_x, eye_y, eye_size)
                    last_eye_box, last_eye_crop = eye_box, eye_crop
                    if eye_crop is not None:
                        last_eye_class, last_eye_conf, last_eye_probs = run_classifier(
                            eye_model, EYE_CLASS_NAMES, eye_crop
                        )
                    else:
                        last_eye_class = None

                face_risky = last_face_class in FACE_ALERT_CLASSES and last_face_conf >= confidence_threshold
                eye_risky = (
                    last_eye_class in EYE_ALERT_CLASSES and last_eye_conf >= confidence_threshold
                    if last_eye_class is not None else False
                )
                is_risky = face_risky or eye_risky

                reasons = []
                if face_risky:
                    reasons.append(f"{last_face_class.replace('_', ' ')} ({last_face_conf * 100:.0f}%)")
                if eye_risky:
                    reasons.append(f"{last_eye_class.replace('_', ' ')} ({last_eye_conf * 100:.0f}%)")
                risk_summary = " and ".join(reasons) if reasons else f"{last_face_class.replace('_', ' ')}"

                streak = streak + 1 if is_risky else 0

                # Smoothed risk score (0-100)
                score = 0.8 * score + 0.2 * (100.0 if is_risky else 0.0)

                # Eyes-closed percentage over the last minute (PERCLOS)
                closed = eye_risky if eye_model is not None else (face_risky and last_face_class == "drowsy")
                closed_window.append((now, 1 if closed else 0))
                while closed_window and now - closed_window[0][0] > PERCLOS_WINDOW_SEC:
                    closed_window.popleft()
                perclos = 100.0 * sum(c for _, c in closed_window) / max(len(closed_window), 1)

                st.session_state.history.append(score)
                st.session_state.history = st.session_state.history[-HISTORY_POINTS:]
                st.session_state.last_score = score

            triggered = streak >= alert_streak_threshold
            if triggered and not was_triggered:
                # count one alert per episode, not one per frame
                st.session_state.alert_count += 1
                st.session_state.events_log.insert(0, (time.strftime("%H:%M:%S"), risk_summary))
                st.session_state.events_log = st.session_state.events_log[:6]
                if sound_on:
                    with sound_ph:
                        components.html(SOUND_HTML, height=0)
            if not triggered:
                sound_ph.empty()
            was_triggered = triggered

            level = "alert" if triggered else ("watch" if (is_risky or score >= 35) else "safe")

            # ---- Video overlay ----
            display = rgb_frame.copy()
            color = LEVEL_BGR[level]
            if last_eye_box is not None:
                x1, y1, x2, y2 = last_eye_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (56, 189, 248), 2)
            if triggered:
                cv2.rectangle(display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1), color, 8)
            label = f"Face {last_face_class.replace('_', ' ')} {last_face_conf * 100:.0f}%"
            if last_eye_class is not None:
                label += f"  |  Eye {last_eye_class.replace('_', ' ')} {last_eye_conf * 100:.0f}%"
            cv2.rectangle(display, (12, 12), (min(display.shape[1] - 12, 560), 50), color, -1)
            cv2.putText(display, label, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (11, 18, 27), 2, cv2.LINE_AA)
            show_image(video_ph, display)

            # ---- Panels (only when new predictions arrive, keeps the video smooth) ----
            if predicted:
                render_panels(level, score, risk_summary, streak, fps, perclos, last_face_probs, last_eye_probs)
                if show_eye_crop and last_eye_crop is not None:
                    with eye_ph.container():
                        st.markdown('<div class="dg-sub" style="margin:10px 0 6px 0">Eye region sent to the eye model</div>',
                                    unsafe_allow_html=True)
                        st.image(last_eye_crop, channels="RGB", width=220)
                else:
                    eye_ph.empty()
    finally:
        cap.release()
        sound_ph.empty()


if __name__ == "__main__":
    main()