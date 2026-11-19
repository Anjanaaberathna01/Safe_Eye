import time
import cv2 # type: ignore
import numpy as np # type: ignore
import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
from PIL import Image # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore

st.set_page_config(
    page_title="DriveGuard AI — Driver Drowsiness Dashboard",
    page_icon="👁️",
    layout="wide",
)

MODEL_PATH = "training_results/best_drowsiness_model.keras"
CLASS_NAMES = [
    "drowsy",
    "eye_close",
    "eye_open",
    "no_yawn",
    "non_drowsy",
    "yawn",
]
ALERT_CLASSES = {"drowsy", "eye_close", "yawn"}
IMG_SIZE = (224, 224)


@st.cache_resource
def load_drowsiness_model(path: str):
    return tf.keras.models.load_model(path)


def predict_frame(model, rgb_frame: np.ndarray):
    resized = cv2.resize(rgb_frame, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_array = np.expand_dims(resized.astype(np.float32), axis=0)
    processed = preprocess_input(img_array)
    probs = model.predict(processed, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return CLASS_NAMES[top_idx], float(probs[top_idx]), probs


def main():
    st.title("👁️ DriveGuard AI — Driver Drowsiness Detection")
    st.caption("Real-Time Webcam Monitoring with MobileNetV2 (6-Class)")

    # Sidebar Controls
    st.sidebar.header("⚙️ Settings")
    camera_index = st.sidebar.number_input("Webcam Index", min_value=0, max_value=5, value=0, step=1)
    alert_streak_threshold = st.sidebar.slider(
        "Consecutive Alert Frames", min_value=3, max_value=25, value=8,
        help="මෙම Frame ගණන දිගටම Drowsy/Eye_Close/Yawn ආවොත් Alert එක පෙන්වයි."
    )
    confidence_threshold = st.sidebar.slider(
        "Min Confidence (%)", min_value=30, max_value=95, value=60, step=5
    ) / 100.0

    try:
        model = load_drowsiness_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model load කරන්න බැරි වුණා: `{MODEL_PATH}`\n\nError: {e}")
        return

    col_cam, col_info = st.columns([2, 1])

    with col_cam:
        st.subheader("📷 Live Camera Feed")
        run_cam = st.checkbox("Start Live Webcam", value=False)
        frame_placeholder = st.empty()

    with col_info:
        st.subheader("📊 Driver Status")
        status_placeholder = st.empty()
        metric_placeholder = st.empty()
        probs_placeholder = st.empty()
        log_placeholder = st.empty()

    if "alert_streak" not in st.session_state:
        st.session_state.alert_streak = 0
    if "alert_count" not in st.session_state:
        st.session_state.alert_count = 0
    if "events_log" not in st.session_state:
        st.session_state.events_log = []

    if run_cam:
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            st.error("Webcam එක open කරන්න බැරි වුණා. වෙනත් Camera Index (0, 1...) එකක් try කරන්න.")
            return

        try:
            while run_cam:
                ret, bgr_frame = cap.read()
                if not ret:
                    st.warning("Camera එකෙන් Frame එකක් ගන්න බැරි වුණා.")
                    break

                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                pred_class, conf, probs = predict_frame(model, rgb_frame)

                is_risky = (pred_class in ALERT_CLASSES) and (conf >= confidence_threshold)
                if is_risky:
                    st.session_state.alert_streak += 1
                else:
                    st.session_state.alert_streak = 0

                triggered = st.session_state.alert_streak >= alert_streak_threshold
                if triggered:
                    st.session_state.alert_count += 1
                    timestamp = time.strftime("%H:%M:%S")
                    st.session_state.events_log.insert(
                        0, f"[{timestamp}] ⚠️ {pred_class.upper()} ({conf*100:.1f}%)"
                    )
                    st.session_state.events_log = st.session_state.events_log[:8]

                # Frame මත Overlay Text ඇඳීම
                display_frame = rgb_frame.copy()
                label_text = f"{pred_class} ({conf*100:.1f}%)"
                box_color = (220, 38, 38) if triggered else ((234, 179, 8) if is_risky else (22, 163, 74))
                cv2.rectangle(display_frame, (10, 10), (360, 55), box_color, -1)
                cv2.putText(
                    display_frame, label_text, (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2
                )

                frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

                with status_placeholder.container():
                    if triggered:
                        st.error(f"🚨 DROWSINESS ALERT! Detected: **{pred_class}**")
                    elif is_risky:
                        st.warning(f"⚠️ Monitoring Risk: **{pred_class}** ({st.session_state.alert_streak}/{alert_streak_threshold})")
                    else:
                        st.success("✅ Driver Awake & Safe")

                with metric_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Prediction", pred_class)
                    m2.metric("Confidence", f"{conf*100:.1f}%")
                    m3.metric("Alerts", st.session_state.alert_count)

                with probs_placeholder.container():
                    st.markdown("**Class Probabilities**")
                    prob_dict = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
                    st.bar_chart(prob_dict)

                with log_placeholder.container():
                    if st.session_state.events_log:
                        st.markdown("**Recent Alerts**")
                        for item in st.session_state.events_log:
                            st.write(item)
        finally:
            cap.release()
    else:
        st.info("Live Webcam පටන් ගන්න Sidebar එකේ හෝ වම් පස **Start Live Webcam** tick කරන්න.")


if __name__ == "__main__":
    main()