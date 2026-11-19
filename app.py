import time
import cv2  # type: ignore
import numpy as np  # type: ignore
import streamlit as st  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore
import tensorflow as tf  # type: ignore

st.set_page_config(
    page_title="DriveGuard AI — Driver Drowsiness Dashboard",
    page_icon="👁️",
    layout="wide",
)

FACE_MODEL_PATH = "training_results_class/best_4class_drowsiness_model.keras"
FACE_CLASS_NAMES = ["drowsy", "non_drowsy", "no_yawn", "yawn"]
FACE_ALERT_CLASSES = {"drowsy", "yawn"}

EYE_MODEL_PATH = "training_results_eye/best_eye_model.keras"
EYE_CLASS_NAMES = ["eye_close", "eye_open"]
EYE_ALERT_CLASSES = {"eye_close"}

IMG_SIZE = (224, 224)


@st.cache_resource
def load_keras_model(path: str):
    return tf.keras.models.load_model(path)


def run_classifier(model, class_names: list[str], rgb_img: np.ndarray):
    resized = cv2.resize(rgb_img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_array = np.expand_dims(resized.astype(np.float32), axis=0)
    processed = preprocess_input(img_array)
    probs = model.predict(processed, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return class_names[top_idx], float(probs[top_idx]), probs


def extract_eye_region_crop(rgb_frame: np.ndarray):
    """CascadeClassifier නොමැතිව Frame එකේ ඇස් තියෙන කලාපය Crop කරයි."""
    h, w, _ = rgb_frame.shape
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    y1 = int(h * 0.22)
    y2 = int(h * 0.48)
    if x2 > x1 and y2 > y1:
        return rgb_frame[y1:y2, x1:x2], (x1, y1, x2, y2)
    return None, None


def main():
    st.title("👁️ DriveGuard AI — Driver Drowsiness Detection")
    st.caption("Real-Time Webcam Monitoring with 4-Class Full-Face + 2-Class Eye Models (MobileNetV2)")

    # Sidebar Controls
    st.sidebar.header("⚙️ Settings")
    camera_index = st.sidebar.number_input("Webcam Index", min_value=0, max_value=5, value=0, step=1)
    frame_skip = st.sidebar.slider(
        "Predict Every N Frames", min_value=1, max_value=5, value=2,
        help="Laptop speed එක smooth තියාගන්න frames කිහිපයකට වරක් predict කරයි."
    )
    alert_streak_threshold = st.sidebar.slider(
        "Consecutive Alert Frames", min_value=3, max_value=25, value=8,
        help="මෙම Risk Frame ගණන දිගටම ආවොත් Alert එක පෙන්වයි."
    )
    confidence_threshold = (
        st.sidebar.slider("Min Confidence (%)", min_value=30, max_value=95, value=60, step=5)
        / 100.0
    )

    try:
        face_model = load_keras_model(FACE_MODEL_PATH)
    except Exception as e:
        st.error(f"4-Class Face Model load කරන්න බැරි වුණා: `{FACE_MODEL_PATH}`\n\nError: {e}")
        return

    eye_model = None
    try:
        eye_model = load_keras_model(EYE_MODEL_PATH)
        st.sidebar.success("✅ Eye Model Loaded")
    except Exception:
        st.sidebar.warning("⚠️ Eye Model (`best_eye_model.keras`) හමු නොවුණා — 4-Class Face Model එක පමණක් ක්‍රියාත්මක වේ.")

    col_cam, col_info = st.columns([2, 1])

    with col_cam:
        st.subheader("📷 Live Camera Feed")
        run_cam = st.checkbox("Start Live Webcam", value=False)
        frame_placeholder = st.empty()
        eye_crop_placeholder = st.empty()

    with col_info:
        st.subheader("📊 Driver Status")
        status_placeholder = st.empty()
        metric_placeholder = st.empty()
        face_probs_placeholder = st.empty()
        eye_probs_placeholder = st.empty()
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

        frame_idx = 0
        last_face_class, last_face_conf, last_face_probs = "non_drowsy", 0.0, np.zeros(len(FACE_CLASS_NAMES))
        last_eye_class, last_eye_conf, last_eye_probs = None, 0.0, np.zeros(len(EYE_CLASS_NAMES))
        last_eye_box = None
        last_eye_crop = None

        try:
            while run_cam:
                ret, bgr_frame = cap.read()
                if not ret:
                    st.warning("Camera එකෙන් Frame එකක් ගන්න බැරි වුණා.")
                    break

                frame_idx += 1
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

                if frame_idx % frame_skip == 0:
                    last_face_class, last_face_conf, last_face_probs = run_classifier(
                        face_model, FACE_CLASS_NAMES, rgb_frame
                    )

                    if eye_model is not None:
                        eye_crop, eye_box = extract_eye_region_crop(rgb_frame)
                        last_eye_box = eye_box
                        last_eye_crop = eye_crop
                        if eye_crop is not None:
                            last_eye_class, last_eye_conf, last_eye_probs = run_classifier(
                                eye_model, EYE_CLASS_NAMES, eye_crop
                            )
                        else:
                            last_eye_class = None

                face_risky = (last_face_class in FACE_ALERT_CLASSES) and (last_face_conf >= confidence_threshold)
                eye_risky = (
                    (last_eye_class in EYE_ALERT_CLASSES)
                    and (last_eye_conf >= confidence_threshold)
                )
                is_risky = face_risky or eye_risky

                risk_reasons = []
                if face_risky:
                    risk_reasons.append(f"{last_face_class} ({last_face_conf*100:.1f}%)")
                if eye_risky:
                    risk_reasons.append(f"{last_eye_class} ({last_eye_conf*100:.1f}%)")
                risk_summary = " + ".join(risk_reasons) if risk_reasons else f"{last_face_class} ({last_face_conf*100:.1f}%)"

                if is_risky:
                    st.session_state.alert_streak += 1
                else:
                    st.session_state.alert_streak = 0

                triggered = st.session_state.alert_streak >= alert_streak_threshold
                if triggered:
                    st.session_state.alert_count += 1
                    timestamp = time.strftime("%H:%M:%S")
                    st.session_state.events_log.insert(
                        0, f"[{timestamp}] ⚠️ {risk_summary.upper()}"
                    )
                    st.session_state.events_log = st.session_state.events_log[:8]

                display_frame = rgb_frame.copy()
                if last_eye_box is not None:
                    x1, y1, x2, y2 = last_eye_box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (59, 130, 246), 2)

                overlay_label = f"Face: {last_face_class} ({last_face_conf*100:.0f}%)"
                if last_eye_class is not None:
                    overlay_label += f" | Eye: {last_eye_class} ({last_eye_conf*100:.0f}%)"

                box_color = (220, 38, 38) if triggered else ((234, 179, 8) if is_risky else (22, 163, 74))
                cv2.rectangle(display_frame, (10, 10), (min(display_frame.shape[1] - 10, 620), 52), box_color, -1)
                cv2.putText(
                    display_frame, overlay_label, (18, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2
                )

                frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

                if last_eye_crop is not None:
                    eye_crop_placeholder.caption("👁️ Cropped Eye Region -> Eye Model")
                    eye_crop_placeholder.image(last_eye_crop, channels="RGB", width=180)
                else:
                    eye_crop_placeholder.empty()

                with status_placeholder.container():
                    if triggered:
                        st.error(f"🚨 DROWSINESS ALERT! Detected: **{risk_summary}**")
                    elif is_risky:
                        st.warning(
                            f"⚠️ Monitoring Risk: **{risk_summary}** "
                            f"({st.session_state.alert_streak}/{alert_streak_threshold})"
                        )
                    else:
                        st.success("✅ Driver Awake & Safe")

                with metric_placeholder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Face State", last_face_class)
                    m2.metric("Eye State", last_eye_class if last_eye_class else "N/A")
                    m3.metric("Alerts", st.session_state.alert_count)

                with face_probs_placeholder.container():
                    st.markdown("**4-Class Face Probabilities**")
                    face_prob_dict = {cls: float(p) for cls, p in zip(FACE_CLASS_NAMES, last_face_probs)}
                    st.bar_chart(face_prob_dict)

                if eye_model is not None and last_eye_class is not None:
                    with eye_probs_placeholder.container():
                        st.markdown("**2-Class Eye Probabilities**")
                        eye_prob_dict = {cls: float(p) for cls, p in zip(EYE_CLASS_NAMES, last_eye_probs)}
                        st.bar_chart(eye_prob_dict)
                else:
                    eye_probs_placeholder.empty()

                with log_placeholder.container():
                    if st.session_state.events_log:
                        st.markdown("**Recent Alerts**")
                        for item in st.session_state.events_log:
                            st.write(item)
        finally:
            cap.release()
    else:
        st.info("Live Webcam පටන් ගන්න වම් පස **Start Live Webcam** tick කරන්න.")


if __name__ == "__main__":
    main()