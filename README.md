# DriveGuard AI

Real-time driver drowsiness detection dashboard built with Streamlit. It watches the driver through a webcam, uses two deep learning models to check the face and the eyes, and raises an alert when signs of drowsiness keep appearing.

## Features

- Live webcam monitoring with a clean dark dashboard
- Two MobileNetV2 models working together: a 4-class face model and a 2-class eye model
- Status states: driver alert, signs of fatigue, drowsiness alert
- Live risk score from 0 to 100, smoothed over recent checks
- Eyes-closed percentage over the last 60 seconds
- Risk timeline, model confidence bars and a log of recent alerts
- Optional alert sound
- Adjustable settings in the sidebar (camera, detection speed, sensitivity)

## How it works

1. **Capture**: OpenCV reads each frame from the webcam (optionally mirrored).
2. **Predict**: every N frames, the frame goes to the face model. A fixed region around the eyes is cropped and sent to the eye model.
3. **Check**: a check counts as risky when either model reports a risky class with confidence above the minimum confidence setting.
4. **Score**: results feed a smoothed risk score and an eyes-closed percentage.
5. **Alert**: when the number of consecutive risky checks reaches the set limit, the dashboard turns red, plays a beep and writes an entry in the alert log. Each drowsy episode is counted as one alert.

### Models

| Model | Classes | Risky classes | Default path |
| --- | --- | --- | --- |
| Face model (MobileNetV2) | `drowsy`, `non_drowsy`, `no_yawn`, `yawn` | `drowsy`, `yawn` | `training_results_class/best_4class_drowsiness_model.keras` |
| Eye model (MobileNetV2) | `eye_close`, `eye_open` | `eye_close` | `training_results_eye/best_eye_model.keras` |

The eye model is optional. If its file is missing, the app still runs with the face model only and eye detection is switched off.

## Project structure

```
.
├── driveguard_dashboard.py
├── training_results_class/
│   └── best_4class_drowsiness_model.keras
└── training_results_eye/
    └── best_eye_model.keras
```

Put your trained `.keras` files in the folders above, or change `FACE_MODEL_PATH` and `EYE_MODEL_PATH` at the top of `driveguard_dashboard.py`.

## Installation

Requires Python 3.9 or newer and a working webcam.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install tensorflow opencv-python numpy streamlit
```

## Run

```bash
streamlit run driveguard_dashboard.py
```

Open `http://localhost:8501`, then turn on the **Camera** switch at the top right.

## Settings

| Setting | What it does |
| --- | --- |
| Webcam index | Which camera to use. `0` is usually the built-in webcam. |
| Mirror view | Flips the video like a mirror. |
| Show eye crop | Shows the eye region that is sent to the eye model. |
| Eye box size (%) | Size of the eye region box. |
| Predict every N frames | Higher values are faster but check less often. With a 30 FPS webcam, `2` gives about 15 checks per second. |
| Consecutive risky checks | How many risky checks in a row trigger an alert. Lower is more sensitive. |
| Minimum confidence (%) | A model result below this level is ignored. |
| Play alert sound | Beeps while an alert is active. Some browsers only allow sound after you click on the page. |
| Reset session | Clears the alert count, timeline and log. |

## Tips for better results

- Place the camera at eye level and keep the face centered. The eye region is a fixed crop of the frame, so a face far off-center can give wrong eye results.
- Use even lighting on the face. Strong backlight and deep shadows lower accuracy.
- If you get false alerts, raise **Minimum confidence** or **Consecutive risky checks**. If alerts come too late, lower them.

## Known limitations

- The eye region is a fixed crop, not found with a face or landmark detector. Adding a detector such as MediaPipe would make it follow the face.
- Results depend on how well the models were trained and on lighting and camera position.
- Tested with a laptop webcam. Performance on other hardware will vary.

## Safety notice

DriveGuard AI is a learning and research project. It is not a certified safety device and must not be relied on to keep a driver safe. If you feel sleepy while driving, stop in a safe place and rest.

## Built with

- [TensorFlow / Keras](https://www.tensorflow.org/) (MobileNetV2)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)

## License

Add your license here (for example MIT).
