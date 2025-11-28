import threading
import time
import cv2
import numpy as np
import torch
import timm
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, request

YOLO_MODEL_PATH = "driver.pt"
VIT_WEIGHTS_PATH = "deit_drowsiness_model.pth"
VIT_ARCH = "deit_small_patch16_224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INFERENCE_INTERVAL = 0.25   # ~4 FPS inference
STREAM_SLEEP = 0.03         # ~30 FPS output
PERSIST_SECONDS = 5.0

VIT_CLASS_NAMES = ["class_0", "class_1", "class_2", "class_3"]
ALARM_INDICES = {0, 2}

# APP 
app = Flask(__name__, template_folder="templates", static_folder="static")
_latest_frame = None
_frame_lock = threading.Lock()

_capture_thread = None
_infer_thread = None
_running_event = threading.Event()  

_status_lock = threading.Lock()
_current_prediction = "idle"
_persistent_start_ts = None
_alarm_flag = False

_camera = None

print("Loading YOLO model from", YOLO_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)
try:
    yolo_names = yolo_model.names
except Exception:
    yolo_names = None

print("Loading DeiT model from", VIT_WEIGHTS_PATH)
ckpt = torch.load(VIT_WEIGHTS_PATH, map_location="cpu")
if isinstance(ckpt, dict) and "head.weight" in ckpt:
    ncls = ckpt["head.weight"].shape[0]
else:
    ncls = len(VIT_CLASS_NAMES)

vit_model = timm.create_model(VIT_ARCH, pretrained=False, num_classes=ncls)
vit_model.load_state_dict(ckpt, strict=False)
vit_model.to(DEVICE)
vit_model.eval()

# preprocessing
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_vit(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
    return tensor

def camera_capture_loop(device_index=0):
    """Continuously read frames from camera and store the latest in _latest_frame."""
    global _camera, _latest_frame
    print("Camera capture thread starting, opening device", device_index)
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    # try to reduce internal buffer
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    _camera = cap
    while _running_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        with _frame_lock:
            _latest_frame = frame
    try:
        cap.release()
    except Exception:
        pass
    _camera = None
    print("Camera capture thread exiting")

def inference_loop(inference_interval=INFERENCE_INTERVAL):
    """Run YOLO -> crop -> DeiT at a lower frequency and update alarm state."""
    global _latest_frame, _current_prediction, _persistent_start_ts, _alarm_flag
    print("Inference thread starting, interval:", inference_interval)
    last_inf = 0.0
    while _running_event.is_set():
        now = time.time()
        if now - last_inf < inference_interval:
            time.sleep(0.005)
            continue
        last_inf = now

        with _frame_lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
        if frame is None:
            continue

        predicted_label = "no_detection"
        try:
            results = yolo_model(frame, verbose=False)
            boxes = results[0].boxes
        except Exception as e:
            boxes = None
            print("YOLO error:", e)

        if boxes is not None and len(boxes) > 0:
            best_xy = None
            best_conf = -1.0
            for b in boxes:
                try:
                    xy = b.xyxy[0].cpu().numpy().astype(int)
                    conf = float(b.conf.cpu().numpy()) if hasattr(b, "conf") else float(b.conf)
                except Exception:
                    try:
                        xy = b.xyxy[0].cpu().numpy().astype(int)
                        conf = float(b.conf)
                    except Exception:
                        continue
                if conf > best_conf:
                    best_conf = conf
                    best_xy = xy
            if best_xy is not None:
                x1, y1, x2, y2 = best_xy
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    crop = frame[y1:y2, x1:x2]
                    try:
                        inp = preprocess_vit(crop)
                        with torch.no_grad():
                            out = vit_model(inp)
                            idx = int(out.argmax(dim=1).item())
                            if idx < len(VIT_CLASS_NAMES):
                                predicted_label = VIT_CLASS_NAMES[idx]
                            else:
                                predicted_label = f"class_{idx}"
                    except Exception as e:
                        predicted_label = "infer_error"
                        print("ViT error:", e)
                else:
                    predicted_label = "crop_small"
            else:
                predicted_label = "no_box"
        else:
            predicted_label = "no_detection"

        with _status_lock:
            _current_prediction = predicted_label
            try:
                idx_guess = None
                if predicted_label in VIT_CLASS_NAMES:
                    idx_guess = VIT_CLASS_NAMES.index(predicted_label)
                elif predicted_label.startswith("class_"):
                    try:
                        idx_guess = int(predicted_label.split("_", 1)[1])
                    except Exception:
                        idx_guess = None
                if idx_guess is not None and idx_guess in ALARM_INDICES:
                    if _persistent_start_ts is None:
                        _persistent_start_ts = time.time()
                    else:
                        if time.time() - _persistent_start_ts >= PERSIST_SECONDS:
                            _alarm_flag = True
                else:
                    _alarm_flag = False
                    _persistent_start_ts = None
            except Exception as e:
                # safety reset on error
                _alarm_flag = False
                _persistent_start_ts = None
                print("Persistent logic error:", e)

    print("Inference thread exiting")

def stream_generator():
    global _latest_frame
    while _running_event.is_set():
        with _frame_lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            time.sleep(0.005)
            continue
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(STREAM_SLEEP)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global _capture_thread, _infer_thread, _running_event
    if _running_event.is_set():
        return jsonify({"status": "already_running"})
    _running_event.set()
    _capture_thread = threading.Thread(target=camera_capture_loop, args=(0,), daemon=True)
    _capture_thread.start()
    _infer_thread = threading.Thread(target=inference_loop, daemon=True)
    _infer_thread.start()
    return jsonify({"status": "started"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global _running_event, _latest_frame, _alarm_flag, _persistent_start_ts
    if not _running_event.is_set():
        return jsonify({"status": "not_running"})
    _running_event.clear()
    with _frame_lock:
        _latest_frame = None
    with _status_lock:
        _alarm_flag = False
        _persistent_start_ts = None
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    if not _running_event.is_set():
        return "Camera is not running", 503
    return Response(stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    with _status_lock:
        return jsonify({
            "prediction": _current_prediction,
            "alarm": bool(_alarm_flag)
        })

# Run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
