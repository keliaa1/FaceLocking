# main_tracker.py
"""
Face Tracker with Servo Control
- Searches by sweeping servo left-right until Kelia is found
- Locks onto Kelia and follows her face direction
- Optimized: recognition every N frames, async servo, faster detection
"""

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import requests
import time
import threading
from pathlib import Path

# ============================================================
# CONFIGURATION - EDIT THESE
# ============================================================
ESP8266_IP = "10.12.74.240"       # <-- REPLACE with your ESP8266 IP
TARGET_NAME = "Kelia"              # <-- Your enrolled name
SIMILARITY_THRESHOLD = 0.55        # Adjust based on evaluate.py results
SMOOTHING_FACTOR = 0.35            # Snappiness of face-follow (0=frozen, 1=instant)
FRAME_WIDTH = 640                  # Resize frame for speed
FRAME_HEIGHT = 480
RECOGNIZE_EVERY_N = 6              # Run ArcFace only every N frames
MESH_EVERY_N  = 2                  # Run FaceMesh (expression/blink) every N frames
DEAD_ZONE_PX = 20                  # Ignore jitter within ±N px of frame center
# -- Sweep (search) settings --
SWEEP_MIN   = 10                   # Leftmost servo angle during sweep
SWEEP_MAX   = 170                  # Rightmost servo angle during sweep
SWEEP_STEP  = 2                    # Degrees per frame during sweep
LOCK_GRACE_FRAMES = 20             # Frames to keep lock after Kelia disappears
# -- Expression detection thresholds --
EAR_THRESH       = 0.22            # Eye Aspect Ratio below this = eye closed
EAR_CONSEC_FRAMES = 2             # Consecutive closed frames needed to count a blink
SMILE_WIDTH_THRESH = 0.45          # Mouth width / face WIDTH ratio  (relaxed ~0.35, big smile ~0.50+)
SMILE_LIFT_THRESH  = 0.02          # Corner lift / face height; both conditions must be true (AND)
# ============================================================

# ---- Load Face Database ----
DB_PATH = Path("data/db/face_db.npz")
if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found at {DB_PATH}. Run enrollment first.")

data = np.load(DB_PATH)
db_embeddings = {k: data[k].astype(np.float32) for k in data.files}

if TARGET_NAME not in db_embeddings:
    raise ValueError(f"'{TARGET_NAME}' not found in DB. Available: {list(db_embeddings.keys())}")

my_embedding = db_embeddings[TARGET_NAME]
print(f"Loaded embedding for '{TARGET_NAME}' (dim={my_embedding.size})")

# ---- Load ArcFace ONNX Model ----
ort_session = ort.InferenceSession(
    "models/embedder_arcface.onnx",
    providers=["CPUExecutionProvider"],
)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
print(f"ONNX model loaded. Input: {input_name}, Output: {output_name}")

# ---- MediaPipe FaceMesh ----
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,          # Disabled iris – not needed, saves ~5ms/frame
    min_detection_confidence=0.5,
    min_tracking_confidence=0.65,    # Higher = fewer re-detects, faster tracking
)

# ---- Haar Cascade ----
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade. Did you download the XML file?")

# ---- Landmark index groups ----
# 5-point alignment
IDX_LEFT_EYE    = 33
IDX_RIGHT_EYE   = 263
IDX_NOSE_TIP    = 1
IDX_MOUTH_LEFT  = 61
IDX_MOUTH_RIGHT = 291

# Eye Aspect Ratio – 6 points per eye (p1..p6)
# EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
L_EYE_PTS = [33, 160, 158, 133, 153, 144]   # left eye
R_EYE_PTS = [263, 387, 385, 362, 380, 373]  # right eye

# Smile geometry
MOUTH_LEFT_C  = 61    # left mouth corner
MOUTH_RIGHT_C = 291   # right mouth corner
UPPER_LIP_MID = 13    # upper inner lip centre
LOWER_LIP_MID = 14    # lower inner lip centre
CHIN_IDX      = 152   # chin tip (for face-height normalisation)

# ---- Canonical 112x112 alignment targets ----
DST_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def run_face_mesh(roi_bgr):
    """Run MediaPipe FaceMesh on a BGR ROI. Returns landmark list or None."""
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    res = mp_face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    return res.multi_face_landmarks[0].landmark


def get_5pt_from_lm(lm, W, H):
    """Extract the 5 alignment keypoints from a landmark list."""
    idxs = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]
    pts  = [[lm[i].x * W, lm[i].y * H] for i in idxs]
    return np.array(pts, dtype=np.float32)


def compute_ear(lm, eye_pts, W, H):
    """Eye Aspect Ratio for one eye. <EAR_THRESH means eye is closed."""
    p = np.array([[lm[i].x * W, lm[i].y * H] for i in eye_pts])
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3]) + 1e-6
    return (A + B) / (2.0 * C)


def compute_smile(lm, W, H):
    """Returns (is_smiling, norm_width, norm_lift) from mouth landmarks.
    Smile = mouth is BOTH wide AND corners are lifted -- requires both (AND).
    """
    lc   = np.array([lm[MOUTH_LEFT_C].x  * W, lm[MOUTH_LEFT_C].y  * H])
    rc   = np.array([lm[MOUTH_RIGHT_C].x * W, lm[MOUTH_RIGHT_C].y * H])
    ul   = np.array([lm[UPPER_LIP_MID].x * W, lm[UPPER_LIP_MID].y * H])
    ll   = np.array([lm[LOWER_LIP_MID].x * W, lm[LOWER_LIP_MID].y * H])
    # Use left/right face edge for face width (cheekbones area)
    left_cheek  = np.array([lm[234].x * W, lm[234].y * H])   # MediaPipe left face edge
    right_cheek = np.array([lm[454].x * W, lm[454].y * H])   # MediaPipe right face edge
    nose = np.array([lm[IDX_NOSE_TIP].x  * W, lm[IDX_NOSE_TIP].y  * H])
    chin = np.array([lm[CHIN_IDX].x      * W, lm[CHIN_IDX].y      * H])

    face_w      = np.linalg.norm(right_cheek - left_cheek) + 1e-6
    face_h      = np.linalg.norm(chin - nose) + 1e-6
    mouth_w     = np.linalg.norm(rc - lc)

    # Corner lift: positive means corners are ABOVE the lip midpoint (smile shape)
    lip_mid_y    = (ul[1] + ll[1]) / 2
    corner_mid_y = (lc[1] + rc[1]) / 2
    lift         = (lip_mid_y - corner_mid_y) / face_h   # normalised by face height
    norm_w       = mouth_w / face_w                       # normalised by face width

    # Require BOTH: mouth wide enough AND corners are raised
    is_smiling = (norm_w > SMILE_WIDTH_THRESH) and (lift > SMILE_LIFT_THRESH)
    return is_smiling, norm_w, lift


def align_face(frame, kps):
    """Warp face to canonical 112x112 using 5 landmarks."""
    M, _ = cv2.estimateAffinePartial2D(kps, DST_POINTS, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)


def get_embedding(aligned_face):
    """Run ArcFace ONNX inference. Returns L2-normalized 512-D vector."""
    img = cv2.resize(aligned_face, (112, 112))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0
    x = rgb[None, ...]  # NHWC (1, 112, 112, 3) -- matches our model
    y = ort_session.run([output_name], {input_name: x})[0]
    emb = y.reshape(-1).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


# ---- Async servo sender ----
# Tracks last result so HUD can show live status.
_servo_lock   = threading.Lock()
_servo_thread = None
_servo_status = "INIT"   # "OK" | "FAIL" | "SKIP" | "INIT"

def send_servo_command(angle):
    """Send HTTP request to ESP8266 on a background daemon thread.
    If the previous request is still in flight we skip this one (avoids queue buildup),
    but the status is shown on the HUD so the user can see what's happening.
    """
    global _servo_thread, _servo_status

    def _send(a):
        global _servo_status
        try:
            url = f"http://{ESP8266_IP}/MOVE?angle={a}"
            r = requests.get(url, timeout=0.5)   # 500 ms – enough for WiFi
            _servo_status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
        except requests.exceptions.ConnectionError:
            _servo_status = "FAIL-CONN"
        except requests.exceptions.Timeout:
            _servo_status = "FAIL-TIMEOUT"
        except Exception as e:
            _servo_status = f"FAIL-{type(e).__name__}"

    with _servo_lock:
        if _servo_thread is not None and _servo_thread.is_alive():
            _servo_status = "SKIP"   # previous request still running, drop this frame
            return
        _servo_thread = threading.Thread(target=_send, args=(angle,), daemon=True)
        _servo_thread.start()


# ---- Startup connectivity check ----
print(f"\nTesting connection to ESP8266 at {ESP8266_IP}...")
try:
    test_r = requests.get(f"http://{ESP8266_IP}/MOVE?angle=90", timeout=3)
    print(f"ESP8266 reachable. Status: {test_r.status_code}")
except Exception as e:
    print(f"WARNING: Cannot reach ESP8266 at {ESP8266_IP}: {e}")
    print("Servo commands will fail. Check IP address and WiFi connection.")


# ---- Main Loop ----
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

# Keep buffer minimal so we always grab the freshest frame
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"\nTracking '{TARGET_NAME}'. Press 'q' to quit.\n")

last_angle    = 90
prev_time     = time.time()
fps           = 0.0
frame_count   = 0

# Persisted between recognition frames
last_is_me    = False
last_sim      = 0.0

# Sweep state (used when searching)
sweep_angle   = 90
sweep_dir     = 1

# Grace counter
grace_counter = 0

# ---- Expression state ----
blink_counter    = 0          # total blinks detected
closed_frames    = 0          # consecutive frames with EAR below threshold
eye_was_open     = True       # for edge-detection (open→closed→open = 1 blink)
is_smiling       = False
expression_label = "Neutral"  # displayed on HUD

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    vis   = frame.copy()
    frame_count += 1

    do_recognize = (frame_count % RECOGNIZE_EVERY_N == 0)
    do_mesh      = (frame_count % MESH_EVERY_N == 0)

    # -- Face detection (fast, every frame) --
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.15, 4, minSize=(60, 60))

    status_text  = f"Searching for {TARGET_NAME}..."
    status_color = (128, 128, 128)

    if len(faces) > 0:
        # Pick largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]
        roi = frame[y:y + h, x:x + w]
        roi_H, roi_W = roi.shape[:2]

        # -- FaceMesh: expressions + alignment (every MESH_EVERY_N frames) --
        if do_mesh:
            lm = run_face_mesh(roi)
            if lm is not None:
                # --- Blink detection (EAR) ---
                left_ear  = compute_ear(lm, L_EYE_PTS,  roi_W, roi_H)
                right_ear = compute_ear(lm, R_EYE_PTS,  roi_W, roi_H)
                avg_ear   = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESH:
                    closed_frames += 1
                    eye_was_open   = False
                else:
                    if not eye_was_open and closed_frames >= EAR_CONSEC_FRAMES:
                        blink_counter += 1   # rising edge: eye just re-opened after being closed
                    closed_frames  = 0
                    eye_was_open   = True

                # --- Expression logic ---
                is_smiling, norm_w, lift = compute_smile(lm, roi_W, roi_H)
                if avg_ear < EAR_THRESH:
                    expression_label = "😴 Eyes Closed"
                elif is_smiling:
                    expression_label = "😊 Smiling!"
                else:
                    expression_label = "👀 Eyes Open"

                # --- Extract 5pt for alignment ---
                kps_5pt = get_5pt_from_lm(lm, roi_W, roi_H)
                kps_5pt[:, 0] += x
                kps_5pt[:, 1] += y

                # -- ArcFace recognition (only on recognition frames) --
                if do_recognize:
                    aligned = align_face(frame, kps_5pt)
                    if aligned is not None:
                        query_emb  = get_embedding(aligned)
                        sim        = float(np.dot(query_emb, my_embedding))
                        last_sim   = sim
                        last_is_me = sim >= SIMILARITY_THRESHOLD
                        if last_is_me:
                            grace_counter = LOCK_GRACE_FRAMES

                # Draw alignment dots
                for (px, py) in kps_5pt.astype(int):
                    cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 255), -1)

        # -- Decide mode based on cached identity + grace --
        if last_is_me or grace_counter > 0:
            # ===== LOCKED / TRACKING MODE =====
            status_text  = f"LOCKED: {TARGET_NAME} ({last_sim:.2f})"
            status_color = (0, 255, 0)

            face_center_x  = x + w // 2
            frame_center_x = FRAME_WIDTH // 2

            # Follow face – dead-zone prevents micro-jitter
            if abs(face_center_x - frame_center_x) > DEAD_ZONE_PX:
                target_angle   = int(np.interp(face_center_x, [0, FRAME_WIDTH], [180, 0]))
                smoothed_angle = int(last_angle + (target_angle - last_angle) * SMOOTHING_FACTOR)
                send_servo_command(smoothed_angle)
                last_angle   = smoothed_angle
                sweep_angle  = smoothed_angle  # keep sweep in sync so resume is smooth

            cv2.putText(vis, f"Servo: {last_angle} deg", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if face_center_x < frame_center_x - DEAD_ZONE_PX:
                direction_text = "<- Face Left"
            elif face_center_x > frame_center_x + DEAD_ZONE_PX:
                direction_text = "Face Right ->"
            else:
                direction_text = "- Centered -"
                
            cv2.putText(vis, direction_text, (FRAME_WIDTH - 240, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Draw a solid lock indicator ring around the box
            cv2.rectangle(vis, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 255, 0), 1)

        else:
            # ===== UNKNOWN FACE – keep sweeping =====
            status_text  = f"Unknown ({last_sim:.2f}) – scanning..."
            status_color = (0, 100, 255)

        # Draw bounding box and center crosshair every frame
        cv2.rectangle(vis, (x, y), (x + w, y + h), status_color, 2)
        cx = x + w // 2
        cv2.line(vis, (cx, y), (cx, y + h), (255, 0, 255), 1)

    else:
        # No face visible – count down grace, then go back to sweep
        if grace_counter > 0:
            grace_counter -= 1
            status_text  = f"Holding lock... ({grace_counter})"
            status_color = (0, 200, 100)
        else:
            last_is_me = False
            last_sim   = 0.0

    # ===== SWEEP LOGIC =====
    # Run whenever we are NOT locked (or holding lock failed)
    is_locked = last_is_me or grace_counter > 0
    if not is_locked:
        # Advance sweep position
        sweep_angle += SWEEP_STEP * sweep_dir
        if sweep_angle >= SWEEP_MAX:
            sweep_angle = SWEEP_MAX
            sweep_dir   = -1
        elif sweep_angle <= SWEEP_MIN:
            sweep_angle = SWEEP_MIN
            sweep_dir   = 1

        send_servo_command(int(sweep_angle))
        last_angle = int(sweep_angle)



    # ---- Rolling FPS (exponential moving average, updated every frame) ----
    now = time.time()
    dt  = now - prev_time
    if dt > 0:
        fps = fps * 0.9 + (1.0 / dt) * 0.1
    prev_time = now

    # ---- HUD ----
    cv2.putText(vis, status_text,       (10, 30),               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 110),              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Expression label – only shown when locked on
    if last_is_me or grace_counter > 0:
        if expression_label == "😴 Eyes Closed":
            expr_color = (255, 150, 0)
        elif expression_label == "😊 Smiling!":
            expr_color = (0, 255, 128)
        elif expression_label == "👀 Eyes Open":
            expr_color = (0, 200, 255)
        else:
            expr_color = (200, 200, 200)
        cv2.putText(vis, expression_label,           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, expr_color, 2)

    # Blink counter – always visible once a face is seen
    cv2.putText(vis, f"Blinks: {blink_counter}",    (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    cv2.putText(vis, "q=quit",                       (10, FRAME_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Face Tracker", vis)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()