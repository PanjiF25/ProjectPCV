import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import math
import time
from collections import deque
import threading

# --- KONFIGURASI UTAMA ---
OSC_IP = "127.0.0.1"  # IP VSeeFace kamu
OSC_PORT = 5555
WEBCAM_ID = 0
TARGET_FPS = 30  # Sweet spot untuk stabilitas
PROCESS_EVERY_N_FRAMES = 1  # Process setiap frame
MIN_DETECTION_CONFIDENCE = 0.5  # Balanced
USE_GPU = True  # GPU acceleration

# ==========================================
# === DATA KALIBRASI FINAL ===
# ==========================================

# 1. LENGAN (1, 1, 1)
ARM_INVERT_X = 1.0  
ARM_INVERT_Y = 1.0 
ARM_INVERT_Z = 1.0 
ARM_GAIN_XY  = 0.95  # Reduced sedikit untuk gerakan lebih smooth
ARM_GAIN_Z   = 0.55  # Reduced untuk depth lebih stabil
ARM_SMOOTHING = 0.7  # Exponential smoothing factor untuk arm 

# 2. JARI (Finger: L=Z, R=Z | Sign: L=1, R=-1)
# Axis Index: 0=X, 1=Y, 2=Z
FINGER_AXIS_L = 2 
FINGER_AXIS_R = 2
FINGER_SIGN_L = 1.0
FINGER_SIGN_R = -1.0
FINGER_SENSITIVITY = 1.1  # Reduced untuk less twitchy
FINGER_DEADZONE = 0.08  # Deadzone untuk finger curl detection

# 3. JEMPOL (Thumb: L=Y, R=Y | Sign: L=-1, R=-1)
THUMB_AXIS_L = 1
THUMB_AXIS_R = 1
THUMB_SIGN_L = 1.0  # Inverted untuk left
THUMB_SIGN_R = -1.0  # Inverted untuk right
THUMB_SENSITIVITY = 1.15  # Reduced untuk responsivitas lebih stabil
THUMB_DEADZONE = 0.1  # Deadzone lebih besar untuk thumb

# ==========================================

# --- TUNING LAINNYA ---
EYE_Y_OFFSET = 0.0  # Natural position
GAZE_SENSITIVITY = 1.1   # Reduced untuk gerakan mata lebih halus
PITCH_CORRECTION_FACTOR = 0.01
DEADZONE = 0.55     # Ditingkatkan untuk mengurangi jitter
ADAPTIVE_DEADZONE_MULTIPLIER = 1.5  # Deadzone lebih besar saat gerakan cepat
NECK_RATIO = 0.5
EAR_THRESH_CLOSE, EAR_THRESH_OPEN = 0.12, 0.20  # Untuk mata sipit & kacamata
MOUTH_OPEN_MIN, MOUTH_OPEN_MAX = 5.0, 40.0
SMOOTHING_WINDOW = 4  # Ditingkatkan untuk gerakan lebih smooth
OSC_RATE_LIMIT = 0.016  # ~60Hz max untuk mengurangi bandwidth  

# --- HELPER FUNCTIONS ---
def euler_to_quaternion(pitch, yaw, roll):
    qx = np.sin(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) - np.cos(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    qy = np.cos(pitch/2) * np.sin(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.cos(yaw/2) * np.sin(roll/2)
    qz = np.cos(pitch/2) * np.cos(yaw/2) * np.sin(roll/2) - np.sin(pitch/2) * np.sin(yaw/2) * np.cos(roll/2)
    qw = np.cos(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    return [qx, qy, qz, qw]

# Fungsi Rotasi Jari Custom (Bisa pilih Axis X/Y/Z)
def get_finger_quat(angle, axis_idx):
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    if axis_idx == 0:   return [s, 0, 0, c] # X
    elif axis_idx == 1: return [0, s, 0, c] # Y
    elif axis_idx == 2: return [0, 0, s, c] # Z
    return [0, 0, 0, 1]

def get_limb_rotation(start, end, rest_vector):
    v_curr = np.array(end) - np.array(start)
    norm = np.linalg.norm(v_curr)
    if norm < 1e-6: return [0,0,0,1]
    v_curr = v_curr / norm
    v_rest = np.array(rest_vector)
    v_rest = v_rest / np.linalg.norm(v_rest)
    dot = np.dot(v_rest, v_curr)
    dot = max(-1.0, min(1.0, dot)) 
    angle = math.acos(dot)
    axis = np.cross(v_rest, v_curr)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6: return [0, 0, 0, 1]
    axis = axis / axis_len
    sin_half = math.sin(angle / 2)
    qx = axis[0] * sin_half
    qy = axis[1] * sin_half
    qz = axis[2] * sin_half
    qw = math.cos(angle / 2)
    return [qx, qy, qz, qw]

def calculate_ear(face_landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h  = np.linalg.norm(coords[0] - coords[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def get_relative_iris(face_landmarks, iris_idx, inner_idx, outer_idx, img_w, img_h):
    iris = np.array([face_landmarks.landmark[iris_idx].x * img_w, face_landmarks.landmark[iris_idx].y * img_h])
    inner = np.array([face_landmarks.landmark[inner_idx].x * img_w, face_landmarks.landmark[inner_idx].y * img_h])
    outer = np.array([face_landmarks.landmark[outer_idx].x * img_w, face_landmarks.landmark[outer_idx].y * img_h])
    eye_width = np.linalg.norm(outer - inner)
    eye_vec = outer - inner
    eye_vec_norm = eye_vec / (np.linalg.norm(eye_vec) + 1e-6)
    iris_vec = iris - inner
    proj_x = np.dot(iris_vec, eye_vec_norm)
    norm_x = (proj_x / eye_width) * 2.0 - 1.0
    cross_prod = (eye_vec[0] * (iris[1] - inner[1])) - (eye_vec[1] * (iris[0] - inner[0]))
    dist_y = cross_prod / eye_width
    norm_y = dist_y / (eye_width * 0.3)
    return norm_x, norm_y

def map_range(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def send_osc_throttled(address, value, threshold=0.001):
    """Send OSC message dengan throttling untuk reduce bandwidth"""
    global last_sent_values
    key = address + str(value[0] if isinstance(value, list) else value)
    
    # Check jika nilai berubah signifikan
    if key in last_sent_values:
        if isinstance(value, list) and len(value) > 4:  # Quaternion
            old_val = last_sent_values[key]
            if all(abs(v - o) < threshold for v, o in zip(value[1:], old_val[1:])):
                return  # Skip jika perubahan terlalu kecil
    
    client.send_message(address, value)
    last_sent_values[key] = value

def smooth_quaternion(q_new, q_prev, alpha=0.7):
    """Exponential smoothing for quaternions"""
    return [alpha * n + (1 - alpha) * p for n, p in zip(q_new, q_prev)]

def apply_deadzone(value, prev_value, deadzone):
    """Apply deadzone to reduce jitter"""
    if abs(value - prev_value) < deadzone:
        return prev_value
    return value

def get_finger_curl(landmarks, tip_idx, knuckle_idx, wrist_idx, is_thumb=False):
    tip = np.array([landmarks.landmark[tip_idx].x, landmarks.landmark[tip_idx].y, landmarks.landmark[tip_idx].z])
    wrist = np.array([landmarks.landmark[wrist_idx].x, landmarks.landmark[wrist_idx].y, landmarks.landmark[wrist_idx].z])
    dist_tip_wrist = np.linalg.norm(tip - wrist)
    knuckle = np.array([landmarks.landmark[knuckle_idx].x, landmarks.landmark[knuckle_idx].y, landmarks.landmark[knuckle_idx].z])
    dist_palm = np.linalg.norm(knuckle - wrist)
    ratio = dist_tip_wrist / (dist_palm + 1e-6)
    
    # Threshold yang lebih baik untuk deteksi curl dengan range lebih stabil
    if is_thumb:
        # Range disesuaikan untuk tracking jempol yang akurat dan smooth
        curl = np.clip((ratio - 1.6) / (0.8 - 1.6), 0.0, 1.0)  # Range dipersempit
        sensitivity = (THUMB_SENSITIVITY if 'THUMB_SENSITIVITY' in globals() else FINGER_SENSITIVITY) * 0.95
    else:
        # Range lebih stabil dengan clipping langsung
        curl = np.clip((ratio - 1.85) / (0.95 - 1.85), 0.0, 1.0)
        sensitivity = FINGER_SENSITIVITY * 0.95  # Sedikit reduced untuk less jittery
    
    return curl * sensitivity

# --- CLASS STABILIZER ---
class Stabilizer:
    def __init__(self, state_num=2, measure_num=1, cov_process=0.0001, cov_measure=0.1):
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)
        self.state = np.zeros((state_num, 1), dtype=np.float32)
        self.filter.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.filter.measurementMatrix = np.array([[1, 1]], np.float32)
        self.filter.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * cov_process
        self.filter.measurementNoiseCov = np.array([[1]], np.float32) * cov_measure
    def update(self, measurement):
        self.filter.predict()
        self.filter.correct(np.array([[np.float32(measurement)]]))
        self.state = self.filter.statePost
        return self.state[0][0]

# Moving Average Filter untuk smoothing tambahan
class MovingAverageFilter:
    def __init__(self, window_size=4):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def update(self, value):
        self.values.append(value)
        return sum(self.values) / len(self.values)

# --- INIT ---
print("=" * 60)
print(" 🎭 VTuber ULTIMATE Tracking - FULL BODY + FINGERS")
print(" ⚡ Optimized for RTX 4050")
print("=" * 60)
print(f"Target: {OSC_IP}:{OSC_PORT}")
print(f"Resolution: 640x480 @ {TARGET_FPS}fps")
print(f"Model: Balanced (Complexity 1)")
print("Starting MediaPipe...")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Balanced model - optimal untuk RTX 4050
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,  # Balanced
    min_tracking_confidence=0.5,   # Balanced
    refine_face_landmarks=True,    # Iris tracking untuk eye gaze
    model_complexity=1             # Model 1 sweet spot
)
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# Stabilizers - Dioptimalkan untuk smoothness maksimal dengan parameter lebih agresif
stab_pitch = Stabilizer(cov_process=0.005, cov_measure=0.25)  # Lebih smooth
stab_yaw   = Stabilizer(cov_process=0.005, cov_measure=0.25)
stab_roll  = Stabilizer(cov_process=0.005, cov_measure=0.25)
stab_eye_x = Stabilizer(cov_process=0.0005, cov_measure=0.3)  # Sangat halus untuk eye tracking
stab_eye_y = Stabilizer(cov_process=0.0005, cov_measure=0.3)
stab_spine_roll = Stabilizer(cov_process=0.01, cov_measure=0.2)
stab_spine_yaw  = Stabilizer(cov_process=0.01, cov_measure=0.2)

# Stabilizer Jari (10 Jari) - Smoothing ditingkatkan untuk gerakan natural
stab_fingers_L = [Stabilizer(cov_process=0.02, cov_measure=0.18) for _ in range(5)]
stab_fingers_R = [Stabilizer(cov_process=0.02, cov_measure=0.18) for _ in range(5)]

# Moving Average Filters untuk layer kedua smoothing
ma_pitch = MovingAverageFilter(SMOOTHING_WINDOW)
ma_yaw = MovingAverageFilter(SMOOTHING_WINDOW)
ma_roll = MovingAverageFilter(SMOOTHING_WINDOW)
ma_eye_x = MovingAverageFilter(SMOOTHING_WINDOW)
ma_eye_y = MovingAverageFilter(SMOOTHING_WINDOW)

model_points = np.array([
    (0.0, 0.0, 0.0), 
    (0.0, -330.0, -65.0), 
    (-225.0, 170.0, -135.0), 
    (225.0, 170.0, -135.0), 
    (-150.0, -150.0, -125.0), 
    (150.0, -150.0, -125.0)
], dtype=np.float64)

LEFT_EYE_IDXS, RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
L_IRIS_C, L_IN, L_OUT = 468, 133, 33  
R_IRIS_C, R_IN, R_OUT = 473, 362, 263 
last_raw_pitch, last_raw_yaw, last_raw_roll = 0, 0, 0
blink_l_state, blink_r_state = 0.0, 0.0
prev_time = 0
frame_count = 0  # Frame counter
last_osc_time = 0  # OSC rate limiting
last_sent_values = {}  # Cache untuk menghindari duplicate messages

# Config Jari
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGER_INDICES = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)] 
BONE_SUFFIXES = ["Proximal", "Intermediate", "Distal"]

# Cache untuk arm smoothing (exponential smoothing)
prev_arm_rotations = {
    "LeftUpperArm": [0, 0, 0, 1],
    "LeftLowerArm": [0, 0, 0, 1],
    "RightUpperArm": [0, 0, 0, 1],
    "RightLowerArm": [0, 0, 0, 1]
}

# Cache untuk finger curl previous values
prev_finger_curls_L = [0.0] * 5
prev_finger_curls_R = [0.0] * 5

# --- CAMERA --- Balanced settings
cap = cv2.VideoCapture(WEBCAM_ID, cv2.CAP_DSHOW)  # DirectShow untuk Windows
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Sweet spot
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 480p optimal balance
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce latency
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG codec 

if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()

print("✅ Camera opened!")
print("📹 Tracking started... Press 'Q' to quit\n")
print("=" * 60)

while cap.isOpened():
    try:
        success, image = cap.read()
        if not success: 
            print("Failed to read frame, retrying...")
            continue
        
        frame_count += 1

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        image = cv2.flip(image, 1)
        img_h, img_w, _ = image.shape
        
        # Process dengan error handling
        image.flags.writeable = False
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True
    except Exception as e:
        print(f"Error in processing: {e}")
        continue

    # === 1. FACE TRACKING ===
    if results.face_landmarks:
        # Custom drawing style untuk face - lebih unik dengan warna cyan-purple
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(255,100,200), thickness=1, circle_radius=0),
            mp_drawing.DrawingSpec(color=(100,255,255), thickness=2, circle_radius=2)
        )
        fl = results.face_landmarks
        
        # Head pose estimation
        image_points = np.array([
            (fl.landmark[1].x * img_w, fl.landmark[1].y * img_h),
            (fl.landmark[152].x * img_w, fl.landmark[152].y * img_h),
            (fl.landmark[263].x * img_w, fl.landmark[263].y * img_h),
            (fl.landmark[33].x * img_w, fl.landmark[33].y * img_h),
            (fl.landmark[287].x * img_w, fl.landmark[287].y * img_h),
            (fl.landmark[57].x * img_w, fl.landmark[57].y * img_h)
        ], dtype=np.float64)
        
        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_w/2], [0, focal_length, img_h/2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        success_pnp, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        dY = (fl.landmark[263].y * img_h) - (fl.landmark[33].y * img_h)
        dX = (fl.landmark[263].x * img_w) - (fl.landmark[33].x * img_w)
        curr_pitch, curr_yaw, curr_roll = angles[0], angles[1], math.degrees(math.atan2(dY, dX))

        # Adaptive Deadzone - lebih besar saat perubahan cepat untuk reduce jitter
        velocity_pitch = abs(curr_pitch - last_raw_pitch)
        velocity_yaw = abs(curr_yaw - last_raw_yaw)
        velocity_roll = abs(curr_roll - last_raw_roll)
        
        adaptive_deadzone_p = DEADZONE * (1.0 + (velocity_pitch / 10.0) * ADAPTIVE_DEADZONE_MULTIPLIER)
        adaptive_deadzone_y = DEADZONE * (1.0 + (velocity_yaw / 10.0) * ADAPTIVE_DEADZONE_MULTIPLIER)
        adaptive_deadzone_r = DEADZONE * (1.0 + (velocity_roll / 10.0) * ADAPTIVE_DEADZONE_MULTIPLIER)
        
        if abs(curr_pitch - last_raw_pitch) < adaptive_deadzone_p: curr_pitch = last_raw_pitch
        else: last_raw_pitch = curr_pitch
        if abs(curr_yaw - last_raw_yaw) < adaptive_deadzone_y: curr_yaw = last_raw_yaw
        else: last_raw_yaw = curr_yaw
        if abs(curr_roll - last_raw_roll) < adaptive_deadzone_r: curr_roll = last_raw_roll
        else: last_raw_roll = curr_roll

        # Double-stage filtering: Kalman + Moving Average
        k_pitch = stab_pitch.update(curr_pitch)
        k_yaw = stab_yaw.update(curr_yaw)
        k_roll = stab_roll.update(curr_roll)
        
        s_pitch = ma_pitch.update(k_pitch)
        s_yaw = ma_yaw.update(k_yaw)
        s_roll = ma_roll.update(k_roll)

        # Neck + Head split
        neck_pitch, neck_yaw, neck_roll = s_pitch * NECK_RATIO, s_yaw * NECK_RATIO, s_roll * NECK_RATIO
        head_pitch, head_yaw, head_roll = s_pitch - neck_pitch, s_yaw - neck_yaw, s_roll - neck_roll
        
        # Eye blink (EAR)
        raw_ear_l = calculate_ear(fl, LEFT_EYE_IDXS, img_w, img_h)
        raw_ear_r = calculate_ear(fl, RIGHT_EYE_IDXS, img_w, img_h)
        if raw_ear_l < EAR_THRESH_CLOSE: blink_l_state = 1.0
        elif raw_ear_l > EAR_THRESH_OPEN: blink_l_state = 0.0
        if raw_ear_r < EAR_THRESH_CLOSE: blink_r_state = 1.0
        elif raw_ear_r > EAR_THRESH_OPEN: blink_r_state = 0.0
        
        # Fix blink saat kepala miring
        if s_yaw > 20.0: blink_r_state = blink_l_state 
        elif s_yaw < -20.0: blink_l_state = blink_r_state

        # Eye gaze (iris tracking) - dengan outlier rejection dan clamping
        lx, ly = get_relative_iris(fl, L_IRIS_C, L_IN, L_OUT, img_w, img_h)
        rx, ry = get_relative_iris(fl, R_IRIS_C, R_IN, R_OUT, img_w, img_h)
        
        # Outlier rejection - jika perbedaan kedua mata terlalu besar, ada error
        eye_diff = abs(lx - rx)
        if eye_diff < 0.4:  # Threshold lebih ketat untuk detect outlier
            avg_x = np.clip((lx + rx)/2.0, -0.7, 0.7)  # Range lebih terbatas
            avg_y = np.clip(((ly + ry)/2.0) - (s_pitch * PITCH_CORRECTION_FACTOR) + EYE_Y_OFFSET, -0.5, 0.5)
        else:
            # Gunakan nilai sebelumnya jika outlier detected
            avg_x, avg_y = stab_eye_x.state[0][0], stab_eye_y.state[0][0]
        
        if not (blink_l_state > 0.5 or blink_r_state > 0.5):
            # Double-stage filtering untuk mata
            k_eye_x = stab_eye_x.update(avg_x)
            k_eye_y = stab_eye_y.update(avg_y)
            smooth_eye_x = ma_eye_x.update(k_eye_x)
            smooth_eye_y = ma_eye_y.update(k_eye_y)
        else:
            smooth_eye_x, smooth_eye_y = stab_eye_x.state[0][0], stab_eye_y.state[0][0]
        
        # Mouth open
        mouth_dist = np.linalg.norm(
            np.array([fl.landmark[13].x*img_w, fl.landmark[13].y*img_h]) - 
            np.array([fl.landmark[14].x*img_w, fl.landmark[14].y*img_h])
        )
        mouth_open = max(0.0, min(1.0, (mouth_dist - 5.0) / 35.0))

        # Send to VSeeFace
        nqx, nqy, nqz, nqw = euler_to_quaternion(math.radians(neck_pitch), math.radians(neck_yaw), math.radians(neck_roll))
        client.send_message("/VMC/Ext/Bone/Pos", ["Neck", 0.0, 0.0, 0.0, float(nqx), float(nqy), float(nqz), float(nqw)])
        
        hqx, hqy, hqz, hqw = euler_to_quaternion(math.radians(head_pitch), math.radians(head_yaw), math.radians(head_roll))
        client.send_message("/VMC/Ext/Bone/Pos", ["Head", 0.0, 0.0, 0.0, float(hqx), float(hqy), float(hqz), float(hqw)])
        
        client.send_message("/VMC/Ext/Root/Pos", ["Root", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_L", float(blink_l_state)])
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_R", float(blink_r_state)])
        client.send_message("/VMC/Ext/Blend/Val", ["A", float(mouth_open)])
        
        # Eye rotation dengan multiplier yang lebih kecil untuk gerakan natural
        eye_mult_x = smooth_eye_x * GAZE_SENSITIVITY * 20.0  # Further reduced untuk lebih smooth
        eye_mult_y = smooth_eye_y * GAZE_SENSITIVITY * 16.0  # Further reduced untuk lebih smooth
        eqx, eqy, eqz, eqw = euler_to_quaternion(math.radians(eye_mult_y), math.radians(eye_mult_x), 0)
        client.send_message("/VMC/Ext/Bone/Pos", ["LeftEye", 0.0, 0.0, 0.0, float(eqx), float(eqy), float(eqz), float(eqw)])
        client.send_message("/VMC/Ext/Bone/Pos", ["RightEye", 0.0, 0.0, 0.0, float(eqx), float(eqy), float(eqz), float(eqw)])

    # === 2. BODY & ARM TRACKING ===
    if results.pose_landmarks:
        # Custom style untuk body tracking - warna orange-yellow gradient
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,165,255), thickness=3, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=3)
        )
        lm = results.pose_landmarks.landmark

        def get_vec(idx): 
            return [lm[idx].x, lm[idx].y, lm[idx].z]
        
        def to_unity_vec(mp_vec): 
            return np.array([
                mp_vec[0] * ARM_INVERT_X * ARM_GAIN_XY, 
                mp_vec[1] * ARM_INVERT_Y * ARM_GAIN_XY, 
                mp_vec[2] * ARM_INVERT_Z * ARM_GAIN_Z
            ])

        # Spine rotation (shoulder tilt)
        l_sh, r_sh = get_vec(11), get_vec(12)
        spine_roll = stab_spine_roll.update((l_sh[1] - r_sh[1]) * -120.0) 
        spine_yaw = stab_spine_yaw.update((l_sh[2] - r_sh[2]) * -80.0)
        sqx, sqy, sqz, sqw = euler_to_quaternion(0, math.radians(spine_yaw), math.radians(spine_roll))
        client.send_message("/VMC/Ext/Bone/Pos", ["Spine", 0.0, 0.0, 0.0, float(sqx), float(sqy), float(sqz), float(sqw)])

        # Left arm - dengan exponential smoothing
        if lm[11].visibility > 0.5 and lm[13].visibility > 0.5:
            start, end = to_unity_vec(get_vec(11)), to_unity_vec(get_vec(13))
            q_lu_raw = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
            q_lu = smooth_quaternion(q_lu_raw, prev_arm_rotations["LeftUpperArm"], ARM_SMOOTHING)
            prev_arm_rotations["LeftUpperArm"] = q_lu
            client.send_message("/VMC/Ext/Bone/Pos", ["LeftUpperArm", 0.0, 0.0, 0.0, float(q_lu[0]), float(q_lu[1]), float(q_lu[2]), float(q_lu[3])])
            
            if lm[15].visibility > 0.5:
                start, end = to_unity_vec(get_vec(13)), to_unity_vec(get_vec(15))
                q_ll_raw = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
                q_ll = smooth_quaternion(q_ll_raw, prev_arm_rotations["LeftLowerArm"], ARM_SMOOTHING)
                prev_arm_rotations["LeftLowerArm"] = q_ll
                client.send_message("/VMC/Ext/Bone/Pos", ["LeftLowerArm", 0.0, 0.0, 0.0, float(q_ll[0]), float(q_ll[1]), float(q_ll[2]), float(q_ll[3])])

        # Right arm - dengan exponential smoothing
        if lm[12].visibility > 0.5 and lm[14].visibility > 0.5:
            start, end = to_unity_vec(get_vec(12)), to_unity_vec(get_vec(14))
            q_ru_raw = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
            q_ru = smooth_quaternion(q_ru_raw, prev_arm_rotations["RightUpperArm"], ARM_SMOOTHING)
            prev_arm_rotations["RightUpperArm"] = q_ru
            client.send_message("/VMC/Ext/Bone/Pos", ["RightUpperArm", 0.0, 0.0, 0.0, float(q_ru[0]), float(q_ru[1]), float(q_ru[2]), float(q_ru[3])])
            
            if lm[16].visibility > 0.5:
                start, end = to_unity_vec(get_vec(14)), to_unity_vec(get_vec(16))
                q_rl_raw = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
                q_rl = smooth_quaternion(q_rl_raw, prev_arm_rotations["RightLowerArm"], ARM_SMOOTHING)
                prev_arm_rotations["RightLowerArm"] = q_rl
                client.send_message("/VMC/Ext/Bone/Pos", ["RightLowerArm", 0.0, 0.0, 0.0, float(q_rl[0]), float(q_rl[1]), float(q_rl[2]), float(q_rl[3])])

    # === 3. FINGER TRACKING ===
    
    # LEFT HAND (Finger=Z, Thumb=Y) - dengan safety checks
    if results.left_hand_landmarks:
        try:
            # Custom style untuk left hand - warna purple-blue
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,100,150), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,150,100), thickness=2)
            )
            for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
                is_thumb = (name == "Thumb")
                raw_curl = get_finger_curl(results.left_hand_landmarks, tip, knuckle, 0, is_thumb)
                
                # Apply deadzone untuk reduce jitter
                deadzone = THUMB_DEADZONE if is_thumb else FINGER_DEADZONE
                raw_curl = apply_deadzone(raw_curl, prev_finger_curls_L[i], deadzone)
                prev_finger_curls_L[i] = raw_curl
                
                curl = stab_fingers_L[i].update(raw_curl)
                
                # Gunakan Axis & Sign dari Kalibrasi
                if is_thumb:
                    # Angle disesuaikan agar tidak over-curl
                    angle = curl * (math.pi / 1.8) * THUMB_SIGN_L
                    axis = THUMB_AXIS_L
                else:
                    angle = curl * (math.pi / 1.5) * FINGER_SIGN_L
                    axis = FINGER_AXIS_L
                
                fqx, fqy, fqz, fqw = get_finger_quat(angle, axis)
                
                for suffix in BONE_SUFFIXES: 
                    client.send_message("/VMC/Ext/Bone/Pos", 
                        [f"Left{name}{suffix}", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])
        except Exception as e:
            pass  # Skip jika ada error

    # RIGHT HAND (Finger=Z, Thumb=Y) - dengan safety checks
    if results.right_hand_landmarks:
        try:
            # Custom style untuk right hand - warna green-cyan
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(100,255,150), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(150,255,200), thickness=2)
            )
            for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
                is_thumb = (name == "Thumb")
                raw_curl = get_finger_curl(results.right_hand_landmarks, tip, knuckle, 0, is_thumb)
                
                # Apply deadzone untuk reduce jitter
                deadzone = THUMB_DEADZONE if is_thumb else FINGER_DEADZONE
                raw_curl = apply_deadzone(raw_curl, prev_finger_curls_R[i], deadzone)
                prev_finger_curls_R[i] = raw_curl
                
                curl = stab_fingers_R[i].update(raw_curl)
                
                if is_thumb:
                    # Angle disesuaikan agar tidak over-curl
                    angle = curl * (math.pi / 1.8) * THUMB_SIGN_R
                    axis = THUMB_AXIS_R
                else:
                    angle = curl * (math.pi / 1.5) * FINGER_SIGN_R
                    axis = FINGER_AXIS_R
                
                fqx, fqy, fqz, fqw = get_finger_quat(angle, axis)
                
                for suffix in BONE_SUFFIXES: 
                    client.send_message("/VMC/Ext/Bone/Pos", 
                        [f"Right{name}{suffix}", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])
        except Exception as e:
            pass  # Skip jika ada error

    # === DISPLAY === dengan UI overlay custom dan info debug
    try:
        # Background semi-transparent untuk info panel
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (250, 95), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # FPS counter dengan color indicator
        fps_color = (100, 255, 100) if fps >= 25 else (100, 200, 255)
        cv2.putText(image, f"FPS: {int(fps)}", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, fps_color, 2)
        
        # Detection status indicators
        face_status = "FACE: ON" if results.face_landmarks else "FACE: OFF"
        hands_l = "L-HAND: ON" if results.left_hand_landmarks else "L-HAND: OFF"
        hands_r = "R-HAND: ON" if results.right_hand_landmarks else "R-HAND: OFF"
        
        cv2.putText(image, face_status, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150) if results.face_landmarks else (100, 100, 100), 1)
        cv2.putText(image, hands_l, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 200) if results.left_hand_landmarks else (100, 100, 100), 1)
        cv2.putText(image, hands_r, (130, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 150) if results.right_hand_landmarks else (100, 100, 100), 1)
        
        # Status indicator
        status_text = "TRACKING ACTIVE"
        cv2.putText(image, status_text, (img_w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)
    except Exception as e:
        pass  # Skip UI jika error
    
    cv2.imshow('VTuber ULTIMATE', image)
    
    # FPS limiting untuk prevent overheating dan maintain consistency
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):  # Toggle GPU info
        print(f"GPU Mode: {USE_GPU} | FPS: {int(fps)} | Resolution: {img_w}x{img_h}")

# Cleanup dengan error handling
try:
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("\n✅ Tracking stopped!")
except Exception as e:
    print(f"\n⚠️ Cleanup warning: {e}")
finally:
    print("Program terminated safely.")