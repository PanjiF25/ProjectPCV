# 🎭 VTuber Motion Capture - VSeeFace Integration

Sistem motion capture full body untuk VTuber menggunakan **MediaPipe** dan **VSeeFace**. Tracking lengkap meliputi wajah, mata, badan, tangan, dan 10 jari dengan protokol VMC (Virtual Motion Capture).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## ✨ Fitur Utama

### 🎭 Face & Expression Tracking
- **Head Pose Estimation** - Rotasi kepala 3D (pitch, yaw, roll) dengan pemisahan Neck & Head
- **Eye Tracking** - Pelacakan iris untuk eye gaze direction
- **Blink Detection** - Deteksi kedipan mata otomatis (EAR algorithm)
- **Mouth Tracking** - Deteksi mulut terbuka untuk lip sync
- **Optimized for Glasses** - Disesuaikan untuk pengguna kacamata dan mata sipit

### 💪 Body & Arm Tracking
- **Spine Rotation** - Tracking gerakan bahu dan torso
- **Full Arm Chain** - Upper arm dan lower arm dengan inverse kinematics
- **Visibility Filtering** - Hanya track landmark yang terdeteksi jelas

### 🖐️ Hand & Finger Tracking
- **10 Finger Control** - Tracking semua jari (Thumb, Index, Middle, Ring, Little)
- **Curl Detection** - Deteksi tekukan jari dengan distance-based algorithm
- **Per-Bone Control** - Setiap jari memiliki 3 bones (Proximal, Intermediate, Distal)
- **Custom Axis Rotation** - Konfigurasi axis (X/Y/Z) per jari

### 🎯 Stabilization & Performance
- **Kalman Filtering** - Smoothing untuk gerakan stabil tanpa jitter
- **Adaptive Deadzone** - Mengurangi micro-movements yang tidak perlu
- **Error Handling** - Anti-crash protection untuk stabilitas maksimal
- **Optimized for Gaming Laptops** - Tested di RTX 4050 dan GPU modern

---

### Software Dependencies
- **Python**: 3.8 atau lebih baru
- **VSeeFace**: v1.13.38c4 (included dalam repo)
- **OS**: Windows 10/11

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/PanjiF25/ProjectVtuberPCV.git
cd ProjectVtuberPCV
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python` - Camera capture & display
- `mediapipe` - AI tracking (face, hands, pose)
- `numpy` - Mathematical operations
- `python-osc` - Komunikasi dengan VSeeFace via VMC protocol

### 3. Setup VSeeFace

#### a. Extract & Run VSeeFace
1. Extract file `VSeeFace-v1.13.38c4.zip`
2. Jalankan `VSeeFace.exe`
3. Load VRM model kamu (file `.vrm` included)

#### b. Enable VMC Protocol
1. Di VSeeFace, klik **Settings** (ikon gear)
2. Cari bagian **"VMC Protocol"**
3. Enable **"VMC protocol receiver"**
4. Set **Port**: `39539`
5. Set **IP**: `127.0.0.1` (localhost)

#### c. Enable Body Tracking
1. Di VSeeFace Settings → **Tracking**
2. Enable:
   - ✅ Body tracking / Hip tracking
   - ✅ Additional bone tracking
   - ✅ External tracking / VMC receiver

---

## 🎮 How to Use

### Step 1: Start VSeeFace
Pastikan VSeeFace sudah running dan VMC receiver aktif (lihat Setup di atas).

### Step 2: Run Tracking Program
```bash
python main.py
```

Program akan otomatis:
- Open webcam
- Mulai tracking wajah, badan, dan tangan
- Kirim data ke VSeeFace via OSC/VMC protocol

### Step 3: Posisi Optimal
Untuk hasil tracking terbaik:
- ✅ Jarak **1-1.5 meter** dari kamera
- ✅ Pencahayaan cukup terang
- ✅ Background kontras dengan tubuh
- ✅ Wajah dan tangan dalam frame
- ✅ Hindari backlight yang terlalu kuat

### Keyboard Controls
| Key | Function |
|-----|----------|
| **Q** | Quit/Exit program |
| **G** | Show GPU & performance info |

---

## ⚙️ Configuration & Tuning

Jika tracking kurang pas, edit parameter di `main.py`:

### Eye & Face Tracking
```python
GAZE_SENSITIVITY = 1.3      # Sensitivitas gerakan mata (0.5-2.0)
DEADZONE = 0.45            # Deadzone rotasi kepala (0.2-0.8)
EAR_THRESH_CLOSE = 0.12    # Threshold mata tertutup (0.10-0.15)
EAR_THRESH_OPEN = 0.20     # Threshold mata terbuka (0.18-0.25)
```

### Finger Tracking
```python
FINGER_SENSITIVITY = 1.2    # Sensitivitas 4 jari (0.8-1.5)
THUMB_SENSITIVITY = 1.25    # Sensitivitas jempol (0.8-1.5)
```

### Performance Settings
```python
TARGET_FPS = 30                    # Target framerate
CAMERA_WIDTH = 640                 # Resolution width
CAMERA_HEIGHT = 480                # Resolution height
model_complexity = 1               # MediaPipe quality (0-2)
```

---

## 📁 Project Structure

```
ProjectVtuberPCV/
├── main.py                        # Main tracking script
├── README.md                      # Documentation
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── VSeeFace-v1.13.38c4.zip       # VSeeFace application
└── 6606265595692162344.vrm       # Sample VRM model
```

---

## 🔧 Technical Details

### VMC Protocol Settings
```python
OSC_IP = "127.0.0.1"       # VSeeFace IP (localhost)
OSC_PORT = 39539           # VMC protocol port
WEBCAM_ID = 0              # Camera index
```

### Tracking Components

#### 1. Face Tracking (468 landmarks)
- **Head Pose**: solvePnP untuk estimasi rotasi 3D
- **Eye Gaze**: Iris tracking dengan relative position
- **Blink Detection**: Eye Aspect Ratio (EAR) algorithm
- **Mouth Open**: Lip distance calculation

#### 2. Body Tracking (33 pose landmarks)
- **Spine**: Shoulder tilt detection
- **Arms**: Full IK chain dengan visibility filtering
- **Inverse Kinematics**: Limb rotation dari joint positions

#### 3. Finger Tracking (21 hand landmarks per hand)
- **Curl Detection**: Distance ratio tip-to-wrist vs base-to-wrist
- **Bone Control**: Proximal, Intermediate, Distal per finger
- **Axis Rotation**: Custom X/Y/Z rotation per finger type

### Stabilization Techniques
- **Kalman Filtering**: Mengurangi jitter pada semua tracking
- **Temporal Smoothing**: Buffer untuk consistency
- **Outlier Rejection**: Mendeteksi dan mengabaikan data error
- **Adaptive Deadzone**: Threshold dinamis untuk micro-movements

### Performance Optimization
```python
# Low-end PC (integrated graphics)
model_complexity = 0
CAMERA_HEIGHT = 360

# Mid-range (GTX series)
model_complexity = 1
CAMERA_HEIGHT = 480

# High-end (RTX 4050+)
model_complexity = 1-2
CAMERA_HEIGHT = 720
```

---

## 🎨 Visual Customization

Program menampilkan tracking overlay dengan warna custom:

| Component | Color | Style |
|-----------|-------|-------|
| **Face Mesh** | Cyan-Purple gradient | Landmarks + contours |
| **Body Pose** | Orange-Yellow | Skeleton connections |
| **Left Hand** | Purple-Pink | Hand landmarks |
| **Right Hand** | Green-Cyan | Hand landmarks |
| **UI Overlay** | Cyan text | FPS + Status indicators |

### Ubah Warna Tracking
Edit di `main.py` bagian drawing:
```python
# Face - warna custom
mp_drawing.DrawingSpec(color=(B, G, R), thickness=2)

# Format BGR (Blue, Green, Red)
# Contoh: (255, 100, 200) = Purple-Pink
```

---

## 🐛 Troubleshooting

### ❌ Avatar Tidak Bergerak di VSeeFace

**Solusi:**
1. ✅ Pastikan **VMC protocol receiver** aktif di VSeeFace Settings
2. ✅ Check **port** harus `39539` di VSeeFace dan `main.py`
3. ✅ Pastikan **Body tracking** enabled di VSeeFace
4. ✅ Restart VSeeFace dan jalankan ulang `main.py`
5. ✅ Check Windows Firewall tidak block port 39539

### 🐌 FPS Rendah / Lag

**Untuk PC Low-end:**
```python
# Edit main.py
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # Turunkan resolution
model_complexity = 0                      # Lowest quality
```

**Tips Tambahan:**
- Close aplikasi lain yang berat
- Disable antivirus sementara
- Pastikan GPU driver up-to-date

### 🖐️ Tangan Tidak Terdeteksi

**Checklist:**
- ✅ Tangan dalam frame kamera
- ✅ Jarak tidak terlalu dekat/jauh (1-1.5m optimal)
- ✅ Pencahayaan cukup terang
- ✅ Jari tidak tertutup atau overlap
- ✅ Background kontras dengan tangan

### 👍 Jempol Tidak Menutup dengan Benar

**Tuning sensitivity:**
```python
THUMB_SENSITIVITY = 1.25  # Coba nilai 1.0 - 1.5
THUMB_SIGN_L = 1.0       # Coba flip: -1.0
THUMB_SIGN_R = -1.0      # Coba flip: 1.0
```

### 👁️ Eye Tracking Terlalu Sensitif

**Reduce sensitivity:**
```python
GAZE_SENSITIVITY = 1.0    # Turunkan dari 1.3
DEADZONE = 0.5           # Naikkan untuk lebih stabil
```

### 💥 Program Crash / Force Close

**Penyebab umum:**
- Webcam tidak support resolution yang di-set
- Memory/RAM tidak cukup
- MediaPipe model terlalu berat

**Solusi:**
```python
# Gunakan konfigurasi safe mode
model_complexity = 0
CAMERA_HEIGHT = 360
TARGET_FPS = 30
```

### 🔥 Firewall Blocking

Jika Windows Firewall block:
1. Control Panel → Windows Defender Firewall
2. Allow an app → Python
3. Check both Private and Public networks

---

## 📊 Performance Benchmark

**Test Configuration:**
- **Hardware**: Lenovo LOQ (Ryzen 5 7000 Series + RTX 4050 6GB)
- **Settings**: 640x480 @ 30fps, model_complexity=1
- **Tracking**: Face + Body + 10 Fingers

**Results:**
| Metric | Value |
|--------|-------|
| FPS | 28-32 (stable) |
| CPU Usage | ~35% |
| GPU Usage | ~15% |
| VRAM | ~1.5GB / 6GB |
| Latency | < 50ms |

---

## 🎓 Algorithms & Techniques

### Computer Vision Methods
1. **solvePnP** - 3D head pose estimation dari 2D landmarks
2. **Eye Aspect Ratio (EAR)** - Blink detection
3. **Distance-based Detection** - Finger curl & mouth opening
4. **Kalman Filtering** - Temporal smoothing untuk stabilitas
5. **Inverse Kinematics** - Arm rotation dari joint positions

### MediaPipe Models
- **Face Mesh**: 468 facial landmarks dengan iris tracking
- **Pose Detection**: 33 body landmarks untuk full body
- **Hand Tracking**: 21 landmarks per hand untuk finger control

### Data Communication
- **OSC/VMC Protocol**: Real-time data streaming via UDP
- **Bone Mapping**: Transform MediaPipe data ke Unity bone structure
- **Quaternion Rotation**: Euler to quaternion conversion untuk smooth rotation

---

## 📚 References

- [MediaPipe Documentation](https://google.github.io/mediapipe/) - Google's ML solutions
- [VSeeFace Official](https://www.vseeface.icu/) - VTuber software
- [VMC Protocol Spec](https://protocol.vmc.info/) - Virtual Motion Capture standard
- [OpenCV Documentation](https://docs.opencv.org/) - Computer vision library

---

## 📝 Project Info

**Purpose**: Tugas Mata Kuliah Pengolahan Citra Video (PCV)

**Author**: [@PanjiF25](https://github.com/PanjiF25)

**Repository**: [ProjectVtuberPCV](https://github.com/PanjiF25/ProjectVtuberPCV)

---

## 💬 Support & Contact

Untuk pertanyaan atau issue:
- 🐛 [Report Bug](https://github.com/PanjiF25/ProjectVtuberPCV/issues)
- 📖 Baca Troubleshooting section di atas
- 📧 Contact via GitHub

---

**Made for VTuber & Computer Vision enthusiasts** 🎭✨

⭐ Star repository ini jika helpful!
