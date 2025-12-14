# Motion Capture Full Body untuk VTuber
## Menggunakan MediaPipe dan VSeeFace

---

## Daftar Isi
- [Pendahuluan](#pendahuluan)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Fitur Utama](#fitur-utama)
- [Cara Kerja Sistem](#cara-kerja-sistem)
- [Implementasi](#implementasi)
- [Hasil dan Performa](#hasil-dan-performa)
- [Kesimpulan](#kesimpulan)

---

## Pendahuluan

### Latar Belakang
Proyek ini merupakan sistem **motion capture full body** untuk aplikasi VTuber yang dikembangkan menggunakan webcam standar dan teknologi computer vision. Berbeda dengan sistem motion capture tradisional yang membutuhkan perangkat mahal (ratusan juta rupiah), sistem ini memanfaatkan **MediaPipe** dan **VSeeFace** untuk menciptakan solusi tracking yang lebih terjangkau namun tetap akurat.

### Tujuan
Mengembangkan sistem tracking komprehensif yang dapat:
- Melacak gerakan wajah, mata, tubuh, tangan, dan jari secara real-time
- Memberikan performa stabil untuk streaming dan content creation
- Menjadi alternatif terjangkau untuk motion capture profesional

### Manfaat
- Memungkinkan content creator individual untuk menjadi VTuber tanpa investasi besar
- Implementasi praktis dari konsep pengolahan citra video
- Solusi open-source yang dapat dikembangkan lebih lanjut

---

## Teknologi yang Digunakan

### Library Utama
- **OpenCV** - Pengambilan dan pemrosesan frame dari webcam
- **MediaPipe** - Framework machine learning untuk deteksi landmark
  - 468 facial landmarks untuk wajah dan mata
  - 33 pose landmarks untuk tubuh
  - 21 hand landmarks per tangan (10 jari total)
- **Python-OSC** - Komunikasi dengan VSeeFace menggunakan protokol VMC
- **NumPy** - Operasi matematis dan array processing

### Protokol Komunikasi
**VMC (Virtual Motion Capture)** melalui **OSC (Open Sound Control)**
- Mengirim data tracking dari Python ke VSeeFace
- Real-time dengan latency minimal
- Support untuk avatar format VRM

---

## Fitur Utama

### 1. **Tracking Kepala (Head Tracking)**
Menggunakan algoritma **Perspective-n-Point (PnP)** untuk menghitung rotasi kepala:
- **Yaw** (kiri-kanan)
- **Pitch** (atas-bawah)  
- **Roll** (kemiringan)

Sistem menggunakan 6 landmark wajah kunci yang dipetakan ke model 3D untuk menghitung matriks rotasi secara akurat.

### 2. **Tracking Mata (Eye Tracking)**
- **Deteksi Kedipan** menggunakan **Eye Aspect Ratio (EAR)**
  - Formula: `EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)`
  - Threshold: Mata tertutup < 0.12, Mata terbuka > 0.20
- **Tracking Iris** untuk arah pandangan
  - Deteksi posisi iris relatif terhadap sudut mata
  - Normalisasi untuk parameter X dan Y

### 3. **Tracking Mulut (Mouth Tracking)**
- Mengukur jarak vertikal antara bibir atas dan bawah
- Dinormalisasi untuk parameter animasi (0.0 - 1.0)
- Range kalibrasi: 5.0 - 40.0 pixel

### 4. **Tracking Tubuh (Body/Pose Tracking)**
Menggunakan 33 pose landmarks dari MediaPipe:
- **Body Tilt**: Kemiringan horizontal tubuh
- **Body Roll**: Rotasi tubuh
- **Spine Position**: Gerakan tulang belakang

### 5. **Tracking Lengan (Arm Tracking)**
Menghitung rotasi sendi berdasarkan posisi:
- **Shoulder** - Gerakan bahu (naik/turun, rotasi)
- **Elbow** - Gerakan siku (tekuk/lurus)
- **Wrist** - Orientasi pergelangan tangan

Menggunakan **Inverse Kinematics** untuk menghitung quaternion rotasi dari vektor posisi.

### 6. **Tracking Jari (Finger Tracking)**
Deteksi untuk **10 jari** (kedua tangan):
- **Thumb** (Jempol) - Axis Y, sensitivitas 1.15
- **Index** (Telunjuk) - Axis Z
- **Middle** (Tengah) - Axis Z  
- **Ring** (Manis) - Axis Z
- **Little** (Kelingking) - Axis Z

**Metode Deteksi:**
```
curl = (dist_tip_to_wrist / dist_palm) normalized
```
- Threshold berbeda untuk jempol dan jari lain
- Deadzone untuk mengurangi jitter (0.08 - 0.10)

---

## Cara Kerja Sistem

### Alur Kerja Per Frame

```
1. Capture frame dari webcam
2. Flip horizontal (mirror effect)
3. Proses dengan MediaPipe:
   - Face Mesh (468 landmarks)
   - Pose Detection (33 landmarks)
   - Hand Tracking (21 landmarks × 2)
4. Ekstraksi data landmark
5. Kalkulasi parameter tracking
6. Smoothing dengan Kalman Filter
7. Kirim ke VSeeFace via OSC/VMC
8. Display preview window
```

### Teknik Optimasi

#### 1. **Kalman Filtering**
Mengurangi noise pada data tracking:
```python
class Stabilizer:
    - State prediction
    - Measurement update
    - Process noise: 0.0001
    - Measurement noise: 0.1
```

#### 2. **Exponential Smoothing**
Menghaluskan transisi gerakan:
```python
smooth_value = alpha × new_value + (1-alpha) × old_value
```

Alpha berbeda per parameter:
- Kepala: 0.4
- Mata: 0.6 (lebih responsif)
- Iris: 0.2 (lebih halus)
- Tubuh: 0.2-0.3
- Lengan: 0.7
- Jari: disesuaikan

#### 3. **Adaptive Deadzone**
Mencegah jitter pada gerakan kecil:
```python
if abs(new_value - old_value) < deadzone:
    return old_value  # Tidak update
```

Deadzone:
- Parameter utama: 0.55
- Jari: 0.08 - 0.10
- Jempol: 0.10

#### 4. **OSC Throttling**
Mengurangi bandwidth komunikasi:
- Update hanya jika perubahan > threshold
- Rate limit: ~60Hz maksimal
- Skip message untuk perubahan kecil

---

## Implementasi

### Spesifikasi Hardware
- **Webcam**: HD 720p atau lebih tinggi
- **Processor**: Intel Core i5 gen 8 atau lebih
- **GPU**: NVIDIA RTX 4050 (atau setara)
- **RAM**: Minimal 8GB
- **OS**: Windows 10/11

### Spesifikasi Software
```
Python 3.8+
OpenCV 4.x
MediaPipe 0.10.x
Python-OSC
NumPy
VSeeFace (aplikasi eksternal)
```

### Konfigurasi Penting

#### Parameter MediaPipe
```python
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 1  # Balance antara akurasi dan speed
```

#### Parameter OSC
```python
OSC_IP = "127.0.0.1"
OSC_PORT = 5555
TARGET_FPS = 30
```

#### Kalibrasi Tracking
**Lengan:**
```python
ARM_GAIN_XY = 0.95   # Sensitivitas horizontal/vertikal
ARM_GAIN_Z = 0.55    # Sensitivitas depth
ARM_SMOOTHING = 0.7  # Factor smoothing
```

**Jari:**
```python
FINGER_SENSITIVITY = 1.1
FINGER_DEADZONE = 0.08
THUMB_SENSITIVITY = 1.15
THUMB_DEADZONE = 0.10
```

### Struktur Kode Utama

#### 1. Inisialisasi
```python
# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Setup OSC Client
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# Webcam
cap = cv2.VideoCapture(WEBCAM_ID)
```

#### 2. Main Loop
```python
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror
    
    # Process MediaPipe
    face_results = face_mesh.process(frame)
    pose_results = pose.process(frame)
    hand_results = hands.process(frame)
    
    # Calculate tracking parameters
    # ... (head, eyes, mouth, body, arms, fingers)
    
    # Send to VSeeFace
    send_osc_data()
    
    # Display
    cv2.imshow('Tracking Preview', frame)
```

#### 3. Helper Functions
- `euler_to_quaternion()` - Konversi rotasi Euler ke Quaternion
- `get_limb_rotation()` - Hitung rotasi sendi dari posisi
- `get_finger_curl()` - Deteksi tingkat tekukan jari
- `calculate_ear()` - Hitung Eye Aspect Ratio
- `smooth_quaternion()` - Smoothing quaternion
- `apply_deadzone()` - Terapkan deadzone

---

## Hasil dan Performa

### Performa Sistem
- **FPS**: 28-32 (stabil)
- **Latency**: < 50ms
- **CPU Usage**: 40-60%
- **GPU Usage**: 30-50%

### Akurasi Tracking

| Komponen | Akurasi | Responsivitas |
|----------|---------|---------------|
| Kepala | ★★★★★ | Sangat Baik |
| Mata | ★★★★☆ | Baik |
| Mulut | ★★★★★ | Sangat Baik |
| Tubuh | ★★★★☆ | Baik |
| Lengan | ★★★★☆ | Baik |
| Jari | ★★★☆☆ | Cukup Baik |

### Kelebihan
- Hanya butuh webcam standar  
- Tracking full body tanpa sensor tambahan  
- Real-time dengan latency rendah  
- Gratis dan open-source  
- Performa stabil untuk streaming  
- Support 10 jari (full hand tracking)

### Keterbatasan
- Memerlukan pencahayaan yang baik  
- Tracking jari sensitif terhadap oklusi  
- Background harus kontras dengan subjek  
- Memerlukan GPU untuk performa optimal  
- Terkadang jitter pada gerakan cepat

### Kondisi Optimal
- Pencahayaan: Terang merata dari depan
- Jarak webcam: 50-100 cm
- Background: Solid color atau minim distraksi
- Posisi: Duduk tegak menghadap kamera
- Pakaian: Kontras dengan background

---

## Kesimpulan

### Pencapaian
Proyek ini berhasil mengimplementasikan sistem motion capture full body yang komprehensif untuk aplikasi VTuber. Sistem dapat melakukan tracking:
- Wajah dengan 468 landmark points
- Tubuh dengan 33 pose landmarks
- 10 jari (kedua tangan) dengan 21 landmarks per tangan

Kombinasi MediaPipe dan VSeeFace menghasilkan sistem yang **responsif**, **akurat**, dan **terjangkau** dibandingkan solusi komersial yang bisa mencapai ratusan juta rupiah.

### Pembelajaran Teknis
1. **Computer Vision**: Implementasi algoritma PnP, EAR, dan landmark detection
2. **Signal Processing**: Penerapan Kalman Filter dan exponential smoothing
3. **3D Mathematics**: Penggunaan quaternion dan inverse kinematics
4. **Network Communication**: Protokol OSC/VMC untuk real-time data transfer
5. **Performance Optimization**: Teknik throttling, deadzone, dan adaptive filtering

### Aplikasi Praktis
Sistem ini **suitable** untuk:
- Live streaming VTuber
- Content creation YouTube/TikTok
- Virtual meeting dengan avatar
- Game character control
- Motion capture recording

### Pengembangan Masa Depan

**Peningkatan Teknis:**
- Implementasi machine learning untuk prediksi gerakan
- Auto-calibration per user
- Support multiple avatar models
- Recording dan playback motion data
- Full body tracking dengan sensor tambahan

**Optimasi:**
- Performa untuk low-end hardware
- Tracking lebih responsif untuk gerakan cepat
- Ekspresi wajah yang lebih kompleks
- Stabilitas jari yang lebih baik

**Integrasi:**
- Plugin untuk OBS Studio
- Support untuk software streaming lain
- Mobile app support (Android/iOS)
- Cloud-based processing option

