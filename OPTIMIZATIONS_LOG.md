# Log Optimalisasi - Mengurangi Sensitivitas Gerakan

## Tanggal: 9 Desember 2025

### Masalah yang Diidentifikasi:
- Pergerakan terlalu sensitif dan jittery
- Gerakan tiba-tiba (sudden movements) tidak ter-handle dengan baik
- Finger tracking terlalu reactive

### Optimalisasi yang Diterapkan:

#### 1. **Peningkatan Smoothing System (Multi-Layer)**
   - **Smoothing Window**: Ditingkatkan dari 2 → 4 frame
   - **Double-Stage Filtering**: Kalman Filter + Moving Average Filter
   - **Kalman Parameters**: Ditingkatkan untuk lebih aggressive smoothing
     - `cov_process`: 0.01 → 0.005 (head), 0.001 → 0.0005 (eyes)
     - `cov_measure`: 0.15 → 0.25 (head), 0.2 → 0.3 (eyes)

#### 2. **Adaptive Deadzone System**
   - **Base Deadzone**: 0.45 → 0.55 (lebih tinggi untuk reduce jitter)
   - **Adaptive Multiplier**: 1.5x saat perubahan cepat terdeteksi
   - Formula: `adaptive_deadzone = DEADZONE * (1.0 + (velocity / 10.0) * MULTIPLIER)`
   - Mencegah jitter saat gerakan cepat dengan deadzone yang dinamis

#### 3. **Eye Tracking Improvements**
   - **Gaze Sensitivity**: 1.3 → 1.1 (gerakan mata lebih halus)
   - **Eye Multiplier**: Reduced dari 25/20 → 20/16 (X/Y)
   - **Outlier Detection**: Threshold lebih ketat 0.5 → 0.4
   - **Range Limiting**: Clipping range dipersempit untuk mencegah extreme values
     - Horizontal: ±0.8 → ±0.7
     - Vertical: ±0.6 → ±0.5

#### 4. **Arm Tracking Smoothing**
   - **Exponential Smoothing**: Alpha = 0.7 untuk quaternion smoothing
   - **Gain Reduction**: 
     - ARM_GAIN_XY: 1.0 → 0.95
     - ARM_GAIN_Z: 0.6 → 0.55
   - **Quaternion Interpolation**: Smooth transition antar frame
   - Cache previous rotation untuk mencegah sudden jumps

#### 5. **Finger Tracking Stability**
   - **Sensitivity Reduction**:
     - FINGER_SENSITIVITY: 1.2 → 1.1
     - THUMB_SENSITIVITY: 1.25 → 1.15
   - **Deadzone Implementation**:
     - FINGER_DEADZONE: 0.08 (8% threshold)
     - THUMB_DEADZONE: 0.1 (10% threshold)
   - **Improved Curl Detection**:
     - Range lebih stabil dengan np.clip
     - Sensitivity multiplier: × 0.95
     - Previous value caching untuk prevent jitter

#### 6. **OSC Message Optimization**
   - **Rate Limiting**: 60Hz max (0.016s minimum interval)
   - **Throttling Function**: Skip messages jika perubahan < 0.001
   - **Value Caching**: Mencegah duplicate messages
   - Mengurangi bandwidth dan improve performance

### Parameter Sebelum vs Sesudah:

| Parameter | Sebelum | Sesudah | Perubahan |
|-----------|---------|---------|-----------|
| DEADZONE | 0.45 | 0.55 | +22% |
| SMOOTHING_WINDOW | 2 | 4 | +100% |
| GAZE_SENSITIVITY | 1.3 | 1.1 | -15% |
| ARM_GAIN_XY | 1.0 | 0.95 | -5% |
| FINGER_SENSITIVITY | 1.2 | 1.1 | -8% |
| Eye X Mult | 25.0 | 20.0 | -20% |
| Eye Y Mult | 20.0 | 16.0 | -20% |

### Fitur Baru yang Ditambahkan:

1. **MovingAverageFilter Class**
   ```python
   class MovingAverageFilter:
       def __init__(self, window_size=4):
           self.window_size = window_size
           self.values = deque(maxlen=window_size)
       
       def update(self, value):
           self.values.append(value)
           return sum(self.values) / len(self.values)
   ```

2. **Smooth Quaternion Function**
   ```python
   def smooth_quaternion(q_new, q_prev, alpha=0.7):
       """Exponential smoothing for quaternions"""
       return [alpha * n + (1 - alpha) * p for n, p in zip(q_new, q_prev)]
   ```

3. **Apply Deadzone Function**
   ```python
   def apply_deadzone(value, prev_value, deadzone):
       """Apply deadzone to reduce jitter"""
       if abs(value - prev_value) < deadzone:
           return prev_value
       return value
   ```

4. **OSC Throttling Function**
   ```python
   def send_osc_throttled(address, value, threshold=0.001):
       """Send OSC message dengan throttling untuk reduce bandwidth"""
       # Skip jika perubahan terlalu kecil
   ```

### Cache Variables Ditambahkan:

- `prev_arm_rotations`: Dictionary untuk arm smoothing
- `prev_finger_curls_L`: Array untuk left hand deadzone
- `prev_finger_curls_R`: Array untuk right hand deadzone
- `last_sent_values`: Dictionary untuk OSC throttling

### Expected Results:

✅ Gerakan lebih smooth dan natural
✅ Reduced jitter pada tracking statis
✅ Gerakan cepat ter-handle dengan lebih baik
✅ Finger tracking lebih stabil
✅ Eye tracking tidak over-reactive
✅ Arm movements lebih fluid
✅ Bandwidth OSC lebih efisien
✅ Performance improvement (reduced CPU overhead)

### Testing Guidelines:

1. Test gerakan kepala pelan → harus smooth tanpa jitter
2. Test gerakan kepala cepat → tidak ada over-swing
3. Test eye tracking → tidak jumping atau twitching
4. Test finger curling → smooth transition, no flickering
5. Test arm movements → fluid motion, no sudden jumps
6. Monitor FPS → harus stabil ~30fps

### Catatan Tambahan:

- Semua optimalisasi bersifat non-breaking changes
- Backward compatible dengan konfigurasi sebelumnya
- Dapat di-tweak lebih lanjut sesuai preferensi user
- Disarankan untuk test dengan lighting yang baik
- GPU acceleration tetap aktif untuk performance optimal

### Cara Rollback (jika diperlukan):

Jika gerakan terasa terlalu lambat atau sluggish:
1. Kurangi `SMOOTHING_WINDOW` dari 4 ke 3
2. Turunkan `DEADZONE` dari 0.55 ke 0.50
3. Tingkatkan `GAZE_SENSITIVITY` dari 1.1 ke 1.2
4. Sesuaikan `ARM_SMOOTHING` dari 0.7 ke 0.8 (lebih responsive)
