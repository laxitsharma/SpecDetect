
# SpecDetect

**SpecDetect** is a desktop GUI application that uses computer vision to detect whether a person is wearing glasses. It leverages OpenCV for face and eyeglass detection, and provides an intuitive interface for collecting data, training a recognition model, and testing it in real time.

---

## 🚀 Features

- 🎥 Real-time webcam feed with face and glasses detection.
- 🖼️ Easy data capture for "with glasses" and "without glasses" images.
- 🧠 On-device model training using OpenCV's face recognizer.
- ✅ Instant testing and prediction with confidence scores.
- 🧩 Built using Python, OpenCV, Tkinter, and PIL.

---

## 🛠 Requirements

- Python 3.6+
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- Tkinter (comes pre-installed with Python)
- Pillow
- Numpy
- Pandas

Install dependencies:
```bash
pip install -r req.txt
```

---

## 📦 Usage

1. **Run the app**:
   ```bash
   python glasses_detection_ui.py
   ```

2. **Capture Data**:
   - Click "Start With Glasses" or "Start Without Glasses"
   - Images will be saved automatically every 0.5s

3. **Train Model**:
   - Click "Train Model" once you have enough samples

4. **Test Model**:
   - Click "Test Detection" and view live predictions

---

## 📁 Project Structure

```
.
├── glasses_detection_ui.py
├── haarcascade_frontalface_default.xml
├── haarcascade_eye_tree_eyeglasses.xml
├── req.txt
├── data/
│   ├── with_glasses/
│   └── without_glasses/
```

---

## 📸 Sample

![Sample UI Screenshot](#) <!-- Add your screenshot path here if available -->

---

## 📄 License

This project is open-source and available under the MIT License.
