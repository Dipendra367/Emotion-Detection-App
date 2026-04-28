# 😊 Emotion Detection App

> A real-time facial emotion detection system powered by deep learning (CNN) and OpenCV.  
> Trained on the **FER2013** dataset to classify **7 human emotions** live from webcam.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv"/>
  <img src="https://img.shields.io/badge/Keras-red?style=for-the-badge&logo=keras"/>
</div>

---

## 📸 Live Demo

<img width="1280" height="591" alt="WhatsApp Image 2026-04-28 at 9 54 06 PM" src="https://github.com/user-attachments/assets/1a7c2ee2-d6b9-43d4-bf56-575f2a1d76c7" />


<div align="center">
  
  <br/><br/>
  <b>🎯 Detected Emotion: 😊 Happy</b>
</div>

> 💡 To try it yourself — clone the repo, run `realtime_emotion_detect.py`, and your webcam will detect emotions live!

---

## 🎭 Emotions Detected

| # | Emotion  | Emoji | Description |
|---|----------|-------|-------------|
| 1 | Angry    | 😠    | Detects frustration or rage |
| 2 | Disgust  | 🤢    | Detects expressions of disgust |
| 3 | Fear     | 😨    | Detects fearful expressions |
| 4 | Happy    | 😊    | Detects smiles and joy |
| 5 | Neutral  | 😐    | Detects calm, no emotion |
| 6 | Sad      | 😢    | Detects sadness or grief |
| 7 | Surprise | 😲    | Detects shocked expressions |

---

## 🧠 How It Works

```
📷 Webcam Feed
     ↓
🔲 Face Detection (OpenCV Haar Cascade)
     ↓
✂️  Face Cropped & Resized to 48x48
     ↓
🧠 CNN Model Predicts Emotion
     ↓
🏷️  Emotion Label Displayed on Screen
```

---

## 📁 Project Structure

```
Emotion-Detection-App/
│
├── archive/
│   ├── train/          # Training images (FER2013)
│   └── test/           # Test images (FER2013)
│
├── demo/
│   └── your_photo.jpg  # Demo photo
│
├── main.py                    # Main entry point
├── train_emotion_model.py     # Model training script
├── realtime_emotion_detect.py # Live webcam detection
├── convert_csv_to_images.py   # Dataset preprocessing
├── emotion_model.h5           # Trained model weights
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Dipendra367/Emotion-Detection-App.git
cd Emotion-Detection-App
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Download the **FER2013** dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it in the `archive/` folder:
```
archive/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    ├── happy/
    └── ...
```

---

## 🏋️ Train the Model

```bash
python train_emotion_model.py
```
> This will train the CNN and save the model as `emotion_model.h5`

---

## 🎥 Run Real-Time Detection

```bash
python realtime_emotion_detect.py
```
> Make sure your webcam is connected. Press `Q` to quit.

---

## 📊 Model Architecture

- **Input:** 48x48 grayscale face image
- **Layers:** Conv2D → BatchNorm → MaxPooling → Dropout → Dense
- **Output:** 7 emotion classes (Softmax)
- **Dataset:** FER2013 (~35,000 images)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| TensorFlow / Keras | Model training |
| OpenCV | Face detection & webcam |
| NumPy | Data processing |
| Matplotlib | Visualizations |

---

## 🔮 Future Improvements

- [ ] Deploy as a web app (Flask / Streamlit)
- [ ] Add multi-face detection
- [ ] Mobile app version
- [ ] Improve model accuracy with more data
- [ ] Add emotion history graph

---

## 👤 Author

**Dipendra**  
🔗 [GitHub](https://github.com/Dipendra367)

---

## ⭐ Support

If you found this project helpful, please give it a **star ⭐** on GitHub — it means a lot!
