# AR Virtual Keyboard
A simple **Augmented Reality (AR) keyboard** built using **OpenCV** and **MediaPipe** that allows typing using **hand gestures** through a webcam. The system tracks the **index finger** and detects deliberate tap motions to simulate key presses.

## ✨ Features
- Real-time **hand & finger tracking**  
- **Hover + tap detection** to avoid mistakes   
- Special keys: `SPACE`, `DEL`, `CLEAR` 
- Visual feedback: keys turn **green** when pressed

## 🛠️ How It Works
1. **Hand tracking**: MediaPipe detects hand landmarks
2. **Key detection**: Checks if fingertip is near a key
3. **Tap detection**: Stable hover + downward motion

## Sample Output
![Image](https://github.com/user-attachments/assets/3ca01152-ab70-4a21-9698-f5131b77da1a)

## Usage
```bash
git clone https://github.com/nadarkavin/AR--Virtual-Keyboard.git
cd AR--Virtual-Keyboard
pip install -r requirements.txt
python AR-keyboard.py
```
## Requirements
- Python 3.8+
- `opencv-python` (Version: 4.12.0.88)
- `mediapipe` (Version: 0.10.9)
- `numpy` (Version: 2.2.6)
