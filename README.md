# Jarvis

## Run

```powershell
cd C:\Users\vicky\Jarvis
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python gestures.py
```

## Controls

- Pinch a shape tile on the right panel to switch the active shape.
- Pinch the CAPTURE tile to save a screenshot into the screenshots folder.
- Use one hand to click or drag the hologram.
- Use two hands in pinch or grab mode to scale and rotate the active object.
- Swipe quickly to switch the neon color palette.

## Files

- tracking.py: hand tracking and MediaPipe setup.
- gestures.py: gesture recognition, holographic UI, shape selection, and screenshots.
- hologram.py: older standalone demo.