# Jarvis

## Run

```powershell
cd C:\Users\vicky\Jarvis
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

The HTML UI opens in your browser and the backend keeps the webcam, gestures, and screenshot capture running.

Screenshot behavior:

- The Copy Screenshot button downloads the PNG and tries to copy it to your clipboard.
- Press `S` to trigger the same screenshot flow from the keyboard.

Keyboard shortcuts:

- `P` pauses gesture interactions.
- `C` resumes gesture interactions.
- `S` copies and downloads the current screenshot.
- `Q` or `Esc` exits.

## Controls

- Pinch a shape tile on the right panel to switch the active shape.
- Pinch the CAPTURE tile to save a screenshot into the screenshots folder.
- Use one hand to click or drag the hologram.
- Use two hands in pinch or grab mode to scale and rotate the active object.
- Swipe quickly to switch the neon color palette.
- Your hand position is shown as a live marker and coordinate readout inside the interface.
- Fist pauses gestures.
- Open palm resumes gestures.
- Peace changes the theme.
- Thumbs-up takes a screenshot.
- Point follows your index finger.
- The browser UI is now the primary interface, with backend gesture processing and a cleaner layout.

## Files

- tracking.py: hand tracking and MediaPipe setup.
- gestures.py: gesture recognition, holographic UI, shape selection, and screenshots.
- hologram.py: older standalone demo.