const toast = document.getElementById('toast');
const statusPill = document.getElementById('statusPill');
const themePill = document.getElementById('themePill');
const gestureName = document.getElementById('gestureName');
const statusText = document.getElementById('statusText');
const handCount = document.getElementById('handCount');
const fpsValue = document.getElementById('fpsValue');
const modeValue = document.getElementById('modeValue');
const shapeValue = document.getElementById('shapeValue');
const handList = document.getElementById('handList');
const capturePath = document.getElementById('capturePath');
const screenshotBtn = document.getElementById('screenshotBtn');
const shapeTiles = Array.from(document.querySelectorAll('.shape-tile'));
const controlButtons = Array.from(document.querySelectorAll('[data-action]'));

let activeShape = 'HEXAGON';

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove('show'), 2200);
}

async function postControl(action, value = null) {
  const response = await fetch('/api/control', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, value }),
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || 'Control request failed');
  }
  return payload.state;
}

function updateShapeButtons() {
  shapeTiles.forEach((tile) => {
    tile.classList.toggle('active', tile.dataset.shape === activeShape);
  });
}

async function copyScreenshot() {
  const response = await fetch('/api/screenshot');
  if (!response.ok) {
    throw new Error('Could not capture screenshot');
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);

  const downloadLink = document.createElement('a');
  downloadLink.href = url;
  downloadLink.download = `jarvis-screenshot-${Date.now()}.png`;
  downloadLink.click();

  try {
    if ('clipboard' in navigator && 'ClipboardItem' in window) {
      await navigator.clipboard.write([
        new ClipboardItem({ [blob.type]: blob }),
      ]);
      showToast('Screenshot copied to clipboard and downloaded');
    } else {
      showToast('Screenshot downloaded');
    }
  } catch (error) {
    showToast('Clipboard copy not available, screenshot downloaded');
  }

  URL.revokeObjectURL(url);
}

async function refreshState() {
  const response = await fetch('/api/state');
  const state = await response.json();

  gestureName.textContent = state.gesture || 'IDLE';
  statusText.textContent = state.last_status || 'Ready';
  handCount.textContent = state.hand_count ?? 0;
  fpsValue.textContent = state.fps ?? 0;
  modeValue.textContent = state.enabled ? 'Running' : 'Paused';
  shapeValue.textContent = state.selected_shape || activeShape;
  activeShape = state.selected_shape || activeShape;
  themePill.textContent = `Theme: ${state.theme || 'Cyan'}`;
  statusPill.textContent = state.enabled ? 'Gestures active' : 'Gestures paused';

  capturePath.textContent = state.last_screenshot_path || 'No capture yet.';
  updateShapeButtons();

  handList.innerHTML = '';
  if (!state.hands || state.hands.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'hand-row empty';
    empty.textContent = 'No hands detected yet.';
    handList.appendChild(empty);
    return;
  }

  state.hands.forEach((hand) => {
    const row = document.createElement('div');
    row.className = 'hand-row';
    row.innerHTML = `
      <strong>Hand ${hand.id} - ${hand.gesture}</strong>
      <div>x: ${hand.x} y: ${hand.y}</div>
      <div>pinch: ${hand.pinch} | grab: ${hand.grab}</div>
    `;
    handList.appendChild(row);
  });
}

controlButtons.forEach((button) => {
  button.addEventListener('click', async () => {
    try {
      const state = await postControl(button.dataset.action);
      showToast(state.last_status || 'Updated');
      await refreshState();
    } catch (error) {
      showToast(error.message);
    }
  });
});

shapeTiles.forEach((tile) => {
  tile.addEventListener('click', async () => {
    try {
      const state = await postControl('shape', tile.dataset.shape);
      activeShape = state.selected_shape || tile.dataset.shape;
      updateShapeButtons();
      showToast(`Shape set to ${activeShape}`);
    } catch (error) {
      showToast(error.message);
    }
  });
});

screenshotBtn.addEventListener('click', async () => {
  try {
    await copyScreenshot();
    await refreshState();
  } catch (error) {
    showToast(error.message);
  }
});

document.addEventListener('keydown', async (event) => {
  const key = event.key.toLowerCase();
  if (key === 'p') {
    await postControl('pause');
    showToast('Gestures paused');
  } else if (key === 'c') {
    await postControl('resume');
    showToast('Gestures resumed');
  } else if (key === 's') {
    await copyScreenshot();
  } else if (['1', '2', '3', '4'].includes(key)) {
    const map = { '1': 'CIRCLE', '2': 'SQUARE', '3': 'TRIANGLE', '4': 'HEXAGON' };
    await postControl('shape', map[key]);
    showToast(`Shape set to ${map[key]}`);
  }
});

refreshState();
setInterval(refreshState, 350);