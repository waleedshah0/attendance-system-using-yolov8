# Attendance System (Server-Optimized)

A headless, server-ready attendance tracking system that detects people, recognizes faces, determines IN/OUT relative to a center line, logs attendance to CSV, and sends Slack notifications. Optimized for CPU-only servers and gracefully falls back when YOLO is unavailable.

## Features
- RTSP camera ingestion with resilient frame reader
- Person detection: YOLOv8 (if available) → OpenCV HOG fallback → Motion fallback
- Face detection + ArcFace embeddings via InsightFace
- FAISS-powered face recognition against employee embeddings
- IN/OUT using a horizontal center line (above line = OUT, below line = IN)
- CSV logging (12-hour time, hours:minutes only)
- Slack notifications via webhook
- Headless operation (no GUI)

## Repository Layout
- `main_server_fixed.py` — main server runtime
- `recognize.py` — FAISS index and recognition utilities
- `build_embeddings.py` — build embeddings and FAISS index
- `embeddings/` — generated files: `face_index.faiss`, `labels.pkl`
- `attendance_reversed.csv` — attendance log (auto-created)
- `requirements.txt` — dependencies
- `SERVER_DEPLOYMENT_GUIDE.md` — deployment notes

## Requirements
- Python 3.9+
- Linux server (recommended), Windows for development
- RTSP-capable camera

## Installation
```bash
# 1) Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
Edit `main_server_fixed.py` as needed:

- RTSP camera URL (inside `main()`):
```python
camera_url = "rtsp://user:pass@host:port"
```

- Slack webhook (near top of file):
```python
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/XXX/YYY/ZZZ"
SLACK_ENABLED = True  # set False to disable
```

- Time format: CSV and Slack use 12-hour HH:MM AM/PM by default.
- Line logic: above center line = OUT, below center line = IN (already implemented).

## Build Employee Embeddings
Create/update the FAISS index and labels from your face dataset:
```bash
source venv/bin/activate
python build_embeddings.py
```
This generates/updates `embeddings/face_index.faiss` and `embeddings/labels.pkl` (names like `Usman`, `Waleed`, ...).

## Running
### Foreground
```bash
source venv/bin/activate
python main_server_fixed.py
```

### Persist with screen
```bash
sudo apt-get update && sudo apt-get install -y screen
screen -S attendance
source venv/bin/activate
python main_server_fixed.py
# Detach: Ctrl+A then D   |   Reattach: screen -r attendance
```

### Persist with nohup
```bash
source venv/bin/activate
nohup python main_server_fixed.py > attendance.log 2>&1 &
# Tail logs
tail -f attendance.log
```

## CSV Output
`attendance_reversed.csv` (auto-created):
```
Name,ID,Direction,Timestamp,Status
Usman,0,IN,2025-10-17 02:45 PM,PRESENT
Waleed,1,OUT,2025-10-17 03:10 PM,ABSENT
```
- IN when detected below the center line; OUT when above.
- Only recognized employees are logged (Unknowns skipped).
- Duplicate suppression: same track ID suppressed for 60 seconds.

## Slack Notifications
Each attendance event posts a message with:
- Employee name
- Action (IN/OUT) with emoji
- Time (12-hour HH:MM)
- Status (PRESENT/ABSENT)

If Slack is unreachable, CSV still logs; errors print to console.

## Tuning & Stability
Key parameters in `ServerAttendanceSystem`:
- `ui_fps`: lower to reduce CPU load (default 8)
- `imgsz`: 320 for CPU stability
- `conf_thres`, `iou_thres`, `max_det`: detection controls
- `min_face_area`: minimum valid face crop area (in pixels)
- `face_check_cooldown`: recognition cadence (internally recognized aggressively)

Threading/CPU env vars are pre-configured at the top of `main_server_fixed.py` for server stability.

## How It Works (Summary)
1. RTSP frames via `FrameReader` (low latency)
2. Detect people (YOLO → HOG → motion fallback)
3. Detect face and extract ArcFace embedding
4. Recognize against FAISS index
5. Compare centroid to center line: above=OUT, below=IN
6. Append CSV and send Slack (recognized employees only)

## Troubleshooting
- YOLO errors: system falls back to HOG; still functional
- Always `Unknown`: ensure clear faces; lower `min_face_area`; verify `embeddings/labels.pkl`
- No Slack: verify `SLACK_WEBHOOK_URL`, outbound network, and `SLACK_ENABLED=True`
- Headless: no GUI is used; all logs go to console/log file
- Performance: reduce `ui_fps`, keep `imgsz=320`

## Security Notes
- Keep the Slack webhook private (consider environment variables via `os.getenv`).
- Do not commit proprietary employee face images.

## License
Proprietary — internal organizational use, Created by Waleed Shah.
