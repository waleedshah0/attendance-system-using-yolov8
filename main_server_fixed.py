"""
Server-optimized version with aggressive YOLO compatibility fixes
This version addresses the "could not create a primitive" error
"""

import os
import sys

# Aggressive environment variables for maximum compatibility
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;0|stimeout;500000|buffer_size;512|fflags;nobuffer|flags;low_delay|analyzeduration;1000000|probesize;1000000"

# Disable all OpenCV threading and problematic modules
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["OPENCV_IO_ENABLE_JASPER"] = "0"
os.environ["OPENCV_IO_ENABLE_HDR"] = "0"
os.environ["OPENCV_IO_ENABLE_JPEG"] = "1"
os.environ["OPENCV_IO_ENABLE_PNG"] = "1"

# Force single-threading across all libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Force CPU-only operation
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_USE_CUDA_DSA"] = "0"

import cv2
import csv
import time
import torch
import numpy as np
from datetime import datetime, timedelta
from threading import Thread, Lock
import warnings
import requests
import json 
warnings.filterwarnings("ignore")

# Configure PyTorch for maximum compatibility
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Configure OpenCV for single-thread environment
try:
    cv2.setNumThreads(1)
    print("[INFO] OpenCV configured for single-thread environment")
except Exception as e:
    print(f"[WARN] Could not configure OpenCV threading: {e}")

# Import YOLO with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("[INFO] YOLO imported successfully")
except Exception as e:
    print(f"[ERROR] YOLO import failed: {e}")
    YOLO_AVAILABLE = False

# Import InsightFace with error handling
try:
    from insightface import app
    INSIGHTFACE_AVAILABLE = True
    print("[INFO] InsightFace imported successfully")
except Exception as e:
    print(f"[ERROR] InsightFace import failed: {e}")
    INSIGHTFACE_AVAILABLE = False

# Import recognition module
try:
    from recognize import recognize_face
    RECOGNIZE_AVAILABLE = True
    print("[INFO] Recognition module imported successfully")
except Exception as e:
    print(f"[ERROR] Recognition module import failed: {e}")
    RECOGNIZE_AVAILABLE = False

# Slack configuration
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/gggggggggggg/BBBBBBB/yyyyyyyyyyyyy"
SLACK_ENABLED = True

def send_slack_notification(name, direction, timestamp, status):
    """
    Send attendance notification to Slack channel
    """
    if not SLACK_ENABLED:
        return
    
    try:
        # Create the message
        emoji = "✅" if direction == "IN" else "❌"
        status_emoji = "🟢" if status == "PRESENT" else "🔴"
        
        message = {
            "text": f"{emoji} *Attendance Update*",
            "attachments": [
                {
                    "color": "good" if direction == "IN" else "warning",
                    "fields": [
                        {
                            "title": "Employee",
                            "value": name,
                            "short": True
                        },
                        {
                            "title": "Action",
                            "value": f"{direction} {status_emoji}",
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": timestamp,
                            "short": True
                        },
                        {
                            "title": "Status",
                            "value": status,
                            "short": True
                        }
                    ],
                    "footer": "Attendance System",
                    "ts": int(time.time())
                }
            ]
        }
        
        # Send to Slack
        response = requests.post(
            SLACK_WEBHOOK_URL,
            data=json.dumps(message),
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"[SLACK] ✓ Notification sent for {name} - {direction}")
        else:
            print(f"[SLACK] ✗ Failed to send notification: {response.status_code}")
            
    except Exception as e:
        print(f"[SLACK] ✗ Error sending notification: {e}")

# ------------------------------ Frame Reader ------------------------------
class FrameReader(Thread):
    """
    Separate thread that continuously pulls frames from the RTSP camera
    """
    def __init__(self, src, target_size=(640, 480)):
        super().__init__(daemon=True)
        self.src = src
        self.target_w, self.target_h = target_size
        
        # Try multiple backends for server compatibility
        backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
        self.cap = None
        
        for backend in backends:
            try:
                print(f"[INFO] Trying backend: {backend}")
                self.cap = cv2.VideoCapture(self.src, backend)
                if self.cap.isOpened():
                    print(f"[INFO] Successfully opened camera with backend: {backend}")
                    break
                else:
                    self.cap.release()
                    self.cap = None
            except Exception as e:
                print(f"[WARN] Backend {backend} failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        if not self.cap or not self.cap.isOpened():
            raise ValueError(f"Could not open camera with any backend: {self.src}")

        # Configure camera properties
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_h)
            # Try different codecs
            codecs = ['MJPG', 'H264', 'XVID']
            for codec in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                    print(f"[INFO] Set codec to: {codec}")
                    break
                except Exception:
                    continue
        except Exception as e:
            print(f"[WARN] Could not set camera properties: {e}")

        self.lock = Lock()
        self.latest_frame = None
        self.last_ok_time = time.time()
        self.stopped = False

    def run(self):
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        
        while not self.stopped:
            try:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    reconnect_attempts += 1
                    if reconnect_attempts > max_reconnect_attempts:
                        print(f"[ERROR] Max reconnection attempts reached. Stopping frame reader.")
                        break
                    
                    print(f"[WARN] Frame read failed, attempting reconnection ({reconnect_attempts}/{max_reconnect_attempts})")
                    time.sleep(1.0)
                    
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    
                    # Try to reconnect
                    backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
                    for backend in backends:
                        try:
                            self.cap = cv2.VideoCapture(self.src, backend)
                            if self.cap.isOpened():
                                print(f"[INFO] Reconnected with backend: {backend}")
                                reconnect_attempts = 0
                                break
                        except Exception as e:
                            print(f"[WARN] Reconnection with backend {backend} failed: {e}")
                            continue
                    else:
                        print("[ERROR] All reconnection attempts failed")
                        continue

                reconnect_attempts = 0

                # Resize if needed
                if frame.shape[:2] != (self.target_h, self.target_w):
                    try:
                        frame = cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)
                    except Exception as e:
                        print(f"[WARN] Frame resize failed: {e}")
                        continue

                with self.lock:
                    self.latest_frame = frame
                    self.last_ok_time = time.time()
                    
            except Exception as e:
                print(f"[ERROR] Frame reader error: {e}")
                time.sleep(1.0)
                continue

    def read(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def age(self):
        return time.time() - self.last_ok_time

    def stop(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass

# ------------------------------ Fallback Detection ------------------------------
class FallbackDetector:
    """
    Simple fallback detection using OpenCV's built-in HOG descriptor
    when YOLO fails completely
    """
    def __init__(self):
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("[INFO] Fallback HOG detector initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize HOG detector: {e}")
            self.hog = None
        
        # Simple motion-based fallback
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.frame_count = 0
    
    def detect(self, frame):
        if self.hog is None:
            return []
        
        try:
            # Detect people using HOG with compatible parameters
            boxes, weights = self.hog.detectMultiScale(
                frame, 
                winStride=(8, 8),
                padding=(32, 32),
                scale=1.05,
                hitThreshold=0.0,
                useMeanshiftGrouping=False
            )
            
            # Debug information
            if len(boxes) > 0:
                print(f"[DEBUG] HOG detected {len(boxes)} people, weights type: {type(weights)}, length: {len(weights) if hasattr(weights, '__len__') else 'scalar'}")
            
            # Convert to YOLO-like format
            detections = []
            if len(boxes) > 0:
                for i, (x, y, w, h) in enumerate(boxes):
                    # Convert to center format and add confidence
                    cx = x + w // 2
                    cy = y + h // 2
                    
                    # Handle different weight formats
                    if len(weights) > i:
                        if hasattr(weights[i], '__len__') and len(weights[i]) > 0:
                            conf = min(float(weights[i][0]) / 2.0, 1.0)
                        else:
                            conf = min(float(weights[i]) / 2.0, 1.0)
                    else:
                        conf = 0.5  # Default confidence
                    
                    detections.append({
                        'bbox': (int(x), int(y), int(x + w), int(y + h)),
                        'confidence': conf,
                        'class': 0,  # person class
                        'id': i  # Simple ID assignment
                    })
            
            return detections
        except Exception as e:
            print(f"[ERROR] HOG detection failed: {e}")
            # Try motion-based detection as final fallback
            return self._detect_motion(frame)
    
    def _detect_motion(self, frame):
        """
        Simple motion-based detection as final fallback
        """
        try:
            self.frame_count += 1
            
            # Skip first few frames for background learning
            if self.frame_count < 10:
                return []
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for i, contour in enumerate(contours):
                # Filter by area
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum area for person
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio (person-like)
                    aspect_ratio = h / w if w > 0 else 0
                    if 1.2 < aspect_ratio < 4.0:  # Person-like aspect ratio
                        detections.append({
                            'bbox': (x, y, x + w, y + h),
                            'confidence': 0.6,  # Fixed confidence for motion detection
                            'class': 0,
                            'id': i
                        })
            
            if len(detections) > 0:
                print(f"[DEBUG] Motion detection found {len(detections)} moving objects")
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] Motion detection failed: {e}")
            return []

# ------------------------------ Server Attendance System ------------------------------
class ServerAttendanceSystem:
    """
    Server-optimized version with YOLO fallback
    """
    
    def __init__(
        self,
        camera_url: str,
        target_size=(640, 480),
        ui_fps: int = 8,             # Very conservative FPS
        imgsz: int = 320,            # Very small model size
        conf_thres: float = 0.40,    # Higher confidence
        iou_thres: float = 0.40,     # Higher IOU
        max_det: int = 10,           # Very few detections
        face_check_cooldown: float = 1.0,  # Shorter cooldown for better recognition
        min_face_area: int = 100     # Very small face area for better detection
    ):
        """
        Initialize with maximum compatibility settings
        """
        self.camera_url = camera_url
        self.target_w, self.target_h = target_size
        self.ui_frame_time = 1.0 / max(1, ui_fps)

        # Device - force CPU
        self.device = "cpu"
        print(f"[INFO] Forcing CPU device for maximum compatibility")

        # Initialize YOLO with aggressive error handling
        self.yolo = None
        self.yolo_working = False
        
        if YOLO_AVAILABLE:
            try:
                print("[INFO] Attempting to load YOLOv8 model with maximum compatibility...")
                
                # Try loading with minimal settings
                self.yolo = YOLO("best.pt")
                
                # Skip all optimizations
                print("[INFO] Skipping all YOLO optimizations for compatibility")
                
                # Force CPU
                self.yolo.to("cpu")
                
                # Test with a dummy image
                dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
                test_result = self.yolo.predict(
                    dummy_img,
                    device="cpu",
                    verbose=False,
                    conf=0.5,
                    imgsz=320,
                    max_det=1
                )
                
                self.yolo_working = True
                print("[INFO] ✓ YOLO model loaded and tested successfully")
                
            except Exception as e:
                print(f"[ERROR] YOLO failed to load: {e}")
                print("[INFO] Will use fallback detection method")
                self.yolo = None
                self.yolo_working = False
        else:
            print("[INFO] YOLO not available, using fallback detection")

        # Initialize fallback detector
        self.fallback_detector = FallbackDetector()

        # Detection settings
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        # Initialize InsightFace
        self.face_app = None
        if INSIGHTFACE_AVAILABLE:
            try:
                print("[INFO] Loading InsightFace with CPU-only settings...")
                self.face_app = app.FaceAnalysis(providers=['CPUExecutionProvider'])
                self.face_app.prepare(ctx_id=0, det_size=(320, 320))  # Smaller detection size
                print("[INFO] ✓ InsightFace loaded successfully")
            except Exception as e:
                print(f"[ERROR] InsightFace failed: {e}")
                self.face_app = None
        else:
            print("[INFO] InsightFace not available")

        self.min_face_area = int(min_face_area)

        # State management
        self.track_names = {}
        self.last_face_check = {}
        self.face_check_cooldown = float(face_check_cooldown)
        
        # Track persistence
        self.track_persistence = {}
        self.track_timeout = 5.0  # Shorter timeout for faster cleanup
        self.reid_threshold = 200

        # IN/OUT logic
        self.reference_line_y = self.target_h // 2
        self.person_states = {}
        self.persistent_states = {}
        self.last_logged = {}
        self.detected_positions = {}

        # CSV log file
        self.csv_file = "attendance_reversed.csv"
        self._init_csv()

        # Reader thread
        print("[INFO] Starting frame reader…")
        self.reader = FrameReader(camera_url, target_size=target_size)
        self.reader.start()

        # Performance tracking
        self.start_time = time.time()
        self.processed = 0
        self.yolo_failures = 0
        self.fallback_usage = 0
        self.last_name_clear = time.time()

        print(f"[INFO] Ready. device={self.device}, imgsz={self.imgsz}, UI target fps={ui_fps}")
        print(f"[INFO] YOLO working: {self.yolo_working}, Fallback available: {self.fallback_detector.hog is not None}")
        print(f"[INFO] REVERSED LOGIC: Above line = OUT, Below line = IN")
        print(f"[INFO] Reference line position: {self.reference_line_y} (frame height: {self.target_h})")

    def _init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                csv.writer(f).writerow(["Name", "ID", "Direction", "Timestamp", "Status"])

    def _detect_people(self, frame):
        """
        Detect people using YOLO or fallback method
        """
        if self.yolo_working and self.yolo is not None:
            try:
                # Try YOLO with maximum conservative settings
                results = self.yolo.predict(
                    frame,
                    device="cpu",
                    verbose=False,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    imgsz=self.imgsz,
                    max_det=self.max_det,
                    classes=[0],  # person class only
                    half=False,
                    agnostic_nms=False,
                    augment=False
                )
                
                # Convert results to our format
                detections = []
                for r in results:
                    if r.boxes is not None:
                        for i, box in enumerate(r.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            detections.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(conf),
                                'class': 0,
                                'id': i
                            })
                
                return detections
                
            except Exception as e:
                self.yolo_failures += 1
                print(f"[ERROR] YOLO detection failed (failure #{self.yolo_failures}): {e}")
                
                # If YOLO fails too many times, disable it
                if self.yolo_failures > 10:
                    print("[WARN] YOLO failed too many times, disabling YOLO")
                    self.yolo_working = False
                    self.yolo = None
                
                # Fall back to HOG detection
                return self._detect_people_fallback(frame)
        else:
            return self._detect_people_fallback(frame)

    def _detect_people_fallback(self, frame):
        """
        Fallback detection using HOG
        """
        self.fallback_usage += 1
        if self.fallback_usage % 100 == 0:
            print(f"[INFO] Using fallback detection (usage: {self.fallback_usage})")
        
        return self.fallback_detector.detect(frame)

    def _get_arcface_embedding(self, img):
        """Extract ArcFace embedding from face image"""
        if self.face_app is None:
            print("[DEBUG] Face app is None, cannot extract embedding")
            return None
            
        try:
            print(f"[DEBUG] Extracting embedding from image of size: {img.shape}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(img_rgb)
            
            print(f"[DEBUG] Found {len(faces)} faces for embedding extraction")
            
            if len(faces) > 0:
                emb = faces[0].embedding.astype("float32")
                norm = np.linalg.norm(emb)
                print(f"[DEBUG] Embedding norm: {norm}")
                
                if norm == 0 or np.isnan(norm):
                    print("[DEBUG] Invalid embedding norm")
                    return None
                emb = emb / norm
                print(f"[DEBUG] ✓ Embedding extracted successfully, shape: {emb.shape}")
                return emb
            else:
                print("[DEBUG] No faces found for embedding extraction")
            return None
        except Exception as e:
            print(f"[ERROR] ArcFace embedding failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _detect_face_in_person_crop(self, crop):
        """Detect face in person crop"""
        if self.face_app is None:
            print("[DEBUG] Face app is None, skipping face detection")
            return None, None
            
        try:
            if crop.size == 0:
                print("[DEBUG] Empty crop, skipping face detection")
                return None, None
            
            print(f"[DEBUG] Detecting face in crop of size: {crop.shape}")
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(img_rgb)
            
            print(f"[DEBUG] Found {len(faces)} faces in crop")
            
            if len(faces) > 0:
                face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                x, y, w, h = face.bbox.astype(int)
                
                print(f"[DEBUG] Largest face bbox: ({x}, {y}, {w}, {h})")
                
                x = max(0, x)
                y = max(0, y)
                w = min(w, crop.shape[1] - x)
                h = min(h, crop.shape[0] - y)
                
                face_area = w * h
                print(f"[DEBUG] Face area: {face_area}, minimum required: {self.min_face_area}")
                
                if w <= 0 or h <= 0 or face_area < self.min_face_area:
                    print(f"[DEBUG] Face too small: {face_area} < {self.min_face_area}")
                    return None, None
                
                face_crop = crop[y:y+h, x:x+w]
                if face_crop.size == 0:
                    print("[DEBUG] Empty face crop")
                    return None, None
                
                print(f"[DEBUG] ✓ Face detected successfully, size: {face_crop.shape}")
                return face_crop, (x, y, w, h)
            else:
                print("[DEBUG] No faces detected in crop")
            return None, None
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _recognize_person_if_needed(self, track_id, frame, x1, y1, x2, y2):
        """Recognize person with throttling"""
        if (x2 - x1) * (y2 - y1) < (30 * 30):
            return None

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        face_crop, face_bbox_rel = self._detect_face_in_person_crop(person_crop)
        if face_crop is None:
            return None

        fx, fy, fw, fh = face_bbox_rel
        face_bbox_frame = (x1 + fx, y1 + fy, fw, fh)

        now = time.time()
        last = self.last_face_check.get(track_id, 0.0)
        should_recognize = (now - last) >= self.face_check_cooldown

        # Force recognition every time - ignore cooldown for better accuracy
        if RECOGNIZE_AVAILABLE:
            self.last_face_check[track_id] = now
            
            print(f"[DEBUG] Extracting embedding for track {track_id}")
            embedding = self._get_arcface_embedding(face_crop)
            if embedding is not None:
                try:
                    print(f"[DEBUG] Searching database for track {track_id}")
                    # Try with very low threshold for better recognition
                    name, score = recognize_face(embedding, threshold=0.4)
                    print(f"[DEBUG] Recognition result for track {track_id}: {name} (score: {score})")
                    
                    if name and name != "Unknown":
                        self.track_names[track_id] = name
                        print(f"[INFO] ✓ Recognized {name} (ID:{track_id}) with score {score}")
                    else:
                        print(f"[DEBUG] No match found for track {track_id} (score: {score})")
                except Exception as e:
                    print(f"[ERROR] Recognition failed for track {track_id}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[DEBUG] No embedding extracted for track {track_id}")
        
        return face_bbox_frame

    def _update_track_persistence(self, current_tracks):
        """Update track persistence"""
        current_time = time.time()
        
        for tid, info in current_tracks.items():
            cx, cy = info["centroid"]
            
            velocity = (0, 0)
            if tid in self.track_persistence:
                prev_centroid = self.track_persistence[tid]["last_centroid"]
                time_diff = current_time - self.track_persistence[tid]["last_seen"]
                if time_diff > 0:
                    velocity = ((cx - prev_centroid[0]) / time_diff, (cy - prev_centroid[1]) / time_diff)
            
            self.track_persistence[tid] = {
                "last_seen": current_time,
                "last_centroid": info["centroid"],
                "name": info["name"],
                "last_bbox": info["bbox"],
                "velocity": velocity
            }
        
        # Clean up old tracks and their names
        for tid in list(self.track_persistence.keys()):
            if current_time - self.track_persistence[tid]["last_seen"] > self.track_timeout:
                print(f"[DEBUG] Cleaning up old track {tid} and its name")
                del self.track_persistence[tid]
                # Also clear the name so new person with same ID gets fresh recognition
                if tid in self.track_names:
                    del self.track_names[tid]

    def _centroid_state(self, y):
        """Return 'above' or 'below' relative to reference line"""
        return "above" if y < self.reference_line_y else "below"

    def _inout_direct_detection(self, track_id, cy):
        """REVERSED LOGIC: Direct detection approach"""
        current_state = self._centroid_state(cy)
        
        self.detected_positions[track_id] = current_state
        prev_state = self.persistent_states.get(track_id, self.person_states.get(track_id))
        
        self.person_states[track_id] = current_state
        self.persistent_states[track_id] = current_state
        
        if current_state == "above":
            if prev_state is None or prev_state != "above":
                print(f"[IN/OUT] Track {track_id}: OUT detected (above line)")
                return "OUT"
        elif current_state == "below":
            if prev_state is None or prev_state != "below":
                print(f"[IN/OUT] Track {track_id}: IN detected (below line)")
                return "IN"
        
        return None

    def _log_attendance(self, name, track_id, direction):
        """Log attendance to CSV and send Slack notification"""
        now = datetime.now()
        last = self.last_logged.get(track_id)
        if last and (now - last) < timedelta(seconds=60):
            print(f"[DEBUG] Skipping duplicate attendance for {name} (ID:{track_id}) - last logged: {last}")
            return

        status = "PRESENT" if direction == "IN" else "ABSENT"
        # 12-hour format with AM/PM (hours:minutes only)
        timestamp = now.strftime("%Y-%m-%d %I:%M %p")
        
        try:
            # Write to CSV
            print(f"[DEBUG] Writing to CSV: {self.csv_file}")
            with open(self.csv_file, "a", newline="") as f:
                csv.writer(f).writerow([name, track_id, direction, timestamp, status])
            self.last_logged[track_id] = now
            print(f"[ATTENDANCE] {name} (ID:{track_id}) - {direction} at {now.strftime('%I:%M %p')} - Status: {status}")
            
            # Send Slack notification
            send_slack_notification(name, direction, timestamp, status)
            
        except Exception as e:
            print(f"[ERROR] Failed to write to CSV: {e}")

    def _draw_overlay(self, frame, tracks):
        """Server version - minimal drawing, no display"""
        for tid, info in tracks.items():
            name = info.get("name", "Unknown")
            if name != "Unknown":
                x1, y1, x2, y2, conf = info["bbox"]
                cx, cy = info["centroid"]
                position = "above" if cy < self.reference_line_y else "below"
                print(f"[TRACK] {name} (ID:{tid}) at ({cx},{cy}) - {position} line")
    
    def run(self):
        """Main loop with YOLO fallback"""
        print("[INFO] Running server mode with YOLO fallback... Press Ctrl+C to quit")
        last_tick = 0.0
        frame_count = 0

        while True:
            try:
                if self.reader.age() > 10.0:
                    print("[WARN] RTSP stream stale (>10s) — camera/network issue?")

                frame = self.reader.read()
                if frame is None:
                    time.sleep(0.1)
                    continue

                now = time.time()
                if (now - last_tick) < self.ui_frame_time:
                    continue
                last_tick = now

                # Detect people using YOLO or fallback
                detections = self._detect_people(frame)

                # Convert detections to tracks format
                current_tracks = {}
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['confidence']
                    tid = det.get('id', i)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue

                    cx, cy = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))
                    
                    # Always start with Unknown name - force fresh recognition
                    name = "Unknown"

                    current_tracks[tid] = {
                        "bbox": (x1, y1, x2, y2, float(conf)),
                        "centroid": (cx, cy),
                        "name": name,
                        "face_bbox": None
                    }

                # Update track persistence
                self._update_track_persistence(current_tracks)
                
                # Clear track names periodically to force fresh recognition
                current_time = time.time()
                if current_time - self.last_name_clear > 5.0:  # Clear every 5 seconds
                    print("[DEBUG] Clearing all track names for fresh recognition")
                    self.track_names.clear()
                    self.last_name_clear = current_time
                
                # Face recognition - Force recognition for every detection
                for tid, info in current_tracks.items():
                    x1, y1, x2, y2, _ = info["bbox"]
                    
                    # Always attempt recognition - don't rely on track persistence
                    print(f"[DEBUG] Force attempting face recognition for track {tid}")
                    face_bbox = self._recognize_person_if_needed(tid, frame, x1, y1, x2, y2)
                    info["face_bbox"] = face_bbox
                    info["name"] = self.track_names.get(tid, "Unknown")
                    print(f"[DEBUG] Track {tid} name: {info['name']}")

                # IN/OUT logic
                for tid, info in current_tracks.items():
                    _, cy = info["centroid"]
                    direction = self._inout_direct_detection(tid, cy)
                    if direction:
                        # Only log attendance for recognized employees
                        name = info["name"]
                        if name != "Unknown":
                            print(f"[DEBUG] Logging attendance: {name} (ID:{tid}) - {direction}")
                            self._log_attendance(name, tid, direction)
                        else:
                            print(f"[DEBUG] Skipping attendance for unknown person (ID:{tid}) - {direction}")

                # Performance logging
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = frame_count / elapsed
                    print(f"[PERF] Processed {frame_count} frames in {elapsed:.1f}s (~{fps:.1f} FPS)")
                    print(f"[PERF] YOLO working: {self.yolo_working}, Fallback usage: {self.fallback_usage}")
                
                # Log tracking info
                self._draw_overlay(frame, current_tracks)

            except KeyboardInterrupt:
                print("[INFO] Stopping server...")
                break
            except Exception as e:
                print(f"[ERROR] Processing error: {e}")
                time.sleep(1.0)
                continue

        # Cleanup
        self.reader.stop()
        print("[INFO] Server stopped.")

def main():
    camera_url = "rtsp://USER:PASSWORD@IP:PORT"
    
    print("[INFO] Starting Server Attendance System with YOLO Fallback...")
    print(f"[INFO] Camera URL: {camera_url}")
    print("[INFO] Maximum compatibility mode with fallback detection")

    try:
        system = ServerAttendanceSystem(
            camera_url=camera_url,
            target_size=(640, 480),
            ui_fps=8,                 # Very conservative FPS
            imgsz=320,                # Very small model size
            conf_thres=0.40,          # Higher confidence
            iou_thres=0.40,           # Higher IOU
            max_det=10,               # Very few detections
            face_check_cooldown=1.0,  # Shorter cooldown for better recognition
            min_face_area=300         # Very small face area for better detection
        )
        system.run()
    except KeyboardInterrupt:
        print("[INFO] System stopped by user")
    except Exception as e:
        print(f"[ERROR] System failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
