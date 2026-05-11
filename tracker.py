import cv2
import json
import csv
import numpy as np
from ultralytics import YOLO

# --- CONFIG ---
CONF_THRES          = 0.35
KEYPOINT_CONF_THRES = 0.25

MAX_MISSING_FRAMES  = 40
MAX_CENTER_DISTANCE = 200
MIN_MATCH_SCORE     = 0.25

COURT_Y_MIN = 0.28
COURT_Y_MAX = 0.62

MOTION_THRESHOLD = 0.04

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4)
]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def bbox_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def center_distance(boxA, boxB):
    return np.linalg.norm(bbox_center(boxA) - bbox_center(boxB))

def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def safe_crop(frame, box):
    h, w = frame.shape[:2]
    x1 = int(max(0, min(w - 1, box[0])))
    y1 = int(max(0, min(h - 1, box[1])))
    x2 = int(max(0, min(w - 1, box[2])))
    y2 = int(max(0, min(h - 1, box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def color_histogram(frame, box):
    crop = safe_crop(frame, box)
    if crop is None or crop.size == 0:
        return None
    crop = cv2.resize(crop, (64, 128))
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def histogram_similarity(histA, histB):
    if histA is None or histB is None:
        return 0.0
    return max(0.0, min(1.0, cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)))

def motion_score(frame, prev_frame, box):
    if prev_frame is None:
        return 1.0
    crop_curr = safe_crop(frame,      box)
    crop_prev = safe_crop(prev_frame, box)
    if crop_curr is None or crop_prev is None:
        return 0.0
    if crop_curr.shape != crop_prev.shape:
        crop_prev = cv2.resize(crop_prev, (crop_curr.shape[1], crop_curr.shape[0]))
    diff  = cv2.absdiff(crop_curr, crop_prev)
    gray  = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    moved = np.count_nonzero(gray > 25)
    return moved / (gray.size + 1e-6)

def is_inside_court(box, frame_h):
    foot_y = box[3] / frame_h
    return COURT_Y_MIN < foot_y < COURT_Y_MAX

def face_visibility_score(kpt_conf):
    score  = sum(2 for i in [0, 1, 2, 3, 4] if kpt_conf[i] > KEYPOINT_CONF_THRES)
    score += sum(1 for i in [5, 6]           if kpt_conf[i] > KEYPOINT_CONF_THRES)
    return score

def extract_detections(result, frame):
    detections = []
    if result.boxes is None or result.keypoints is None:
        return detections
    boxes     = result.boxes.xyxy.cpu().numpy()
    det_confs = result.boxes.conf.cpu().numpy()
    kpts_xy   = result.keypoints.xy.cpu().numpy()
    kpts_conf = result.keypoints.conf.cpu().numpy() \
                if result.keypoints.conf is not None \
                else np.ones((len(boxes), 17))
    for i in range(len(boxes)):
        if det_confs[i] < CONF_THRES:
            continue
        detections.append({
            "box":           boxes[i].tolist(),
            "det_conf":      float(det_confs[i]),
            "keypoints":     kpts_xy[i].tolist(),
            "keypoint_conf": kpts_conf[i].tolist(),
            "hist":          None,
        })
    return detections

def select_initial_target(detections, frame, prev_frame, frame_h):
    best_det, best_score = None, -1

    for det in detections:
        box      = det["box"]
        kpt_conf = np.array(det["keypoint_conf"])

        if not is_inside_court(box, frame_h):
            continue

        mv = motion_score(frame, prev_frame, box)
        if mv < MOTION_THRESHOLD:
            continue

        if det.get("hist") is None:
            det["hist"] = color_histogram(frame, box)

        score = (
            face_visibility_score(kpt_conf)                  * 4.0 +
            det["det_conf"]                                  * 2.0 +
            min(bbox_area(box) / (frame_h * frame_h), 1.0)  * 1.0 +
            min(mv / 0.20, 1.0)                              * 1.0
        )
        if score > best_score:
            best_score, best_det = score, det

    return best_det

def match_locked_target(detections, frame, prev_frame, locked_box, locked_hist, frame_h):
    best_det, best_score = None, -1

    for det in detections:
        box = det["box"]

        if not is_inside_court(box, frame_h):
            continue

        iou  = compute_iou(locked_box, box)
        dist = center_distance(locked_box, box)
        if dist > MAX_CENTER_DISTANCE and iou < 0.05:
            continue

        area_ratio  = bbox_area(box) / (bbox_area(locked_box) + 1e-6)
        mv          = motion_score(frame, prev_frame, box)

        if det.get("hist") is None:
            det["hist"] = color_histogram(frame, box)

        match_score = (
            iou                                              * 4.0 +
            max(0.0, 1.0 - dist / MAX_CENTER_DISTANCE)      * 2.0 +
            histogram_similarity(locked_hist, det["hist"])   * 3.0 +
            det["det_conf"]                                  * 1.0 +
            (1.0 - min(abs(1.0 - area_ratio), 1.0))         * 1.0
        ) / 11.0

        if match_score > best_score:
            best_score, best_det = match_score, det

    return (None, best_score) if best_score < MIN_MATCH_SCORE else (best_det, best_score)

def draw_pose(frame, kpts, kpt_conf):
    for p1, p2 in SKELETON:
        if kpt_conf[p1] > KEYPOINT_CONF_THRES and kpt_conf[p2] > KEYPOINT_CONF_THRES:
            cv2.line(frame,
                     (int(kpts[p1][0]), int(kpts[p1][1])),
                     (int(kpts[p2][0]), int(kpts[p2][1])),
                     (0, 255, 255), 2)
    for i, (x, y) in enumerate(kpts):
        if kpt_conf[i] > KEYPOINT_CONF_THRES:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

def run_tracking_inference(video_path, out_mp4, out_json, out_csv, model_path="yolov8n-pose.pt", custom_model_path="best.pt", progress_callback=None):
    baseline_model = YOLO(model_path)
    custom_model = YOLO(custom_model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracking_json = {
        "fps": fps,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "frames": []
    }
    
    csv_rows = []
    
    locked = False
    locked_box = None
    locked_hist = None
    missing_count = 0
    last_good_detection = None
    prev_frame = None
    pose_name = "Detecting..."
    shot_conf = 0.0
    
    total_conf = 0.0
    detected_frames = 0

    for frame_id in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps
        
        base_result = baseline_model(frame, conf=CONF_THRES, verbose=False)[0]
        detections = extract_detections(base_result, frame)

        selected = None
        match_score = None

        if not locked:
            if len(detections) > 0:
                selected = select_initial_target(detections, frame, prev_frame, frame_h)
                if selected is not None:
                    locked = True
                    locked_box = selected["box"]
                    locked_hist = selected["hist"]
                    last_good_detection = selected
                    missing_count = 0
        else:
            selected, match_score = match_locked_target(detections, frame, prev_frame, locked_box, locked_hist, frame_h)
            if selected is not None:
                locked_box = selected["box"]
                new_hist = selected["hist"]
                if locked_hist is not None and new_hist is not None:
                    locked_hist = cv2.addWeighted(locked_hist, 0.85, new_hist, 0.15, 0)
                last_good_detection = selected
                missing_count = 0
            else:
                missing_count += 1
                if missing_count <= MAX_MISSING_FRAMES:
                    selected = last_good_detection
                else:
                    selected = None
                    locked = False
                    locked_box = None
                    locked_hist = None
                    last_good_detection = None
                    missing_count = 0

        frame_data = {
            "frame_id": frame_id,
            "timestamp": round(timestamp, 4),
            "player_detected": False,
            "tracking_status": "missing",
            "match_score": match_score,
            "shot_classification": pose_name,
            "shot_confidence": shot_conf,
            "bounding_box": None,
            "detection_confidence": None,
            "keypoints": []
        }

        if selected is not None:
            # Optimize: Only run the classification model every 5 frames if we have a tracked player
            if frame_id % 5 == 0:
                custom_result = custom_model(frame, verbose=False)[0]
                if custom_result.boxes and len(custom_result.boxes) > 0:
                    pose_id = int(custom_result.boxes.cls[0].item())
                    pose_name = custom_result.names[pose_id]
                    shot_conf = float(custom_result.boxes.conf[0].item())

            box = selected["box"]
            kpts = np.array(selected["keypoints"])
            kpt_conf = np.array(selected["keypoint_conf"])
            x1, y1, x2, y2 = map(int, box)

            if missing_count == 0:
                status = "tracked"
            else:
                status = "temporarily_predicted"

            frame_data["player_detected"] = True
            frame_data["tracking_status"] = status
            frame_data["bounding_box"] = {
                "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                "width": float(box[2] - box[0]), "height": float(box[3] - box[1])
            }
            frame_data["detection_confidence"] = float(selected["det_conf"])
            
            total_conf += float(selected["det_conf"])
            detected_frames += 1

            for idx, name in enumerate(KEYPOINT_NAMES):
                score = float(kpt_conf[idx])
                kx, ky = float(kpts[idx][0]), float(kpts[idx][1])
                frame_data["keypoints"].append({
                    "id": idx, "name": name, "x": kx, "y": ky, "confidence": score
                })
                # Add to CSV row
                csv_rows.append([frame_id, 1, name, kx, ky, score])

        tracking_json["frames"].append(frame_data)
        prev_frame = frame.copy()
        
        if progress_callback:
            progress_callback(frame_id + 1, total_frames)

    cap.release()

    with open(out_json, "w") as f:
        json.dump(tracking_json, f, indent=4)
        
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "player_id", "keypoint", "x", "y", "confidence"])
        writer.writerows(csv_rows)
        
    avg_conf = (total_conf / max(detected_frames, 1)) * 100
    
    return {
        "player_tracked": detected_frames > 0,
        "keypoints_detected": 17,
        "average_confidence": round(avg_conf, 2)
    }

def render_pose_video(video_path, out_mp4, json_path):
    import json
    import numpy as np
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames_data = {f['frame_id']: f for f in data['frames']}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"avc1"), fps, (frame_w, frame_h))

    for frame_id in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        fd = frames_data.get(frame_id)
        if fd and fd.get("player_detected"):
            status = fd.get("tracking_status", "missing")
            box_color = (0, 255, 0) if status == "tracked" else (0, 165, 255)
            bb = fd["bounding_box"]
            if bb is not None:
                x1, y1, x2, y2 = int(bb["x1"]), int(bb["y1"]), int(bb["x2"]), int(bb["y2"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(frame, f"Target Player | {status}", (x1, max(30, y1 - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)
                pose_name = fd.get("shot_classification", "Detecting...")
                shot_conf = fd.get("shot_confidence", 0.0)
                cv2.putText(frame, f"Shot: {pose_name} ({shot_conf:.2f})", (x1, max(55, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # Draw pose
            kpts = np.zeros((17, 2))
            kpt_conf = np.zeros(17)
            for kp in fd.get("keypoints", []):
                idx = kp["id"]
                kpts[idx] = [kp["x"], kp["y"]]
                kpt_conf[idx] = kp["confidence"]
            draw_pose(frame, kpts, kpt_conf)

        out.write(frame)

    cap.release()
    out.release()
