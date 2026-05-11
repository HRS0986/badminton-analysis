import cv2
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

def extract_movement_features(input_json_path, output_json_path, output_csv_path, output_video_path=None, original_video_path=None):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    FPS = data.get('fps', 30)
    FRAME_W = data.get('frame_width', 1920)
    FRAME_H = data.get('frame_height', 1080)
    frames_raw = data.get('frames', [])

    COURT_M_HEIGHT = 13.4
    COURT_M_WIDTH = 6.1
    PX_PER_M_Y = FRAME_H / COURT_M_HEIGHT
    PX_PER_M_X = FRAME_W / COURT_M_WIDTH

    CONF_THR = 0.25

    rows = []
    for fr in frames_raw:
        if not fr.get('player_detected', False):
            continue

        bb = fr['bounding_box']
        kps = {kp['name']: kp for kp in fr.get('keypoints', [])}

        def kp_xy(name):
            k = kps.get(name)
            if k and k['confidence'] >= CONF_THR:
                return k['x'], k['y']
            return np.nan, np.nan

        lax, lay = kp_xy('left_ankle')
        rax, ray = kp_xy('right_ankle')
        foot_x = np.nanmean([lax, rax])
        foot_y = np.nanmean([lay, ray])

        lhx, lhy = kp_xy('left_hip')
        rhx, rhy = kp_xy('right_hip')
        hip_x = np.nanmean([lhx, rhx])
        hip_y = np.nanmean([lhy, rhy])

        lwx, lwy = kp_xy('left_wrist')
        rwx, rwy = kp_xy('right_wrist')

        rows.append(dict(
            frame_id=fr['frame_id'],
            timestamp=fr['timestamp'],
            bb_x1=bb['x1'], bb_y1=bb['y1'], bb_x2=bb['x2'], bb_y2=bb['y2'],
            bb_w=bb['width'], bb_h=bb['height'],
            foot_x=foot_x, foot_y=foot_y,
            hip_x=hip_x,  hip_y=hip_y,
            lax=lax, lay=lay, rax=rax, ray=ray,
            lwx=lwx, lwy=lwy, rwx=rwx, rwy=rwy,
            lhx=lhx, lhy=lhy, rhx=rhx, rhy=rhy,
        ))

    df = pd.DataFrame(rows).sort_values('frame_id').reset_index(drop=True)
    if len(df) == 0:
        return {}

    WIN = max(3, int(FPS * 0.10))
    WIN = WIN if WIN % 2 == 1 else WIN + 1

    df['foot_x_sm'] = savgol_filter(df['foot_x'].ffill().bfill(), WIN, 2)
    df['foot_y_sm'] = savgol_filter(df['foot_y'].ffill().bfill(), WIN, 2)

    dx_px = df['foot_x_sm'].diff()
    dy_px = df['foot_y_sm'].diff()
    dt = df['timestamp'].diff().replace(0, np.nan)

    dx_m = dx_px / PX_PER_M_X
    dy_m = dy_px / PX_PER_M_Y

    df['speed_mps'] = np.sqrt(dx_m**2 + dy_m**2) / dt
    df['speed_mps'] = df['speed_mps'].clip(upper=15)

    speed_sm = savgol_filter(df['speed_mps'].fillna(0), WIN, 2)
    df['accel_mps2'] = pd.Series(speed_sm).diff() / dt
    df['accel_mps2'] = df['accel_mps2'].clip(-30, 30)

    def lateral_zone(x):
        if pd.isna(x): return 'Unknown'
        frac = x / FRAME_W
        if frac < 0.33: return 'Left'
        elif frac < 0.67: return 'Centre'
        else: return 'Right'

    def depth_zone(y):
        if pd.isna(y): return 'Unknown'
        frac = y / FRAME_H
        if frac < 0.33: return 'Net'
        elif frac < 0.67: return 'Mid'
        else: return 'Back'

    df['lateral_zone'] = df['foot_x'].apply(lateral_zone)
    df['depth_zone'] = df['foot_y'].apply(depth_zone)
    df['court_zone'] = df['lateral_zone'] + '-' + df['depth_zone']

    left_y = df['lay'].ffill().bfill().values
    right_y = df['ray'].ffill().bfill().values
    min_dist_frames = int(FPS * 0.15)
    left_peaks, _ = find_peaks(-left_y, distance=min_dist_frames, prominence=5)
    right_peaks, _ = find_peaks(-right_y, distance=min_dist_frames, prominence=5)
    total_peaks = len(left_peaks) + len(right_peaks)
    total_seconds = df['timestamp'].max() - df['timestamp'].min()
    stride_freq = total_peaks / total_seconds if total_seconds > 0 else 0

    df['stride_freq_hz'] = np.nan
    all_peaks = sorted(np.concatenate([left_peaks, right_peaks]))
    peak_times = df['timestamp'].iloc[all_peaks].values
    for i, row in df.iterrows():
        t = row['timestamp']
        count = np.sum((peak_times >= t - 0.5) & (peak_times <= t + 0.5))
        df.at[i, 'stride_freq_hz'] = count

    hip_y_series = df['hip_y'].ffill().bfill()
    baseline_window = max(3, int(FPS))
    hip_baseline = hip_y_series.rolling(baseline_window, center=True, min_periods=1).median()
    df['jump_height_px'] = (hip_baseline - hip_y_series).clip(lower=0)
    jump_peaks, jump_props = find_peaks(
        df['jump_height_px'],
        height=10,
        distance=int(FPS * 0.3),
        prominence=8
    )

    BURST_THRESHOLD = df['speed_mps'].quantile(0.75)
    POST_BURST_S = 0.5
    POST_FRAMES = int(POST_BURST_S * FPS)
    burst_frames, _ = find_peaks(
        df['speed_mps'].fillna(0),
        height=BURST_THRESHOLD,
        distance=int(FPS * 0.3)
    )
    recovery_speeds = []
    for bf in burst_frames:
        end_idx = min(bf + POST_FRAMES, len(df) - 1)
        window = df['speed_mps'].iloc[bf:end_idx]
        recovery_speeds.append(window.mean())
    df['is_burst'] = False
    df.loc[df.index[burst_frames], 'is_burst'] = True
    avg_recovery_speed = np.mean(recovery_speeds) if recovery_speeds else 0.0

    df['lat_reach_px'] = abs(df['rwx'] - df['lwx'])
    df['lat_reach_m'] = df['lat_reach_px'] / PX_PER_M_X
    df['hip_cx'] = (df['lhx'].fillna(df['rhx']) + df['rhx'].fillna(df['lhx'])) / 2
    df['left_reach_m'] = abs(df['lwx'] - df['hip_cx']) / PX_PER_M_X
    df['right_reach_m'] = abs(df['rwx'] - df['hip_cx']) / PX_PER_M_X
    df['max_arm_reach_m'] = df[['left_reach_m', 'right_reach_m']].max(axis=1)

    WINDOW_S = 2.0
    WINDOW_FR = max(3, int(WINDOW_S * FPS))
    foot_x = df['foot_x_sm'].values
    foot_y_sm = savgol_filter(df['foot_y'].ffill().bfill(), WIN, 2)
    mei_values = []
    for i in range(len(df)):
        start = max(0, i - WINDOW_FR)
        xs = foot_x[start:i+1]
        ys = foot_y_sm[start:i+1]
        dx_mei = np.diff(xs) / PX_PER_M_X
        dy_mei = np.diff(ys) / PX_PER_M_Y
        path_len = np.sum(np.sqrt(dx_mei**2 + dy_mei**2))
        net_disp = np.sqrt(((xs[-1] - xs[0]) / PX_PER_M_X)**2 + ((ys[-1] - ys[0]) / PX_PER_M_Y)**2)
        mei = net_disp / path_len if path_len > 0.01 else 1.0
        mei_values.append(min(mei, 1.0))
    df['movement_efficiency'] = mei_values

    output_cols = [
        'frame_id', 'timestamp',
        'speed_mps', 'accel_mps2',
        'court_zone', 'lateral_zone', 'depth_zone',
        'stride_freq_hz',
        'jump_height_px',
        'lat_reach_m', 'max_arm_reach_m',
        'movement_efficiency',
        'is_burst'
    ]
    df[output_cols].to_csv(output_csv_path, index=False)

    dx_m_all = df['foot_x_sm'].diff().fillna(0) / PX_PER_M_X
    dy_m_all = pd.Series(savgol_filter(df['foot_y'].ffill().bfill(), WIN, 2)).diff().fillna(0) / PX_PER_M_Y
    total_distance = float(np.sum(np.sqrt(dx_m_all**2 + dy_m_all**2)))

    metrics = {
        "total_distance_covered": round(total_distance, 2),
        "average_speed": round(float(df['speed_mps'].mean()), 3),
        "max_speed": round(float(df['speed_mps'].max()), 2),
        "movement_efficiency": round(float(df['movement_efficiency'].mean()), 3),
        "court_coverage_percentage": 0.0, # Will be replaced
        "jump_count": int(len(jump_peaks)),
        "average_recovery_time": 0.0, # Will be replaced
        "pose_stability_score": 0.0 # Will be replaced
    }
    
    # Calculate Court Coverage Percentage
    court_area = COURT_M_WIDTH * COURT_M_HEIGHT
    coverage_w_m = (df['foot_x_sm'].max() - df['foot_x_sm'].min()) / PX_PER_M_X
    coverage_h_m = (df['foot_y_sm'].max() - df['foot_y_sm'].min()) / PX_PER_M_Y
    coverage_area = coverage_w_m * coverage_h_m
    metrics["court_coverage_percentage"] = round((coverage_area / court_area) * 100, 1)

    # Calculate Average Recovery Time
    recovery_times = []
    avg_speed_val = df['speed_mps'].mean()
    for bf in burst_frames:
        for offset in range(1, int(FPS * 2)): # look ahead up to 2 seconds
            idx = min(bf + offset, len(df) - 1)
            if df['speed_mps'].iloc[idx] <= avg_speed_val:
                recovery_times.append(offset / FPS)
                break
    if recovery_times:
        metrics["average_recovery_time"] = round(float(np.mean(recovery_times)), 3)
        
    # Calculate Pose Stability Score (Average confidence of valid keypoints)
    stability = 0.0
    valid_kps = 0
    for fr in frames_raw:
        if fr.get('player_detected', False):
            for kp in fr.get('keypoints', []):
                stability += kp.get('confidence', 0)
                valid_kps += 1
    metrics["pose_stability_score"] = round(stability / valid_kps if valid_kps > 0 else 0, 3)

    if output_video_path and original_video_path:
        cap = cv2.VideoCapture(original_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

        path_pts = []
        df_index = 0
        df_l = len(df)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Simple heatmap overlay and path drawing
            if df_index < df_l:
                x = df.iloc[df_index]['foot_x_sm']
                y = df.iloc[df_index]['foot_y_sm']
                if not np.isnan(x) and not np.isnan(y):
                    path_pts.append((int(x), int(y)))
                df_index += 1
            
            for i in range(1, len(path_pts)):
                cv2.line(frame, path_pts[i-1], path_pts[i], (255, 0, 255), 2)
            
            if len(path_pts) > 0:
                cv2.circle(frame, path_pts[-1], 6, (0, 0, 255), -1)
                
            out.write(frame)
            
        cap.release()
        out.release()
        
    # Store full dict in a JSON output 
    with open(output_json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
