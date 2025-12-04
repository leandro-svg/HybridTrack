"""
Script to set up history context sequences for KITTI tracking dataset.
Extracts motion histories of nearby agents (other vehicles) for each ego trajectory frame.
Each frame will have the last H waypoints of up to K nearest agents.
"""
import os
import json
import numpy as np
from collections import defaultdict
from utils import read_calib, read_pose, cam_to_velo, write_json


def read_tracking_labels_with_ids(label_path, target_classes=("Car", "Van")):
    """
    Reads KITTI tracking labels and organizes by track_id and frame_id.
    
    Args:
        label_path: Path to KITTI tracking label file (e.g., 0000.txt)
        target_classes: Tuple of object classes to include
        
    Returns:
        tracks_dict: {track_id: {frame_id: [x, y, z, l, w, h, ry], ...}, ...}
                     All positions in velodyne coordinates
    """
    tracks_dict = defaultdict(dict)
    
    if not os.path.exists(label_path):
        return tracks_dict
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            obj_type = parts[2]
            
            if obj_type not in target_classes:
                continue
            
            # KITTI tracking format: frame, id, type, truncated, occluded, alpha,
            # bbox_2d(4), h, w, l, x, y, z, ry, score
            try:
                h, w, l = float(parts[10]), float(parts[11]), float(parts[12])
                x, y, z = float(parts[13]), float(parts[14]), float(parts[15])
                ry = float(parts[16])
                
                # Store as [x, y, z, l, w, h, ry] (7 dims)
                # Note: x,y,z are in camera coords; will convert to velo below
                tracks_dict[track_id][frame_id] = [x, y, z, l, w, h, ry]
            except (ValueError, IndexError):
                continue
    
    return tracks_dict


def convert_tracks_to_velo(tracks_dict, V2C):
    """
    Convert all track positions from camera to velodyne coordinates.
    
    Args:
        tracks_dict: {track_id: {frame_id: [x,y,z,l,w,h,ry], ...}, ...}
        V2C: Velodyne to camera transformation matrix
        
    Returns:
        tracks_dict with positions converted to velo frame
    """
    for track_id in tracks_dict:
        for frame_id in tracks_dict[track_id]:
            state = tracks_dict[track_id][frame_id]
            xyz_cam = np.array([[state[0], state[1], state[2]]], dtype=np.float32)
            xyz_velo = cam_to_velo(xyz_cam, V2C)[0, :3]
            # Update with velo coordinates
            tracks_dict[track_id][frame_id] = [
                xyz_velo[0], xyz_velo[1], xyz_velo[2],  # x, y, z in velo
                state[3], state[4], state[5], state[6]   # l, w, h, ry unchanged
            ]
    return tracks_dict


def select_nearest_agents(ego_position, tracks_dict, current_frame, max_agents=32, max_distance=50.0):
    """
    Select up to K nearest agents to ego at the current frame.
    
    Args:
        ego_position: (3,) array [x, y, z] of ego in velo frame
        tracks_dict: {track_id: {frame_id: [x,y,z,l,w,h,ry], ...}, ...}
        current_frame: int, current frame index
        max_agents: Maximum number of agents to return (K)
        max_distance: Maximum distance threshold in meters
        
    Returns:
        selected_track_ids: List of track_ids sorted by distance (up to K)
    """
    candidates = []
    ego_pos_2d = ego_position[:2]
    
    for track_id, frames in tracks_dict.items():
        if current_frame not in frames:
            continue
        state = frames[current_frame]
        agent_pos_2d = np.array(state[:2])
        distance = np.linalg.norm(agent_pos_2d - ego_pos_2d)
        
        if distance < max_distance:
            candidates.append((track_id, distance))
    
    # Sort by distance and take top K
    candidates.sort(key=lambda x: x[1])
    selected_track_ids = [tid for tid, _ in candidates[:max_agents]]
    return selected_track_ids


def build_agent_history(tracks_dict, track_id, current_frame, history_len=5):
    """
    Build history sequence of last H states for a given agent.
    
    Args:
        tracks_dict: {track_id: {frame_id: [x,y,z,l,w,h,ry], ...}, ...}
        track_id: Agent track ID
        current_frame: Current frame index
        history_len: Number of past frames to include (H)
        
    Returns:
        history: (H, 7) array of states, padded with zeros if insufficient history
        valid_mask: (H,) boolean array, True for valid frames, False for padding
    """
    history = np.zeros((history_len, 7), dtype=np.float32)
    valid_mask = np.zeros(history_len, dtype=bool)
    
    if track_id not in tracks_dict:
        return history, valid_mask
    
    frames = tracks_dict[track_id]
    
    # Collect last H frames up to (and including) current_frame
    for i in range(history_len):
        frame_idx = current_frame - (history_len - 1 - i)
        if frame_idx >= 0 and frame_idx in frames:
            history[i] = frames[frame_idx]
            valid_mask[i] = True
    
    return history, valid_mask


def main():
    # Project root two levels up from this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    root = os.path.join(project_root, "src", "data", "KITTI", "tracking")
    dataset = "training"
    splits = ["train", "validation"]
    
    split_video = {
        "train": ["0000", "0002", "0003", "0004", "0005", "0007", "0009", "0011"],
        "validation": ["0001", "0006", "0008", "0010", "0012", "0013", "0014", "0015", "0016", "0018", "0019"]
    }
    
    # Configuration - reduced MAX_AGENTS to save memory (most frames have <10 nearby agents)
    HISTORY_LEN = 5
    MAX_AGENTS = 10  # Reduced from 32 - only keep 10 nearest agents
    MAX_DISTANCE = 50.0
    
    print(f"Project root: {project_root}")
    print(f"KITTI root: {root}")
    print(f"History length: {HISTORY_LEN}, Max agents: {MAX_AGENTS}, Max distance: {MAX_DISTANCE}m")
    
    for split in splits:
        ann_ego_path = os.path.join(project_root, "src", "data", "ann", split, "trajectories_ann.json")
        if not os.path.exists(ann_ego_path):
            print(f"Error: Ego trajectory annotations not found at {ann_ego_path}")
            print("Please run setup_trajectory.py first!")
            continue
        
        with open(ann_ego_path, 'r') as f:
            ego_trajectories = json.load(f)
        
        history_context = {}
        
        for seq_id in split_video[split]:
            print(f"Processing sequence {seq_id} for {split} split...")
            
            # Load calibration
            calib_path = os.path.join(root, dataset, "calib", f"{seq_id}.txt")
            P2, V2C = read_calib(calib_path)
            
            # Load all agent tracks from GT labels
            label_path = os.path.join(root, dataset, "label_02", f"{seq_id}.txt")
            tracks_dict = read_tracking_labels_with_ids(label_path)
            tracks_dict = convert_tracks_to_velo(tracks_dict, V2C)
            
            print(f"  Loaded {len(tracks_dict)} agent tracks from GT labels")
            
            # Process each ego trajectory in this sequence
            for traj_key, traj_data in ego_trajectories.items():
                if not traj_key.startswith(seq_id):
                    continue
                
                video_id = traj_data["video_id"]
                track_id = traj_data["track_id"]
                frame_ids = traj_data["frame_id"]
                ego_positions = np.array(traj_data["pose_translation"])  # (T, 3)
                
                # For each frame, build history context for nearby agents
                history_per_frame = []
                
                for frame_idx, ego_pos in zip(frame_ids, ego_positions):
                    # Select nearest agents at this frame
                    selected_agents = select_nearest_agents(
                        ego_pos, tracks_dict, int(frame_idx), 
                        max_agents=MAX_AGENTS, max_distance=MAX_DISTANCE
                    )
                    
                    # Build histories for each selected agent
                    frame_histories = []
                    frame_masks = []
                    
                    for agent_id in selected_agents:
                        history, valid_mask = build_agent_history(
                            tracks_dict, agent_id, int(frame_idx), history_len=HISTORY_LEN
                        )
                        frame_histories.append(history.tolist())
                        frame_masks.append(valid_mask.tolist())
                    
                    # DO NOT pad here - store only actual agents (sparse format)
                    # Padding will be done at training time to save memory
                    # num_agents tells us how many real agents there are
                    
                    # Store as variable-length lists (not padded to MAX_AGENTS)
                    history_per_frame.append({
                        "histories": frame_histories,      # (K_actual, H, 7) - K_actual <= MAX_AGENTS
                        "valid_masks": frame_masks,        # (K_actual, H)
                        "num_agents": len(frame_histories) # Actual number of agents
                    })
                
                # Store history context aligned with ego trajectory
                history_context[traj_key] = {
                    "video_id": video_id,
                    "track_id": track_id,
                    "frame_id": frame_ids,
                    "history_per_frame": history_per_frame
                }
        
        # Save history context
        hist_ann_dir = os.path.join(project_root, "src", "data", "ann", split)
        os.makedirs(hist_ann_dir, exist_ok=True)
        output_path = os.path.join(hist_ann_dir, "history_context.json")
        write_json(output_path, history_context)
        print(f"Saved history context to {output_path}")
        print(f"Total trajectories with history: {len(history_context)}")


if __name__ == "__main__":
    main()
