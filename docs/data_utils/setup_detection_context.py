"""
Script to set up detection context sequences for KITTI tracking dataset.
Extracts nearby detected objects for each ego trajectory to provide multi-agent context.
"""
import os
import json
import numpy as np
import re
from utils import read_calib, read_pose, cam_to_velo, write_json, read_directory


def read_detection_frame(det_path, V2C, types=("Car", "Van")):
    """
    Reads all detections from a single frame file.

    Args:
        det_path: Path to detection txt file (e.g., 000000.txt)
        V2C: Velo-to-cam transformation matrix
        types: Iterable of object types to include

    Returns:
        detections: (N, 8) array with [x, y, z, l, w, h, ry, score]
    """
    if not os.path.exists(det_path):
        return np.zeros((0, 8), dtype=np.float32)

    detections = []
    with open(det_path) as f:
        for line in f.readlines():
            infos = re.split(' ', line.strip())
            if len(infos) < 16:
                continue
            class_name = infos[0]
            if class_name in types:
                # KITTI detection format: class, -1, -1, alpha, bbox2d(4), h, w, l, x, y, z, ry, score
                try:
                    h, w, l = float(infos[8]), float(infos[9]), float(infos[10])
                    x, y, z = float(infos[11]), float(infos[12]), float(infos[13])
                    ry = float(infos[14])
                    score = float(infos[15]) if len(infos) > 15 else 1.0

                    # Convert from camera to velo coordinates
                    xyz_cam = np.array([[x, y, z]], dtype=np.float32)
                    xyz_velo = cam_to_velo(xyz_cam, V2C)[0, :3]

                    # Store as [x, y, z, l, w, h, ry, score] in velo frame
                    detections.append([xyz_velo[0], xyz_velo[1], xyz_velo[2], l, w, h, ry, score])
                except (ValueError, IndexError):
                    continue

    return np.array(detections, dtype=np.float32) if detections else np.zeros((0, 8), dtype=np.float32)


def filter_nearby_objects(ego_position, detections, max_distance=50.0):
    """
    Filters detections to only include objects within max_distance of ego.

    Args:
        ego_position: (3,) array with [x, y, z] of ego vehicle
        detections: (N, 8) array of detections
        max_distance: Maximum distance in meters

    Returns:
        filtered_detections: (M, 8) array where M <= N
    """
    if detections.shape[0] == 0:
        return detections

    det_positions = detections[:, :2]  # (N, 2)
    ego_pos_2d = ego_position[:2]      # (2,)
    distances = np.linalg.norm(det_positions - ego_pos_2d, axis=1)
    mask = distances < max_distance
    return detections[mask]


def main():
    # Project root two levels up from this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    root = os.path.join(project_root, "src", "data", "KITTI", "tracking")
    detections_root = os.path.join(root, "detections", "virconv", "training")
    dataset = "training"
    splits = ["train", "validation"]

    split_video = {
        "train": ["0000", "0002", "0003", "0004", "0005", "0007", "0009", "0011"],
        "validation": ["0001", "0006", "0008", "0010", "0012", "0013", "0014", "0015", "0016", "0018", "0019"]
    }

    print(f"Project root: {project_root}")
    print(f"KITTI root: {root}")
    print(f"Detections root: {detections_root}")

    for split in splits:
        ann_ego_path = os.path.join(project_root, "src", "data", "ann", split, "trajectories_ann.json")
        if not os.path.exists(ann_ego_path):
            print(f"Error: Ego trajectory annotations not found at {ann_ego_path}")
            print("Please run setup_trajectory.py first!")
            continue

        with open(ann_ego_path, 'r') as f:
            ego_trajectories = json.load(f)

        detection_context = {}

        for seq_id in split_video[split]:
            print(f"Processing sequence {seq_id} for {split} split...")

            calib_path = os.path.join(root, dataset, "calib", f"{seq_id}.txt")
            P2, V2C = read_calib(calib_path)
            pose_path = os.path.join(root, dataset, "pose", seq_id, "pose.txt")
            poses = read_pose(pose_path)

            det_seq_folder = os.path.join(detections_root, seq_id)

            for traj_key, traj_data in ego_trajectories.items():
                if not traj_key.startswith(seq_id):
                    continue

                video_id = traj_data["video_id"]
                track_id = traj_data["track_id"]
                frame_ids = traj_data["frame_id"]
                ego_positions = np.array(traj_data["pose_translation"])  # (T, 3)

                context_per_frame = []

                for frame_idx, ego_pos in zip(frame_ids, ego_positions):
                    det_file = os.path.join(det_seq_folder, f"{int(frame_idx):06d}.txt")
                    frame_detections = read_detection_frame(det_file, V2C)
                    nearby_detections = filter_nearby_objects(ego_pos, frame_detections, max_distance=50.0)
                    context_per_frame.append(nearby_detections.tolist())

                detection_context[traj_key] = {
                    "video_id": video_id,
                    "track_id": track_id,
                    "frame_id": frame_ids,
                    "detections_per_frame": context_per_frame
                }

        det_ann_dir = os.path.join(project_root, "src", "data", "ann", split)
        os.makedirs(det_ann_dir, exist_ok=True)
        output_path = os.path.join(det_ann_dir, "detection_context.json")
        write_json(output_path, detection_context)
        print(f"Saved detection context to {output_path}")
        print(f"Total trajectories with context: {len(detection_context)}")


if __name__ == "__main__":
    main()
