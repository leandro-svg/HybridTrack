"""
Utility functions for reading and processing data from the KITTI dataset.
"""
import os
import re
import numpy as np
import cv2
from typing import Tuple, List, Dict, Union, Optional

TransformMatrix = np.ndarray

def read_calib(calib_path: str) -> Tuple[Optional[TransformMatrix], Optional[TransformMatrix]]:
    """
    Reads calibration data from a KITTI calibration file.

    Args:
        calib_path: Path to the calibration text file.

    Returns:
        A tuple containing:
            - P2 (np.ndarray, optional): (3,4) Camera projection matrix (camera 3D to image 2D).
            - vtc_mat (np.ndarray, optional): (4,4) Velodyne Lidar to Camera 3D transformation matrix.
        Returns (None, None) if essential matrices are not found.
    """
    P2: Optional[TransformMatrix] = None
    vtc_mat: Optional[TransformMatrix] = None
    R0: Optional[TransformMatrix] = None

    try:
        with open(calib_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("P2:"):
                    P2 = np.array(line.split()[1:], dtype=np.float32).reshape((3, 4))
                elif line.startswith("Tr_velo_to_cam") or line.startswith("Tr_velo_cam"):
                    vtc_mat_3x4 = np.array(line.split()[1:], dtype=np.float32).reshape((3, 4))
                    vtc_mat = np.vstack([vtc_mat_3x4, [0, 0, 0, 1]])
                elif line.startswith("R0_rect") or line.startswith("R_rect"):
                    R0_3x3 = np.array(line.split()[1:], dtype=np.float32).reshape((3, 3))
                    R0 = np.eye(4, dtype=np.float32)
                    R0[:3, :3] = R0_3x3

    except FileNotFoundError:
        print(f"Error: Calibration file not found at {calib_path}")
        return None, None
    except Exception as e:
        print(f"Error reading calibration file {calib_path}: {e}")
        return None, None

    if P2 is None or vtc_mat is None:
        print(f"Warning: P2 or Tr_velo_to_cam not found in {calib_path}")
        return P2, vtc_mat

    if R0 is not None:
        vtc_mat = np.dot(R0, vtc_mat)
    
    return P2, vtc_mat


def read_velodyne(lidar_path: str, P: Optional[TransformMatrix], vtc_mat: Optional[TransformMatrix],
                  reduce_points: bool = True, max_image_row: int = 374, max_image_col: int = 1241) -> Optional[np.ndarray]:
    """
    Reads Lidar point cloud data, optionally filters points outside the camera view.

    Args:
        lidar_path: Path to the Lidar .bin file.
        P: (3,4) Camera projection matrix.
        vtc_mat: (4,4) Velodyne to Camera transformation matrix.
        reduce_points: If True, filters points to keep only those visible in the camera image.
        max_image_row: Maximum row index (height) of the image.
        max_image_col: Maximum column index (width) of the image.

    Returns:
        np.ndarray: (N, 4) Array of Lidar points (x, y, z, intensity).
                    Returns None if essential matrices are missing or an error occurs.
    """
    if P is None or vtc_mat is None:
        print("Error: Projection matrix P or Velodyne-to-Camera matrix vtc_mat is None.")
        return None
        
    try:
        lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
    except FileNotFoundError:
        print(f"Error: Lidar file not found at {lidar_path}")
        return None
    except Exception as e:
        print(f"Error reading Lidar file {lidar_path}: {e}")
        return None

    if not reduce_points:
        return lidar_points

    lidar_points = lidar_points[lidar_points[:, 0] > 0]
    points_h = np.hstack((lidar_points[:, :3], np.ones((lidar_points.shape[0], 1), dtype=np.float32)))
    points_cam = np.dot(points_h, vtc_mat.T)
    img_pts_h = np.dot(points_cam, P.T)
    img_pts_h[:, 0] /= img_pts_h[:, 2]
    img_pts_h[:, 1] /= img_pts_h[:, 2]

    x_coords = img_pts_h[:, 0]
    y_coords = img_pts_h[:, 1]
    depth_coords = points_cam[:, 2]

    mask = (x_coords >= 0) & (x_coords < max_image_col) & \
           (y_coords >= 0) & (y_coords < max_image_row) & \
           (depth_coords > 0)

    return lidar_points[mask]


def _transform_points(points: np.ndarray, transform_matrix: TransformMatrix) -> np.ndarray:
    """Helper function to apply a 4x4 transformation matrix to 3D points."""
    if points.shape[1] != 3:
        raise ValueError("Input points must be of shape (N, 3)")
    points_h = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
    transformed_points_h = np.dot(points_h, transform_matrix.T)
    return transformed_points_h[:, :3]

def cam_to_velo(points_cam: np.ndarray, vtc_mat: TransformMatrix) -> np.ndarray:
    """
    Converts 3D points from camera coordinates to Velodyne Lidar coordinates.

    Args:
        points_cam: (N, 3) Points in camera coordinates.
        vtc_mat: (4,4) Velodyne to Camera transformation matrix.

    Returns:
        np.ndarray: (N, 3) Points in Velodyne Lidar coordinates.
    """
    vtc_inv = np.linalg.inv(vtc_mat)
    return _transform_points(points_cam, vtc_inv)

def velo_to_cam(points_velo: np.ndarray, vtc_mat: TransformMatrix) -> np.ndarray:
    """
    Converts 3D points from Velodyne Lidar coordinates to camera coordinates.

    Args:
        points_velo: (N, 3) Points in Velodyne Lidar coordinates.
        vtc_mat: (4,4) Velodyne to Camera transformation matrix.

    Returns:
        np.ndarray: (N, 3) Points in camera coordinates.
    """
    return _transform_points(points_velo, vtc_mat)

def read_image(image_path: str) -> Optional[np.ndarray]:
    """
    Reads an image from the given path.

    Args:
        image_path: Path to the image file.

    Returns:
        np.ndarray: The image as a NumPy array, or None if reading fails.
    """
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Warning: Failed to decode image at {image_path}")
        return image
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

def read_generic_label_file(
    label_path: str,
    num_values: int,
    class_name_field_idx: int = 0,
    track_id_field_idx: Optional[int] = None,
    allowed_classes: Optional[List[str]] = None,
    skip_header_lines: int = 0
) -> Tuple[List[np.ndarray], List[str], List[Optional[int]]]:
    """
    Reads a generic KITTI-style label file.

    Args:
        label_path: Path to the label file.
        num_values: The number of floating-point values that constitute a detection/box entry.
        class_name_field_idx: Index in the split line where the class name is located.
        track_id_field_idx: Index for track_id. If None, track IDs are not read.
        allowed_classes: Optional list of class names to include. If None, all classes (except "DontCare") are included.
        skip_header_lines: Number of lines to skip at the beginning of the file.

    Returns:
        A tuple containing:
            - boxes (List[np.ndarray]): List of NumPy arrays, each representing a box's data.
            - names (List[str]): List of class names corresponding to each box.
            - track_ids (List[Optional[int]]): List of track IDs. Contains None if track_id_field_idx is None or ID is not parsable.
    """
    boxes: List[np.ndarray] = []
    names: List[str] = []
    track_ids: List[Optional[int]] = []

    try:
        with open(label_path, 'r') as f:
            for i, line_content in enumerate(f):
                if i < skip_header_lines:
                    continue
                line_parts = line_content.strip().split()
                if not line_parts:
                    continue

                current_class_name = line_parts[class_name_field_idx]
                if allowed_classes and current_class_name not in allowed_classes:
                    continue
                if current_class_name == "DontCare":
                    continue

                try:
                    box_data = np.array(line_parts[-num_values:], dtype=np.float32)
                except ValueError:
                    continue

                track_id: Optional[int] = None
                if track_id_field_idx is not None and len(line_parts) > track_id_field_idx:
                    try:
                        track_id = int(line_parts[track_id_field_idx])
                    except ValueError:
                        pass

                boxes.append(box_data)
                names.append(current_class_name)
                track_ids.append(track_id)
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_path}")
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        
    return boxes, names, track_ids

def read_detection_label(label_path: str) -> Tuple[List[np.ndarray], List[str]]:
    """Reads standard detection labels."""
    boxes, names, _ = read_generic_label_file(label_path, num_values=7, class_name_field_idx=0)
    return boxes, names

def read_detection_label_with_track(label_path: str, car_only: bool = False, pedestrian_only: bool = False) -> Tuple[List[np.ndarray], List[str], List[Optional[int]]]:
    """
    Reads detection labels that include track IDs.
    """
    num_values = 11
    class_idx = 2
    track_idx = 1
    
    allowed: Optional[List[str]] = None
    if car_only:
        allowed = ["Car", "Van"]
    elif pedestrian_only:
        allowed = ["Pedestrian", "Person"]

    return read_generic_label_file(label_path, num_values=num_values,
                                   class_name_field_idx=class_idx,
                                   track_id_field_idx=track_idx,
                                   allowed_classes=allowed)

def read_detection_only(label_path: str, target_classes: List[str] = ["Car", "Van"]) -> List[np.ndarray]:
    """
    Reads detection files, typically for specific classes.
    """
    boxes, _, _ = read_generic_label_file(label_path, num_values=12,
                                          class_name_field_idx=0,
                                          allowed_classes=target_classes)
    return boxes

def read_2d_det(detections_path: str, video_name: str) -> Optional[np.ndarray]:
    """
    Reads 2D detections from a .txt file (comma-separated).
    """
    file_path = os.path.join(detections_path, f"{video_name}.txt")
    try:
        dets_2d = np.loadtxt(file_path, delimiter=',')
        return dets_2d
    except FileNotFoundError:
        print(f"Error: 2D detection file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading 2D detection file {file_path}: {e}")
        return None

def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """
    Performs linear assignment (Hungarian algorithm) to find the minimum cost matching.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(row_ind, col_ind)))
    except ImportError:
        print("Error: scipy.optimize.linear_sum_assignment not found. Please install SciPy.")
        raise ImportError("SciPy is required for linear_assignment.")
    except Exception as e:
        print(f"Error during linear assignment: {e}")
        return np.array([])

def iou_2d(box_a: Union[List[float], np.ndarray], box_b: Union[List[float], np.ndarray]) -> float:
    """
    Computes the Intersection over Union (IoU) for two 2D bounding boxes.
    """
    box_a_int = [int(x) for x in box_a]
    box_b_int = [int(x) for x in box_b]

    x_a = max(box_a_int[0], box_b_int[0])
    y_a = max(box_a_int[1], box_b_int[1])
    x_b = min(box_a_int[2], box_b_int[2])
    y_b = min(box_a_int[3], box_b_int[3])

    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (box_a_int[2] - box_a_int[0] + 1) * (box_a_int[3] - box_a_int[1] + 1)
    box_b_area = (box_b_int[2] - box_b_int[0] + 1) * (box_b_int[3] - box_b_int[1] + 1)

    union_area = float(box_a_area + box_b_area - intersection_area)

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou

def read_tracking_label(label_path: str, target_object_types: List[str]) -> Tuple[Dict[int, List[List[float]]], Dict[int, List[str]]]:
    """
    Reads KITTI tracking label data and organizes it by frame ID.
    """
    frame_data: Dict[int, List[List[float]]] = {}
    frame_names: Dict[int, List[str]] = {}

    try:
        with open(label_path, 'r') as f:
            for line_content in f:
                parts = line_content.strip().split()
                if not parts or len(parts) < 17:
                    continue

                frame_id = int(parts[0])
                track_id = int(parts[1])
                class_name = parts[2]

                if class_name in target_object_types:
                    try:
                        obj_values = [float(p) for p in parts[10:17]]
                        obj_values.append(float(track_id))
                        
                        frame_data.setdefault(frame_id, []).append(obj_values)
                        frame_names.setdefault(frame_id, []).append(class_name)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Error: Tracking label file not found at {label_path}")
    except Exception as e:
        print(f"Error reading tracking label file {label_path}: {e}")
        
    return frame_data, frame_names

def read_pose(pose_file_path: str) -> Dict[int, TransformMatrix]:
    """
    Reads pose data from a file, where each line represents a 4x4 transformation matrix for a frame.
    """
    pose_per_frame: Dict[int, TransformMatrix] = {}
    try:
        with open(pose_file_path, 'r') as f:
            for frame_idx, line_content in enumerate(f):
                values = line_content.strip().split()
                if len(values) == 12:
                    pose_3x4 = np.array(values, dtype=np.float32).reshape((3, 4))
                    pose_4x4 = np.vstack([pose_3x4, [0, 0, 0, 1]])
                    pose_per_frame[frame_idx] = pose_4x4
                elif len(values) == 16:
                    pose_4x4 = np.array(values, dtype=np.float32).reshape((4, 4))
                    pose_per_frame[frame_idx] = pose_4x4
                else:
                    print(f"Warning: Skipping line {frame_idx+1} in pose file {pose_file_path} due to unexpected number of values: {len(values)}")
                    continue
    except FileNotFoundError:
        print(f"Error: Pose file not found at {pose_file_path}")
    except Exception as e:
        print(f"Error reading pose file {pose_file_path}: {e}")
        
    return pose_per_frame

if __name__ == '__main__':
    calib_file = 'path/to/your/kitti/calib/0000.txt'
    P2_matrix, vtc_transform = read_calib(calib_file)
    if P2_matrix is not None and vtc_transform is not None:
        print("P2 Matrix:\n", P2_matrix)
        print("Velodyne to Cam Matrix:\n", vtc_transform)

    lidar_file = 'path/to/your/kitti/velodyne/000000.bin'
    if P2_matrix is not None and vtc_transform is not None:
        lidar_data = read_velodyne(lidar_file, P2_matrix, vtc_transform)
        if lidar_data is not None:
            print(f"Read {lidar_data.shape[0]} Lidar points.")

    image_file = 'path/to/your/kitti/image_2/000000.png'
    image_data = read_image(image_file)
    if image_data is not None:
        print(f"Image shape: {image_data.shape}")

    generic_label_file = 'path/to/your/kitti/label_2/0000.txt'
    g_boxes, g_names, g_tracks = read_generic_label_file(generic_label_file, num_values=7, class_name_field_idx=0)
    if g_boxes:
        print(f"Generic Read: Found {len(g_boxes)} objects.")

    tracking_label_file = 'path/to/your/kitti/tracking_label_2/0000.txt'
    t_boxes, t_names, t_ids = read_detection_label_with_track(tracking_label_file)
    if t_boxes:
        print(f"Tracking Read (all): Found {len(t_boxes)} tracked objects.")
    t_boxes_cars, t_names_cars, t_ids_cars = read_detection_label_with_track(tracking_label_file, car_only=True)
    if t_boxes_cars:
        print(f"Tracking Read (cars only): Found {len(t_boxes_cars)} cars/vans.")

    detection_file_cars = 'path/to/your/kitti/detections/cars_0000.txt'
    det_boxes_cars = read_detection_only(detection_file_cars, target_classes=["Car", "Van"])
    if det_boxes_cars:
        print(f"Detections Read (Car/Van with score): Found {len(det_boxes_cars)} objects.")

    pose_data_file = 'path/to/your/kitti/poses/00.txt'
    poses = read_pose(pose_data_file)
    if poses:
        print(f"Read {len(poses)} poses. First pose:\n{poses.get(0)}")

    box1 = [50, 50, 150, 150]
    box2 = [100, 100, 200, 200]
    iou_val = iou_2d(box1, box2)
    print(f"IoU between {box1} and {box2}: {iou_val:.4f}")