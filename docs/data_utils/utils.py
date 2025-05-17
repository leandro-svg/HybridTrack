"""
Utility functions for reading calibration, lidar, detection, and tracking data.
"""
import json
import os
import cv2
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

# Calibration reading

def read_calib(calib_path):
    """
    Reads calibration file and returns camera and lidar transformation matrices.
    Args:
        calib_path (str): Path to calibration txt file.
    Returns:
        tuple: (P2, vtc_mat)
            P2: (3,4) Camera 3D to 2D projection matrix
            vtc_mat: (4,4) Lidar 3D to camera 3D transformation matrix
    """
    P2 = vtc_mat = R0 = None
    with open(calib_path) as f:
        for line in f.readlines():
            if line.startswith("P2"):
                P2 = np.array(re.split(" ", line.strip())[-12:], np.float32).reshape((3, 4))
            if line.startswith("Tr_velo_to_cam") or line.startswith("Tr_velo_cam"):
                vtc_mat = np.array(re.split(" ", line.strip())[-12:], np.float32).reshape((3, 4))
                vtc_mat = np.concatenate([vtc_mat, [[0, 0, 0, 1]]])
            if line.startswith("R0_rect") or line.startswith("R_rect"):
                R0 = np.array(re.split(" ", line.strip())[-9:], np.float32).reshape((3, 3))
                R0 = np.concatenate([R0, [[0], [0], [0]]], -1)
                R0 = np.concatenate([R0, [[0, 0, 0, 1]]])
    vtc_mat = np.matmul(R0, vtc_mat)
    return (P2, vtc_mat)

# Lidar reading and transformation

def read_velodyne(path, P, vtc_mat, IfReduce=True):
    """
    Reads lidar data and projects to image plane if IfReduce is True.
    Returns valid points in lidar coordinates.
    """
    max_row, max_col = 374, 1241
    lidar = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
    if not IfReduce:
        return lidar
    mask = lidar[:, 0] > 0
    lidar = lidar[mask]
    lidar_copy = np.zeros_like(lidar)
    lidar_copy[:, :] = lidar[:, :]
    lidar[:, 3] = 1
    lidar = np.matmul(lidar, vtc_mat.T)
    img_pts = np.matmul(lidar, P.T)
    velo_tocam = np.linalg.inv(vtc_mat)
    normal = velo_tocam[0:3, 0:4]
    lidar = np.matmul(lidar, normal.T)
    lidar_copy[:, 0:3] = lidar
    x, y = img_pts[:, 0] / img_pts[:, 2], img_pts[:, 1] / img_pts[:, 2]
    mask = (x >= 0) & (x < max_col) & (y >= 0) & (y < max_row)
    return lidar_copy[mask]

def cam_to_velo(cloud, vtc_mat):
    """
    Converts 3D camera coordinates to lidar coordinates.
    """
    mat = np.ones((cloud.shape[0], 4), dtype=np.float32)
    mat[:, 0:3] = cloud[:, 0:3]
    normal = np.linalg.inv(vtc_mat)[0:3, 0:4]
    transformed_mat = normal @ mat.T
    return np.array(transformed_mat.T, dtype=np.float32)

def velo_to_cam(cloud, vtc_mat):
    """
    Converts lidar coordinates to 3D camera coordinates.
    """
    mat = np.ones((cloud.shape[0], 4), dtype=np.float32)
    mat[:, 0:3] = cloud[:, 0:3]
    normal = vtc_mat[0:3, 0:4]
    transformed_mat = normal @ mat.T
    return np.array(transformed_mat.T, dtype=np.float32)

# Image reading

def read_image(path):
    """
    Reads an image from file.
    """
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

# Detection label reading

def read_detection_label(path):
    """
    Reads detection labels, returns boxes and names.
    """
    boxes, names = [], []
    with open(path) as f:
        for line in f:
            line = line.split()
            this_name = line[0]
            if this_name != "DontCare":
                boxes.append(np.array(line[-7:], np.float32))
                names.append(this_name)
    return np.array(boxes), np.array(names)

def read_detection_label_with_track(path):
    """
    Reads detection labels with track IDs.
    """
    boxes, names, track = [], [], []
    with open(path) as f:
        for line in f:
            line = line.split()
            this_name, this_track, class_name = line[0], line[1], line[2]
            if class_name != "DontCare":
                boxes.append(np.array(line[-11:], np.float32))
                names.append(this_name)
                track.append(this_track)
    return np.array(boxes), np.array(names), np.array(track)

def read_detection_label_with_track_car_only(path):
    """
    Reads detection labels with track IDs for Car/Van only.
    """
    boxes, names, track = [], [], []
    with open(path) as f:
        for line in f:
            line = line.split()
            this_name, this_track, class_name = line[0], line[1], line[2]
            if class_name in ("Car", "Van"):
                boxes.append(np.array(line[-11:], np.float32))
                names.append(this_name)
                track.append(this_track)
    return np.array(boxes), np.array(names), np.array(track)

def read_detection_label_with_track_pedestrian_only(path):
    """
    Reads detection labels with track IDs for Pedestrian/Person only.
    """
    boxes, names, track = [], [], []
    with open(path) as f:
        for line in f:
            line = line.split()
            this_name, this_track, class_name = line[0], line[1], line[2]
            if class_name in ("Pedestrian", "Person"):
                boxes.append(np.array(line[-11:], np.float32))
                names.append(this_name)
                track.append(this_track)
    return np.array(boxes), np.array(names), np.array(track)

def read_detection(path):
    """
    Reads detection boxes for Car/Van only.
    """
    boxes = []
    with open(path) as f:
        for line in f:
            line = line.split()
            class_name = line[0]
            if class_name in ("Car", "Van"):
                boxes.append(np.array(line[-12:], np.float32))
    return np.array(boxes)

def read_2d_det(path, video):
    """
    Reads 2D detections from file.
    """
    return np.loadtxt(os.path.join(path, f'{video}.txt'), delimiter=',')

# Assignment and IOU

def linear_assignment(cost_matrix):
    """
    Solves linear assignment problem (Hungarian algorithm).
    """
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou_2d(boxA, boxB):
    """
    Computes 2D IOU between two boxes.
    """
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Tracking label reading

def read_tracking_label(path, types):
    """
    Reads tracking labels for given types.
    """
    frame_dict, names_dict = {}, {}
    with open(path) as f:
        for line in f:
            line = line.split()
            this_name = line[2]
            frame_id = int(line[0])
            ob_id = int(line[1])
            if this_name in types:
                label = np.array(line[10:17], np.float32).tolist()
                label.append(ob_id)
                if frame_id in frame_dict:
                    frame_dict[frame_id].append(label)
                    names_dict[frame_id].append(this_name)
                else:
                    frame_dict[frame_id] = [label]
                    names_dict[frame_id] = [this_name]
    return frame_dict, names_dict

def read_pose(path):
    """
    Reads pose matrices from file.
    """
    pose_per_seq = {}
    with open(path) as f:
        for idx, PoseStr in enumerate(f.readlines()):
            pose = np.array(PoseStr.split(' '), dtype=np.float32).reshape((-1, 4))
            pose = np.concatenate([pose, [[0, 0, 0, 1]]])
            pose_per_seq[idx] = pose
    return pose_per_seq




def read_directory(directory):
    return os.listdir(directory)

def read_file(file):
    with open(file) as f:
        return f.read()

def read_json(file):
    with open(file) as f:
        return json.load(f)

def write_file(file, data):
    with open(file, 'w') as f:
        f.write(data)

def write_json(file, data):
    with open(file, 'w') as f:
        json.dump(data, f)

def read_pose(path):
    pose_per_seq = {}
    with open(path) as f:
        for id, PoseStr in enumerate(f.readlines()):
            pose = np.array(PoseStr.split(' '), dtype=np.float32).reshape((-1, 4))
            pose = np.concatenate([pose, [[0, 0, 0, 1]]])
            pose_per_seq[id] = pose
    return pose_per_seq
