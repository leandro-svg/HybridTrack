"""
Script to set up trajectory annotations for KITTI tracking dataset.
"""
import os
import json
import numpy as np
from utils import read_detection_label_with_track_car_only
from utils import *

def main():
    root = "src/data/KITTI/tracking"
    dataset = "training"
    splits = ["train", "validation"]
    split_video = {
        "train": ["0000.txt", "0002.txt", "0003.txt", "0004.txt", "0005.txt", "0007.txt", "0009.txt", "0011.txt"],
        "validation": ["0001.txt", "0006.txt", "0008.txt", "0010.txt", "0012.txt", "0013.txt", "0014.txt", "0015.txt", "0016.txt", "0018.txt", "0019.txt"]
    }
    for split in splits:
        ann_dir = os.path.join("src/data/ann", split)
        label_folder = os.path.join(root, dataset, "label_02")
        label_files = read_directory(label_folder)
        ann = {}
        for file in label_files:
            if file in split_video[split]:
                label_path = os.path.join(label_folder, file)
                pose_path = os.path.join(root, dataset, "pose", file[0:4], "pose.txt")
                pose = read_pose(pose_path)
                labels, label_names, label_track = read_detection_label_with_track_car_only(label_path)
                if labels.shape[0] > 0:
                    bounding_box = labels[:, 0:4]
                    bounding_box_3d_size = labels[:, 4:7]
                    pose_translation = labels[:, 7:10]
                    pose_rotation = labels[:, 10]
                    for index, elem in enumerate(label_track):
                        label_key = f"{file[0:4]}_{elem}"
                        if label_key not in ann:
                            ann[label_key] = {
                                "video_id": file[0:4],
                                "frame_id": np.array([label_names[index]]),
                                "track_id": elem,
                                "bounding_box": np.array([bounding_box[index]]),
                                "bounding_box_3d_size": np.array([bounding_box_3d_size[index]]),
                                "pose_translation": np.array([pose_translation[index]]),
                                "pose_rotation": np.array([pose_rotation[index]]),
                                "pose": np.array([pose[int(label_names[index])]])
                            }
                        else:
                            ann[label_key]["frame_id"] = np.concatenate((ann[label_key]["frame_id"], np.array([label_names[index]])))
                            ann[label_key]["bounding_box"] = np.concatenate((ann[label_key]["bounding_box"], np.array([bounding_box[index]])))
                            ann[label_key]["bounding_box_3d_size"] = np.concatenate((ann[label_key]["bounding_box_3d_size"], np.array([bounding_box_3d_size[index]])))
                            ann[label_key]["pose_translation"] = np.concatenate((ann[label_key]["pose_translation"], np.array([pose_translation[index]])))
                            ann[label_key]["pose_rotation"] = np.concatenate((ann[label_key]["pose_rotation"], np.array([pose_rotation[index]])))
                            ann[label_key]["pose"] = np.concatenate((ann[label_key]["pose"], np.array([pose[int(label_names[index])]])))
        for key in ann:
            ann[key]["frame_id"] = ann[key]["frame_id"].tolist()
            ann[key]["bounding_box"] = ann[key]["bounding_box"].tolist()
            ann[key]["bounding_box_3d_size"] = ann[key]["bounding_box_3d_size"].tolist()
            ann[key]["pose_translation"] = ann[key]["pose_translation"].tolist()
            ann[key]["pose_rotation"] = ann[key]["pose_rotation"].tolist()
            ann[key]["pose"] = ann[key]["pose"].tolist()
        if not os.path.exists(ann_dir):
            os.makedirs(ann_dir)
        write_json(os.path.join(ann_dir, "trajectories_ann.json"), ann)

if __name__ == "__main__":
    main()
