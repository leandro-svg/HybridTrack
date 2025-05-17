import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional

from dataset.training_dataset_utils import read_calib, read_pose, cam_to_velo as common_cam_to_velo
from configs.config_utils import get_cfg

# Define a type alias for configuration objects for clarity
Config = Any  # Replace with a more specific type if available (e.g., CfgNode)


class KITTIDataset(Dataset):
    """
    Dataset class for KITTI tracking data.

    Handles loading annotations, creating sequences of 3D bounding boxes and poses,
    and applying transformations and noise for data augmentation.
    """

    def __init__(self, cfg: Config, mode: str = 'train'):
        """
        Initializes the KITTIDataset.

        Args:
            cfg: Configuration object (e.g., from YACS).
            mode: Dataset mode, one of ['train', 'validation', 'test'].
        """
        super(KITTIDataset, self).__init__()
        self.mode: str = mode
        self.cfg: Config = cfg

        self.root: str = ""
        self.kitti_root: str = ""
        self.kitti_annotations_gt: Dict[str, Any] = {}
        self.kitti_annotations_det: Dict[str, Any] = {}

        self.P2: Optional[np.ndarray] = None
        self.V2C: Optional[np.ndarray] = None

        self.len_seq: int = cfg.DATASET.SEQ_LEN
        self.stride: int = cfg.DATASET.SEQ_STRIDE

        self.kitti_sequences_annotations: List[np.ndarray] = []
        self.kitti_sequences_detection: List[np.ndarray] = []

        self.initialize_dataset()
        self.load_kitti_annotations()
        self.create_sequences()

    def initialize_dataset(self) -> None:
        """Initializes dataset paths based on the configuration."""
        self.root = self.cfg.DATASET.ROOT
        self.kitti_root = os.path.join(self.root, 'src',  'data')  # Used for annotation JSONs

    def load_kitti_annotations(self) -> None:
        """
        Loads KITTI annotations (ground truth and detections) from JSON files.
        The specific JSON file loaded depends on the dataset class (Car/Pedestrian)
        and mode (train/validation/test).
        """
        annotation_mode = 'validation' if self.mode == 'test' else self.mode
        json_filename = 'trajectories_ann.json'
        json_filepath = os.path.join(self.kitti_root, 'ann', annotation_mode, json_filename)

        if not os.path.exists(json_filepath):
            print(f"Error: Annotation file not found at {json_filepath}")
            self.kitti_annotations_gt = {}
            self.kitti_annotations_det = {}
            return

        try:
            with open(json_filepath, 'r') as f:
                annotations = json.load(f)
            self.kitti_annotations_gt = annotations
            self.kitti_annotations_det = annotations
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_filepath}")
            self.kitti_annotations_gt = {}
            self.kitti_annotations_det = {}
        except Exception as e:
            print(f"An unexpected error occurred while loading {json_filepath}: {e}")
            self.kitti_annotations_gt = {}
            self.kitti_annotations_det = {}

    def convert_bbs_type_numpy(self, boxes: np.ndarray, input_box_type: str) -> np.ndarray:
        """
        Converts bounding box formats. Specifically, for 'Kitti' type, it transforms
        (h, w, l, x, y, z, yaw) to (x, y, z, l, w, h, yaw_new).

        Args:
            boxes: NumPy array of bounding boxes. Shape (num_frames, num_features).
            input_box_type: The type of the input bounding box, e.g., "Kitti", "OpenPCDet".

        Returns:
            NumPy array of transformed bounding boxes.
        """
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)

        if input_box_type not in ["Kitti", "OpenPCDet", "Waymo"]:
            raise ValueError(f"Unsupported input box type: {input_box_type}")

        if input_box_type in ["OpenPCDet", "Waymo"]:
            return boxes

        if input_box_type == "Kitti":
            if boxes.shape[1] % 7 != 0:
                raise ValueError("For 'Kitti' type, number of columns must be a multiple of 7.")

            num_box_sets = boxes.shape[1] // 7
            new_boxes = np.copy(boxes)

            for i in range(num_box_sets):
                b_offset = i * 7
                h_orig = boxes[:, b_offset + 0]
                w_orig = boxes[:, b_offset + 1]
                l_orig = boxes[:, b_offset + 2]
                x_orig = boxes[:, b_offset + 3]
                y_orig = boxes[:, b_offset + 4]
                z_orig = boxes[:, b_offset + 5]
                yaw_orig = boxes[:, b_offset + 6]

                new_boxes[:, b_offset + 0] = x_orig
                new_boxes[:, b_offset + 1] = y_orig
                new_boxes[:, b_offset + 2] = z_orig + h_orig / 2
                new_boxes[:, b_offset + 3] = l_orig
                new_boxes[:, b_offset + 4] = w_orig
                new_boxes[:, b_offset + 5] = h_orig
                new_boxes[:, b_offset + 6] = (np.pi - yaw_orig) + (np.pi / 2)

            return new_boxes
        return boxes

    def get_registration_angle_numpy(self, pose_matrices: np.ndarray) -> np.ndarray:
        """
        Extracts the yaw rotation angle from a batch of 2D rotation matrices.

        Args:
            pose_matrices: NumPy array of pose matrices, shape (N, dim, dim).

        Returns:
            NumPy array of angles (yaw) in radians.
        """
        cos_theta = pose_matrices[:, 0, 0]
        sin_theta = pose_matrices[:, 1, 0]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_from_cos = np.arccos(cos_theta)
        angles = np.where(sin_theta >= 0, theta_from_cos, 2 * np.pi - theta_from_cos)
        return angles

    def register_bbs_numpy_initial(self, boxes: np.ndarray, poses: np.ndarray) -> np.ndarray:
        """
        Transforms bounding box coordinates and yaws to a global frame using given poses.

        Args:
            boxes: NumPy array of bounding boxes, shape (num_frames, num_features).
            poses: NumPy array of pose matrices, shape (num_frames, 4, 4).

        Returns:
            NumPy array of transformed bounding boxes in the world frame.
        """
        if poses is None:
            return boxes

        if boxes.shape[0] != poses.shape[0]:
            raise ValueError("Number of frames in boxes and poses must match.")

        transformed_boxes = np.copy(boxes)
        num_box_sets = boxes.shape[1] // 7
        world_sensor_yaws = self.get_registration_angle_numpy(poses)
        ones_col = np.ones((boxes.shape[0], 1))

        for i in range(num_box_sets):
            b_offset = i * 7
            box_xyz_sensor = boxes[:, b_offset: b_offset + 3]
            box_xyz1_sensor = np.concatenate([box_xyz_sensor, ones_col], axis=-1)
            box_xyz1_world = np.einsum('ij,ijk->ik', box_xyz1_sensor, poses)
            transformed_boxes[:, b_offset: b_offset + 3] = box_xyz1_world[:, :3]
            original_box_yaw_relative = boxes[:, b_offset + 6]
            transformed_boxes[:, b_offset + 6] = original_box_yaw_relative + world_sensor_yaws

        return transformed_boxes

    def add_size_dependent_noise_batch(self, bboxes: np.ndarray, image_width: int = 1242, image_height: int = 375) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adds noise to 2D bounding boxes based on their size and normalizes them.

        Args:
            bboxes: NumPy array of 2D bounding boxes, shape (N, 4).
            image_width: Width of the image for normalization.
            image_height: Height of the image for normalization.

        Returns:
            A tuple of (normalized_bboxes, noisy_normalized_bboxes).
        """
        top_lefts = bboxes[:, :2] - bboxes[:, 2:] / 2
        bottom_rights = bboxes[:, :2] + bboxes[:, 2:] / 2
        widths = bottom_rights[:, 0] - top_lefts[:, 0]
        heights = bottom_rights[:, 1] - top_lefts[:, 1]
        width_noise_scales = 0.0 * widths
        height_noise_scales = 0.0 * heights
        noise_x_min = np.random.normal(0, width_noise_scales)
        noise_y_min = np.random.normal(0, height_noise_scales)
        noise_x_max = np.random.normal(0, width_noise_scales)
        noise_y_max = np.random.normal(0, height_noise_scales)
        noisy_top_lefts = top_lefts + np.stack((noise_x_min, noise_y_min), axis=1)
        noisy_bottom_rights = bottom_rights + np.stack((noise_x_max, noise_y_max), axis=1)
        normalized_top_lefts = top_lefts / [image_width, image_height]
        normalized_bottom_rights = bottom_rights / [image_width, image_height]
        normalized_bboxes = np.hstack((normalized_top_lefts, normalized_bottom_rights))
        noisy_normalized_top_lefts = noisy_top_lefts / [image_width, image_height]
        noisy_normalized_bottom_rights = noisy_bottom_rights / [image_width, image_height]
        noisy_normalized_bboxes = np.hstack((noisy_normalized_top_lefts, noisy_normalized_bottom_rights))
        return normalized_bboxes, noisy_normalized_bboxes

    def add_noise_to_translation(self, translations: np.ndarray, scales: List[float]) -> np.ndarray:
        """
        Adds Gaussian noise to 3D translation vectors.

        Args:
            translations: NumPy array of translation vectors, shape (N, 3).
            scales: List or array of three floats representing the standard deviation
                    of noise for x, y, and z components, respectively.

        Returns:
            NumPy array of noisy translation vectors.
        """
        if len(scales) != 3:
            raise ValueError("Scales must be a list or array of three elements for x, y, z.")
        noise_x = np.random.normal(0, scales[0], translations.shape[0])
        noise_y = np.random.normal(0, scales[1], translations.shape[0])
        noise_z = np.random.normal(0, scales[2], translations.shape[0])
        noise = np.stack((noise_x, noise_y, noise_z), axis=1)
        noisy_translations = translations + noise
        return noisy_translations

    def add_noise_to_rotation(self, rotations: np.ndarray, scale: float = np.radians(5)) -> np.ndarray:
        """
        Adds Gaussian noise to rotation angles (e.g., yaw).

        Args:
            rotations: NumPy array of rotation angles (in radians).
            scale: Standard deviation of the noise to be added (in radians).

        Returns:
            NumPy array of noisy rotation angles.
        """
        noise = np.random.normal(0, scale, rotations.shape)
        noisy_rotations = rotations + noise
        return noisy_rotations

    def add_noise_to_3d_bboxes(self, bboxes_dims: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """
        Adds relative Gaussian noise to 3D bounding box dimensions (h, w, l).

        Args:
            bboxes_dims: NumPy array of bounding box dimensions, shape (N, 3).
            scale: Relative scale of the noise. Noise std dev will be scale * dimension.

        Returns:
            NumPy array of noisy bounding box dimensions.
        """
        relative_noise_factor = np.random.normal(0, scale, bboxes_dims.shape)
        noisy_bboxes_dims = bboxes_dims * (1 + relative_noise_factor)
        return noisy_bboxes_dims

    def create_sequences(self) -> None:
        """
        Creates sequences of ground truth annotations and detections.
        Processes loaded annotations, applies transformations, and optionally adds noise for detections.
        """
        if self.mode not in ['train', 'validation']:
            return

        if self.mode == 'validation':
            self.stride = self.len_seq

        kitti_raw_data_root = os.path.join(self.cfg.DATASET.ROOT, "src/data/KITTI/tracking/training")
        if not os.path.isdir(kitti_raw_data_root):
            print(f"Warning: KITTI raw data root not found: {kitti_raw_data_root}.")
            return

        for track_id_key, track_data in self.kitti_annotations_gt.items():
            seq_name = track_id_key[0:4]
            calib_file_path = os.path.join(kitti_raw_data_root, "calib", f"{seq_name}.txt")

            if not os.path.exists(calib_file_path):
                print(f"Warning: Calibration file not found for sequence {seq_name} at {calib_file_path}.")
                continue

            self.P2, self.V2C = read_calib(calib_file_path)
            if self.V2C is None:
                print(f"Warning: Failed to load V2C matrix for sequence {seq_name}.")
                continue

            num_frames_in_track = len(track_data["frame_id"])
            if num_frames_in_track >= self.len_seq:
                for i in range(0, num_frames_in_track - self.len_seq + 1, self.stride):
                    bb_3d_size_gt = np.array(track_data["bounding_box_3d_size"][i: i + self.len_seq])
                    pose_trans_gt = np.array(track_data["pose_translation"][i: i + self.len_seq])
                    pose_rot_gt_yaw = np.array(track_data["pose_rotation"][i: i + self.len_seq])
                    pose_matrices_gt = np.array(track_data["pose"][i: i + self.len_seq])

                    if pose_matrices_gt.shape != (self.len_seq, 4, 4):
                        print(f"Warning: Pose matrix shape mismatch for GT track {track_id_key}, slice {i}.")
                        continue

                    sequence_gt = np.concatenate((
                        bb_3d_size_gt,
                        pose_trans_gt,
                        np.expand_dims(pose_rot_gt_yaw, axis=1)
                    ), axis=1)

                    sequence_gt[:, 3:6] = common_cam_to_velo(sequence_gt[:, 3:6], self.V2C)
                    sequence_gt = self.convert_bbs_type_numpy(sequence_gt, "Kitti")
                    sequence_gt = self.register_bbs_numpy_initial(sequence_gt, pose_matrices_gt)

                    self.kitti_sequences_annotations.append(sequence_gt)
                    self.kitti_sequences_annotations.append(sequence_gt[::-1].copy())

        for track_id_key, track_data in self.kitti_annotations_det.items():
            seq_name = track_id_key[0:4]
            calib_file_path = os.path.join(kitti_raw_data_root, "calib", f"{seq_name}.txt")

            if not os.path.exists(calib_file_path):
                print(f"Warning: Calibration file not found for DET sequence {seq_name} at {calib_file_path}.")
                continue
            
            current_P2, current_V2C = read_calib(calib_file_path)
            if current_V2C is None:
                print(f"Warning: Failed to load V2C matrix for DET sequence {seq_name}.")
                continue

            num_frames_in_track = len(track_data["frame_id"])
            if num_frames_in_track >= self.len_seq:
                for i in range(0, num_frames_in_track - self.len_seq + 1, self.stride):
                    bb_3d_size_det = np.array(track_data["bounding_box_3d_size"][i: i + self.len_seq])
                    pose_trans_det = np.array(track_data["pose_translation"][i: i + self.len_seq])
                    pose_rot_det_yaw = np.array(track_data["pose_rotation"][i: i + self.len_seq])
                    pose_matrices_det = np.array(track_data["pose"][i: i + self.len_seq])

                    if pose_matrices_det.shape != (self.len_seq, 4, 4):
                        print(f"Warning: Pose matrix shape mismatch for DET track {track_id_key}, slice {i}.")
                        continue

                    # noisy_bb_3d_size_det = self.add_noise_to_3d_bboxes(bb_3d_size_det.copy(), scale=self.cfg.DATASET.NOISE.BOX_SIZE_SCALE)
                    # noisy_pose_trans_det = self.add_noise_to_translation(pose_trans_det.copy(), scales=self.cfg.DATASET.NOISE.TRANSLATION_SCALES)
                    # noisy_pose_rot_det_yaw = self.add_noise_to_rotation(pose_rot_det_yaw.copy(), scale=self.cfg.DATASET.NOISE.ROTATION_SCALE)

                    sequence_det = np.concatenate((
                        bb_3d_size_det,
                        pose_trans_det,
                        np.expand_dims(pose_rot_det_yaw, axis=1)
                    ), axis=1)

                    sequence_det[:, 3:6] = common_cam_to_velo(sequence_det[:, 3:6], current_V2C)
                    sequence_det = self.convert_bbs_type_numpy(sequence_det, "Kitti")
                    sequence_det = self.register_bbs_numpy_initial(sequence_det, pose_matrices_det)

                    self.kitti_sequences_detection.append(sequence_det)
                    self.kitti_sequences_detection.append(sequence_det[::-1].copy())

        if self.mode == 'train' and self.cfg.DATASET.RATIO_DATASET < 100.0:
            num_gt_sequences = len(self.kitti_sequences_annotations)
            num_det_sequences = len(self.kitti_sequences_detection)

            if num_gt_sequences != num_det_sequences:
                print(f"Warning: GT ({num_gt_sequences}) and DET ({num_det_sequences}) sequence counts differ.")
                return

            subset_size = int(num_gt_sequences * self.cfg.DATASET.RATIO_DATASET / 100.0)
            indices = random.sample(range(num_gt_sequences), subset_size)
            self.kitti_sequences_annotations = [self.kitti_sequences_annotations[i] for i in indices]
            self.kitti_sequences_detection = [self.kitti_sequences_detection[i] for i in indices]

    def __len__(self) -> int:
        """Returns the number of sequences in the dataset."""
        return len(self.kitti_sequences_annotations)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sequence pair (ground truth and detection) at the given index.

        Args:
            index: The index of the sequence pair.

        Returns:
            A tuple containing:
                - kitti_sequence_gt (torch.Tensor): Ground truth sequence.
                - kitti_sequence_det (torch.Tensor): Detection sequence.
        """
        kitti_sequence_gt = self.kitti_sequences_annotations[index]
        kitti_sequence_det = self.kitti_sequences_detection[index]
        return torch.from_numpy(kitti_sequence_gt).float(), torch.from_numpy(kitti_sequence_det).float()

