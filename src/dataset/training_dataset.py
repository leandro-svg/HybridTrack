import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional

from dataset.training_dataset_utils import read_calib, read_pose, cam_to_velo as common_cam_to_velo
from configs.config_utils import get_cfg
import re

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
        self.kitti_history_context: Dict[str, Any] = {}

        self.P2: Optional[np.ndarray] = None
        self.V2C: Optional[np.ndarray] = None

        self.len_seq: int = cfg.DATASET.SEQ_LEN
        self.stride: int = cfg.DATASET.SEQ_STRIDE

        self.kitti_sequences_annotations: List[np.ndarray] = []
        self.kitti_sequences_detection: List[np.ndarray] = []
        self.kitti_sequences_history: List[Dict[str, Any]] = []  # History context per sequence
        self.kitti_sequences_det_context: List[Dict[str, Any]] = []  # Detection context per sequence

        self.initialize_dataset()
        self.load_kitti_annotations()
        self.load_history_context()
        self.load_detection_context()
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

    def load_history_context(self) -> None:
        """
        Loads history context from JSON file if available AND if USE_HISTORY is enabled.
        History context contains (K, H, 7) agent histories per ego frame.
        Skipped to save memory when USE_HISTORY is false.
        """
        # Check if history is enabled in config
        ctx_cfg = getattr(self.cfg, 'CONTEXT', None)
        use_history = ctx_cfg is not None and getattr(ctx_cfg, 'USE_HISTORY', False)
        
        if not use_history:
            print("Info: USE_HISTORY is disabled in config. Skipping history context loading to save memory.")
            self.kitti_history_context = {}
            return
        
        annotation_mode = 'validation' if self.mode == 'test' else self.mode
        history_json_path = os.path.join(self.kitti_root, 'ann', annotation_mode, 'history_context.json')
        
        if not os.path.exists(history_json_path):
            print(f"Info: History context file not found at {history_json_path}. Proceeding without history.")
            self.kitti_history_context = {}
            return
        
        try:
            with open(history_json_path, 'r') as f:
                self.kitti_history_context = json.load(f)
            print(f"Loaded history context for {len(self.kitti_history_context)} trajectories from {history_json_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {history_json_path}")
            self.kitti_history_context = {}
        except Exception as e:
            print(f"An unexpected error occurred while loading {history_json_path}: {e}")
            self.kitti_history_context = {}

    def corrupt_detection_context(self, det_context: np.ndarray, cfg) -> np.ndarray:
        """
        Corrupt detection context to simulate realistic tracking conditions.
        Adds dropout, position noise, size noise, rotation noise, and score variation.
        
        Args:
            det_context: (N, 8) or (N, 11) array of detection context
            cfg: Config object with TRAIN_NOISE parameters
        
        Returns:
            Corrupted context array
        """
        if det_context is None or len(det_context) == 0:
            return det_context
        
        # Get noise parameters from config
        train_noise = getattr(cfg, 'TRAIN_NOISE', None)
        if train_noise is None or not getattr(train_noise, 'ENABLE', False):
            return det_context  # No corruption if disabled
        
        dropout_prob = float(getattr(train_noise, 'DROPOUT_PROB', 0.3))
        pos_noise_std = float(getattr(train_noise, 'POS_NOISE_STD', 0.2))
        size_noise_std = float(getattr(train_noise, 'SIZE_NOISE_STD', 0.1))
        rot_noise_std = float(getattr(train_noise, 'ROT_NOISE_STD', 0.05))
        score_noise_std = float(getattr(train_noise, 'SCORE_NOISE_STD', 0.1))
        
        corrupted = det_context.copy()
        N = len(corrupted)
        
        # 1. Random dropout (simulates missed detections)
        dropout_mask = np.random.rand(N) > dropout_prob
        corrupted = corrupted[dropout_mask]
        
        if len(corrupted) == 0:
            return np.zeros((0, det_context.shape[1]))  # Preserve dimensionality
        
        # 2. Add position noise (simulates detection errors)
        pos_noise = np.random.randn(len(corrupted), 3) * pos_noise_std
        corrupted[:, :3] += pos_noise
        
        # 3. Add size noise
        size_noise = np.random.randn(len(corrupted), 3) * size_noise_std
        corrupted[:, 3:6] += size_noise
        corrupted[:, 3:6] = np.maximum(corrupted[:, 3:6], 0.1)  # Ensure positive sizes
        
        # 4. Add rotation noise
        rot_noise = np.random.randn(len(corrupted)) * rot_noise_std
        corrupted[:, 6] += rot_noise
        
        # 5. Corrupt scores (simulates confidence variation)
        corrupted[:, 7] *= (1 + np.random.randn(len(corrupted)) * score_noise_std)
        corrupted[:, 7] = np.clip(corrupted[:, 7], 0.0, 1.0)
        
        # 6. If velocity features exist (11-dim), add velocity noise
        if corrupted.shape[1] == 11:
            vel_noise_std = pos_noise_std * 2  # Velocity noise proportional to position
            vel_noise = np.random.randn(len(corrupted), 3) * vel_noise_std
            corrupted[:, 8:11] += vel_noise
        
        return corrupted
    
    def load_detection_context(self) -> None:
        """
        Loads detection context from JSON file if available.
        Detection context contains per-frame nearby detections (N, 8) for each ego frame.
        Format: [x, y, z, l, w, h, ry, score] in velodyne coordinates.
        """
        annotation_mode = 'validation' if self.mode == 'test' else self.mode
        det_context_json_path = os.path.join(self.kitti_root, 'ann', annotation_mode, 'detection_context.json')
        
        if not os.path.exists(det_context_json_path):
            print(f"Info: Detection context file not found at {det_context_json_path}. Proceeding without detection context.")
            self.kitti_detection_context = {}
            return
        
        try:
            with open(det_context_json_path, 'r') as f:
                self.kitti_detection_context = json.load(f)
            print(f"Loaded detection context for {len(self.kitti_detection_context)} trajectories from {det_context_json_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {det_context_json_path}")
            self.kitti_detection_context = {}
        except Exception as e:
            print(f"An unexpected error occurred while loading {det_context_json_path}: {e}")
            self.kitti_detection_context = {}

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

                    # Extract ego positions in world frame (registered coordinates)
                    # sequence_gt after registration has format: [l, w, h, x_world, y_world, z_world, ry_world]
                    ego_positions_world = sequence_gt[:, 3:6]  # (seq_len, 3) [x, y, z] in world frame

                    # Extract corresponding history context if available
                    history_seq = self._extract_history_sequence(track_id_key, i, self.len_seq)
                    history_seq_rev = self._reverse_history_sequence(history_seq)
                    
                    # Extract detection context with ego-relative coordinates
                    det_ctx_seq = self._extract_detection_context_sequence(track_id_key, i, self.len_seq, 
                                                                          pose_matrices_gt, ego_positions_world)
                    
                    # Apply corruption to match inference conditions (CRITICAL FIX)
                    if self.mode == 'train' and hasattr(self.cfg, 'CONTEXT') and hasattr(self.cfg.CONTEXT, 'TRAIN_NOISE'):
                        corrupted_frames = []
                        for frame_dets in det_ctx_seq.get('detections_per_frame', []):
                            corrupted = self.corrupt_detection_context(frame_dets, self.cfg.CONTEXT)
                            corrupted_frames.append(corrupted)
                        det_ctx_seq = {'detections_per_frame': corrupted_frames}
                    
                    det_ctx_seq_rev = self._reverse_detection_context_sequence(det_ctx_seq)

                    self.kitti_sequences_annotations.append(sequence_gt)
                    self.kitti_sequences_history.append(history_seq)
                    self.kitti_sequences_det_context.append(det_ctx_seq)
                    
                    self.kitti_sequences_annotations.append(sequence_gt[::-1].copy())
                    self.kitti_sequences_history.append(history_seq_rev)
                    self.kitti_sequences_det_context.append(det_ctx_seq_rev)

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

                    # Note: History already added during GT processing; DET uses same indices
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

    def _extract_history_sequence(self, track_id_key: str, start_idx: int, seq_len: int) -> Dict[str, Any]:
        """
        Extracts history context for a given sequence slice.
        
        Args:
            track_id_key: Trajectory key in history_context dict
            start_idx: Starting frame index in trajectory
            seq_len: Sequence length
            
        Returns:
            Dictionary with 'histories' and 'valid_masks' lists per frame, or empty dict if unavailable
        """
        if track_id_key not in self.kitti_history_context:
            return {}
        
        history_data = self.kitti_history_context[track_id_key]
        history_per_frame = history_data.get("history_per_frame", [])
        
        if start_idx + seq_len > len(history_per_frame):
            return {}
        
        # Extract slice
        sliced_history = history_per_frame[start_idx: start_idx + seq_len]
        return {"history_per_frame": sliced_history}
    
    def _reverse_history_sequence(self, history_seq: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reverses a history sequence to match reversed trajectory.
        
        Args:
            history_seq: History sequence dictionary
            
        Returns:
            Reversed history sequence
        """
        if not history_seq or "history_per_frame" not in history_seq:
            return {}
        
        reversed_frames = history_seq["history_per_frame"][::-1]
        return {"history_per_frame": reversed_frames}

    def _extract_detection_context_sequence(self, track_id_key: str, start_idx: int, seq_len: int, 
                                            pose_matrices: np.ndarray, ego_positions: np.ndarray) -> Dict[str, Any]:
        """
        Extracts detection context for a given sequence slice.
        Transforms detections to EGO-RELATIVE coordinates (relative to ego vehicle at each frame).
        
        Args:
            track_id_key: Trajectory key in detection_context dict
            start_idx: Starting frame index in trajectory
            seq_len: Sequence length
            pose_matrices: (seq_len, 4, 4) pose matrices for coordinate registration
            ego_positions: (seq_len, 3) ego vehicle positions in world frame [x, y, z]
            
        Returns:
            Dictionary with 'detections_per_frame' list of (N, 8) arrays in ego-relative coords
        """
        if not hasattr(self, 'kitti_detection_context') or track_id_key not in self.kitti_detection_context:
            return {}
        
        det_data = self.kitti_detection_context[track_id_key]
        detections_per_frame = det_data.get("detections_per_frame", [])
        
        if start_idx + seq_len > len(detections_per_frame):
            return {}
        
        # Extract slice and convert to EGO-RELATIVE coordinates
        sliced_detections = []
        for frame_idx in range(start_idx, start_idx + seq_len):
            frame_dets = np.array(detections_per_frame[frame_idx], dtype=np.float32)
            pose_idx = frame_idx - start_idx
            
            if frame_dets.size > 0 and frame_dets.shape[0] > 0:
                # frame_dets: (N, 8) with [x,y,z,l,w,h,ry,score] in velo coords
                # First transform to world coordinates using pose matrix
                pose = pose_matrices[pose_idx]  # (4, 4)
                
                xyz_velo = frame_dets[:, :3]  # (N, 3)
                ones = np.ones((xyz_velo.shape[0], 1))
                xyz_homo = np.concatenate([xyz_velo, ones], axis=1)  # (N, 4)
                xyz_world = (pose @ xyz_homo.T).T[:, :3]  # (N, 3)
                
                # Get world sensor yaw for ry adjustment
                cos_theta = pose[0, 0]
                sin_theta = pose[1, 0]
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta_from_cos = np.arccos(cos_theta)
                world_yaw = theta_from_cos if sin_theta >= 0 else 2 * np.pi - theta_from_cos
                ry_world = frame_dets[:, 6] + world_yaw
                
                # Now compute EGO-RELATIVE positions
                ego_pos = ego_positions[pose_idx]  # (3,) ego position in world frame
                xyz_relative = xyz_world - ego_pos[np.newaxis, :]  # (N, 3) relative to ego
                
                # Ego-relative detection: [dx, dy, dz, l, w, h, ry_world, score]
                # dx,dy,dz are relative positions, ry is in world frame, dims are unchanged
                relative_dets = frame_dets.copy()
                relative_dets[:, :3] = xyz_relative  # Use relative positions
                relative_dets[:, 3:6] = frame_dets[:, 3:6]  # Keep dimensions l,w,h
                relative_dets[:, 6] = ry_world  # World-frame rotation
                relative_dets[:, 7] = frame_dets[:, 7]  # Keep score
                sliced_detections.append(relative_dets)
            else:
                sliced_detections.append(np.zeros((0, 8), dtype=np.float32))
        
        return {"detections_per_frame": sliced_detections}
    
    def _reverse_detection_context_sequence(self, det_ctx_seq: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reverses a detection context sequence to match reversed trajectory.
        
        Args:
            det_ctx_seq: Detection context dictionary
            
        Returns:
            Reversed detection context sequence
        """
        if not det_ctx_seq or "detections_per_frame" not in det_ctx_seq:
            return {}
        
        reversed_frames = det_ctx_seq["detections_per_frame"][::-1]
        return {"detections_per_frame": reversed_frames}

    def __len__(self) -> int:
        """Returns the number of sequences in the dataset."""
        return len(self.kitti_sequences_annotations)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Retrieves a sequence pair (ground truth, detection, and context) at the given index.

        Args:
            index: The index of the sequence pair.

        Returns:
            A tuple containing:
                - kitti_sequence_gt (torch.Tensor): Ground truth sequence.
                - kitti_sequence_det (torch.Tensor): Detection sequence.
                - context_dict (dict): Dictionary containing:
                    - 'history_context': History context dict with 'history_per_frame' if available
                    - 'detection_context': Detection context dict with 'detections_per_frame' if available
        """
        kitti_sequence_gt = self.kitti_sequences_annotations[index]
        kitti_sequence_det = self.kitti_sequences_detection[index]
        history_context = self.kitti_sequences_history[index] if index < len(self.kitti_sequences_history) else None
        detection_context = self.kitti_sequences_det_context[index] if index < len(self.kitti_sequences_det_context) else None
        
        # Combine both contexts into a single dict for easier handling
        context_dict = {
            'history_context': history_context,
            'detection_context': detection_context
        }
        
        return (
            torch.from_numpy(kitti_sequence_gt).float(), 
            torch.from_numpy(kitti_sequence_det).float(),
            context_dict
        )

