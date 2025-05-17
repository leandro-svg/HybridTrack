import os
import sys
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from tools.batch_generation import SystemModel
from dataset.tracking_dataset_utils import read_calib
from tracker.obectPath import ObjectPath
from .box_op_2d import convert_bbs_type_initial, register_bbs_initial
from .cIoU import ciou_3d
from model.LearnableKF import LEARNABLEKF
from model.model_parameters import m, n, f, hRotate

class HYBRIDTRACK:
    def __init__(self, tracking_features: bool = False, bb_as_features: bool = False, box_type: str = 'Kitti', config=None):
        """
        Initialize the 3D tracker.
        Args:
            tracking_features: Track features if True.
            bb_as_features: Use bounding boxes as features if True.
            box_type: Box type ("OpenPCDet", "Kitti", "Waymo").
            config: Configuration object.
        """
        self.config = config
        self.current_timestamp: Optional[int] = None
        self.current_pose = None
        self.current_bbs = None
        self.current_features = None
        self.current_scores = None
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features
        self.box_type = box_type
        self.track_dim = 7
        self.label_seed = 0
        self.batch_size = 5000
        self.active_trajectories: Dict = {}
        self.dead_trajectories: Dict = {}
        self.memory_pool: Dict = {}
        self.device = torch.device(self.config.DEVICE) if self.config and hasattr(self.config, 'DEVICE') else torch.device('cuda:0')
        self._init_lkf()

    def remove_model_from_gpu(self):
        del self.learnableKF

    def tracking(self, bbs_3D=None, features=None, scores=None, pose=None, timestamp=None, v2c=None, p2=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Track objects at the given timestamp.
        Args:
            bbs_3D: (N,7) or (N,7*k) array of 3D bounding boxes or tracklets.
            features: (N,k) array of features.
            scores: (N,) array of detection scores.
            pose: (4,4) pose matrix.
            timestamp: Current timestamp (int).
            v2c, p2: Calibration matrices (optional).
        Returns:
            bbs: (M,7) array of tracked bounding boxes.
            ids: (M,) array of assigned IDs.
        """
        self.current_bbs = bbs_3D
        self.current_features = features
        self.current_scores = scores
        self.current_pose = pose
        self.current_timestamp = timestamp
        self.v2c = v2c
        self.p2 = p2
        return self._pipeline()

    def _init_lkf(self):
        from configs.config_utils import get_cfg
        cfg = get_cfg()
        yaml_path = os.path.join(os.path.dirname(__file__), '../configs/training.yaml')
        cfg.merge_from_file(yaml_path)
        this_config = cfg
        m1x_0 = torch.zeros(self.track_dim, device=self.device)
        m2x_0 = torch.zeros(self.track_dim, device=self.device)
        Q = torch.eye(self.track_dim, device=self.device)
        P = torch.eye(self.track_dim, device=self.device)
        sys_model = SystemModel(f, Q, hRotate, P, 1, 1, m, n)
        sys_model.InitSequence(m1x_0, m2x_0)
        self.learnableKF = LEARNABLEKF(sys_model, this_config)
        self.learnableKF = torch.load(self.config.model_checkpoint, map_location=self.device)
        self.learnableKF = self.learnableKF.to(self.device)
        self.learnableKF.KNet_model.init_hidden_KNet()
        ones_init = torch.ones((self.batch_size, self.track_dim, 1), device=self.device)
        self.learnableKF.KNet_model.InitSequence(ones_init, 0)
        self.learnableKF.KNet_model.m1y = ones_init.clone()
        self.learnableKF.KNet_model.m1x_prior = ones_init.clone()

    def _pipeline(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._trajectories_prediction_step()
        bbs_empty, ids_empty = self._check_current_bbs_step()
        if bbs_empty is not None:
            return bbs_empty, ids_empty
        self._convert_bbs_step()
        self._register_bbs_step()
        ids = self._association_step()
        self._update_trajectories_step(ids)
        return self._trajectories_init_step(ids)

    def _trajectories_prediction_step(self):
        self._trajectories_prediction()

    def _check_current_bbs_step(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.current_bbs is None or len(self.current_bbs) == 0:
            return torch.zeros(0, 7), torch.zeros(0, dtype=torch.int64)
        return None, None

    def _convert_bbs_step(self):
        self.current_bbs = convert_bbs_type_initial(self.current_bbs, self.box_type)

    def _register_bbs_step(self):
        self.current_bbs = register_bbs_initial(self.current_bbs, self.current_pose)

    def _association_step(self):
        return self._association()

    def _update_trajectories_step(self, ids):
        self._trajectories_update(ids)

    def _trajectories_init_step(self, ids):
        return self._trajectorie_init(ids)

    def _trajectories_prediction(self):
        if not self.active_trajectories:
            return
        dead_track_id = []
        self._lkf_prediction()
        for key, traj in self.active_trajectories.items():
            if traj.consecutive_missed_num >= self.config.max_prediction_num:
                dead_track_id.append(key)
                continue
            if len(traj) - traj.consecutive_missed_num == 1 and len(traj) >= self.config.max_prediction_num_for_new_object:
                dead_track_id.append(key)
            adjusted_state = traj.state_prediction(self.current_timestamp, self.learnableKF.KNet_model.m1x_prior[traj.label])
            self.learnableKF.KNet_model.m1x_prior[traj.label] = adjusted_state.unsqueeze(1)
            self.learnableKF.KNet_model.m1y[traj.label] = adjusted_state.unsqueeze(1)
        for id in dead_track_id:
            self.dead_trajectories[id] = self.active_trajectories.pop(id)

    def _lkf_prediction(self):
        with torch.no_grad():
            self.learnableKF.KNet_model.step_prior()

    def _compute_cost_map(self):
        all_ids, all_predictions, all_detections = [], [], []
        for key, traj in self.active_trajectories.items():
            all_ids.append(key)
            state = traj.trajectory[self.current_timestamp].predicted_state.reshape(-1)
            meta = torch.tensor([1, traj.consecutive_missed_num, self.current_timestamp])
            state = torch.cat([state, meta])
            all_predictions.append(state)
        for i, box in enumerate(self.current_bbs):
            noisy_box = box.clone()
            noisy_box[:2] += torch.normal(mean=0.0, std=0.0, size=(2,))
            features = self.current_features[i] if self.current_features is not None else None
            score = self.current_scores[i]
            label = 1
            new_tra = ObjectPath(init_bb=noisy_box, init_features=features, init_score=score, init_timestamp=self.current_timestamp, label=label, tracking_features=self.tracking_features, bb_as_features=self.bb_as_features, config=self.config)
            state = new_tra.trajectory[self.current_timestamp].predicted_state.reshape(-1)
            all_detections.append(state)
        all_detections = torch.stack(all_detections)
        all_predictions = torch.stack(all_predictions)
        ciou_cost_3d, _ = ciou_3d(all_detections, all_predictions[..., :-3])
        threshold_3d_ciou = self.config.threshold_3d
        cost_dis_ciou_3d = 1 - ciou_cost_3d
        return cost_dis_ciou_3d, all_ids, threshold_3d_ciou

    def _association(self):
        if not self.active_trajectories:
            ids = [self.label_seed + i for i in range(len(self.current_bbs))]
            self.label_seed += len(self.current_bbs)
            return ids
        cost_map_3d, all_ids, threshold_3d = self._compute_cost_map()
        ids = []
        for i, _ in enumerate(self.current_bbs):
            min_val, arg_min = torch.min(cost_map_3d[i], dim=0)
            if min_val < threshold_3d:
                min_pred_val, _ = torch.min(cost_map_3d[:, arg_min], dim=0)
                if min_pred_val < min_val:
                    ids.append(self.label_seed)
                    self.label_seed += 1
                else:
                    ids.append(all_ids[arg_min])
                    cost_map_3d[:, arg_min] = 100000
            else:
                ids.append(self.label_seed)
                self.label_seed += 1
        return ids

    def _trajectories_update(self, ids):
        assert len(ids) == len(self.current_bbs), "IDs length must match current bounding boxes length"
        detected_state_template = self.learnableKF.KNet_model.m1x_prior.squeeze(2)
        for i, label in enumerate(ids):
            box = self.current_bbs[i]
            score = self.current_scores[i]
            if label in self.active_trajectories and score > self.config.update_score:
                detected_state_template[label] = box
        self._lkf_update_step(detected_state_template)

    def _lkf_update_step(self, detected_state_template):
        with torch.no_grad():
            detected_state_template = detected_state_template.to(self.device)
            self.learnableKF.KNet_model.y_previous = self.learnableKF.KNet_model.y_previous.to(self.device)
            self.learnableKF.KNet_model.step_KGain_est(detected_state_template)
            self.learnableKF.KNet_model.m1x_prior_previous = self.learnableKF.KNet_model.m1x_prior
            KF_gain = self.learnableKF.KNet_model.KGain
            dy = detected_state_template.unsqueeze(2) - self.learnableKF.KNet_model.m1y
            INOV = torch.bmm(KF_gain, dy)
            self.learnableKF.KNet_model.m1x_posterior_previous_previous = self.learnableKF.KNet_model.m1x_posterior_previous
            self.learnableKF.KNet_model.m1x_posterior_previous = self.learnableKF.KNet_model.m1x_posterior
            self.learnableKF.KNet_model.m1x_posterior = self.learnableKF.KNet_model.m1x_prior + INOV
            self.learnableKF.KNet_model.y_previous = detected_state_template.unsqueeze(-1)

    def _trajectorie_init(self, ids):
        assert len(ids) == len(self.current_bbs), "IDs length must match current bounding boxes length"
        valid_bbs, valid_ids = [], []
        for i, label in enumerate(ids):
            box = self.current_bbs[i]
            features = self.current_features[i] if self.current_features is not None else None
            score = self.current_scores[i]
            if label in self.active_trajectories and score > self.config.update_score:
                track = self.active_trajectories[label]
                track.state_update(bb=box, updated_state=self.learnableKF.KNet_model.m1x_posterior[label], h_sigma=self.learnableKF.KNet_model.h_Sigma.squeeze(0)[label], features=features, score=score, timestamp=self.current_timestamp)
                valid_bbs.append(box)
                valid_ids.append(label)
            elif score > self.config.init_score:
                new_tra = ObjectPath(init_bb=box, init_features=features, init_score=score, init_timestamp=self.current_timestamp, label=label, tracking_features=self.tracking_features, bb_as_features=self.bb_as_features, config=self.config)
                self.active_trajectories[label] = new_tra
                valid_bbs.append(box)
                valid_ids.append(label)
                self.learnableKF.KNet_model.m1x_posterior_previous_previous[label] = (box - 3e-8).unsqueeze(-1)
                self.learnableKF.KNet_model.m1x_posterior_previous[label] = (box - 2e-9).unsqueeze(-1)
                self.learnableKF.KNet_model.m1x_posterior[label] = box.unsqueeze(-1)
                self.learnableKF.KNet_model.m1x_prior_previous[label] = (box - 1e-8).unsqueeze(-1)
                self.learnableKF.KNet_model.m1x_prior[label] = box.unsqueeze(-1)
                self.learnableKF.KNet_model.m1y[label] = box.unsqueeze(-1)
                self.learnableKF.KNet_model.y_previous[label] = self.learnableKF.KNet_model.m1x_prior_previous[label]
        if not valid_bbs:
            return torch.zeros(0, self.current_bbs.shape[1]), torch.zeros(0, dtype=torch.int64)
        return torch.stack(valid_bbs), torch.tensor(valid_ids, dtype=torch.int64)

    def post_processing(self, config):
       
        tra = {**self.dead_trajectories, **self.active_trajectories}
        return tra
