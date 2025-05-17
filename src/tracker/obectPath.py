from model.LearnableKF import LEARNABLEKF
from tools.batch_generation import SystemModel
import torch
from tracker.object import Object

class ObjectPath:
    def __init__(self,init_bb=None,
                 init_features=None,
                 init_score=None,
                 init_timestamp=None,
                 label=None,
                 tracking_features=True,
                 bb_as_features=False,
                 config = None
                 ):
        """
        Args:
            init_bb: array(7) or array(7*k), 3d box or tracklet
            init_features: array(m), features of box or tracklet
            init_score: array(1) or float, score of detection
            init_timestamp: int, init timestamp
            label: int, unique ID for this trajectory
            tracking_features: bool, if track features
            bb_as_features: bool, if treat the bb as features
        """
        assert init_bb is not None

        self.init_bb = init_bb
        self.init_features = init_features
        self.init_score = init_score
        self.init_timestamp = init_timestamp
        self.label = label
        self.tracking_bb_size = True
        self.tracking_features = tracking_features
        self.bb_as_features = bb_as_features
        self.config = config
        self.scanning_interval = 1. / self.config.LiDAR_scanning_frequency

        if self.bb_as_features:
            if self.init_features is None:
                self.init_features = init_bb
            else:
                self.init_features = torch.cat([init_bb, init_features], 0)
        self.trajectory = {}
        self.track_dim = self.compute_track_dim()
        self.init_parameters()
        self.init_trajectory()
        self.consecutive_missed_num = 0
        self.first_updated_timestamp = init_timestamp
        self.last_updated_timestamp = init_timestamp
        self.total_detected_score = init_score
        self.total_detections = 1

    def __len__(self):
        return len(self.trajectory)

    def compute_track_dim(self):
        track_dim = 3
        if self.tracking_bb_size:
            track_dim += 4
        if self.tracking_features:
            track_dim += self.init_features.shape[0]
        return track_dim

    def init_trajectory(self):
        detected_state_template = torch.zeros(self.track_dim)
        update_covariance_template = torch.eye(self.track_dim) * 0.01
        detected_state_template[:3] = self.init_bb[:3]
        if self.tracking_bb_size:
            detected_state_template[3:7] = self.init_bb[3:7]
            if self.tracking_features:
                detected_state_template[7:] = torch.tensor(self.init_features)
        else:
            if self.tracking_features:
                detected_state_template[3:] = torch.tensor(self.init_features)
        update_state_template = self.H @ detected_state_template
        obj = Object()
        obj.updated_state = update_state_template
        obj.predicted_state = update_state_template
        obj.detected_state = detected_state_template
        obj.updated_covariance = update_covariance_template
        obj.score = self.init_score
        obj.features = self.init_features
        obj.timestamp = self.init_timestamp
        self.trajectory[self.init_timestamp] = obj

    def init_parameters(self):
        self.A = torch.eye(self.track_dim)
        self.Q = torch.eye(self.track_dim) * self.config.state_func_covariance
        self.P = torch.eye(self.track_dim) * self.config.measure_func_covariance
        self.B = torch.zeros(self.track_dim, self.track_dim)
        self.B[:, :] = self.A[:, :]
        self.H = self.B.T
        self.K = torch.zeros(self.track_dim, self.track_dim)

    def state_prediction(self, timestamp, state_prior):
        previous_timestamp = timestamp - 1
        assert previous_timestamp in self.trajectory.keys()
        previous_object = self.trajectory[previous_timestamp]
        if self.last_updated_timestamp == previous_timestamp:
            previous_state = previous_object.updated_state
            previous_covariance = previous_object.updated_covariance
            previous_score = previous_object.score
        else:
            previous_score = previous_object.score
            previous_state = previous_object.predicted_state
            previous_covariance = previous_object.predicted_covariance
        current_predicted_state_lkf = state_prior.squeeze(-1).to('cpu')
        current_predicted_covariance = self.A @ previous_covariance @ self.A.T + self.Q
        new_ob = Object()
        object_distance = torch.sqrt(current_predicted_state_lkf[0] ** 2 + current_predicted_state_lkf[1] ** 2 + current_predicted_state_lkf[2] ** 2)
        object_distance_previous = torch.sqrt(previous_state[0] ** 2 + previous_state[1] ** 2 + previous_state[2] ** 2)
        object_velocity = object_distance - object_distance_previous
        ob_last_updated_ob = self.trajectory[self.last_updated_timestamp]
        last_update_state = ob_last_updated_ob.updated_state
        adjusted_predicted_state_lkf = self.adjust_predicted_state(
            timestamp, previous_object, previous_state, current_predicted_state_lkf, object_distance, object_velocity, self.last_updated_timestamp, last_update_state)
        new_ob.predicted_state = adjusted_predicted_state_lkf
        new_ob.predicted_covariance = current_predicted_covariance
        current_score = previous_score
        if self.config.latency == 1:
            if not self.config.post_process_interpolation:
                new_ob.updated_state = adjusted_predicted_state_lkf
            new_ob.score = self.total_detected_score / self.total_detections
        new_ob.timestamp = timestamp
        self.consecutive_missed_num += 1
        self.trajectory[timestamp] = new_ob
        return adjusted_predicted_state_lkf

    def state_update(self, bb=None, updated_state=None, h_sigma=None, features=None, score=None, timestamp=None):
        """
        Update the trajectory with a new detection.
        Args:
            bb: array(7) or array(7*k), 3D box or tracklet
            features: array(m), features of box or tracklet
            score: detection score
            timestamp: current timestamp
        """
        assert bb is not None
        assert timestamp in self.trajectory.keys()
        if self.bb_as_features:
            if features is None:
                features = bb
            else:
                features = torch.cat([bb, features], dim=0)
        detected_state_template = torch.zeros((self.track_dim))
        detected_state_template[:3] = bb[:3]
        if self.tracking_bb_size:
            detected_state_template[3:7] = bb[3:7]
            if self.tracking_features:
                detected_state_template[7:] = features[:]
        else:
            if self.tracking_features:
                detected_state_template[3:] = features[:]
        current_ob = self.trajectory[timestamp]
        updated_state_lkf = updated_state.squeeze(-1).to('cpu')
        updated_covariance_lkf = h_sigma.squeeze(-1).to('cpu')
        updated_covariance_lkf = updated_covariance_lkf.reshape(1, 7, 7).squeeze(0)
        current_ob.updated_state = updated_state_lkf
        current_ob.updated_covariance = updated_covariance_lkf
        current_ob.detected_state = detected_state_template
        current_ob.features = features
        self.consecutive_missed_num = 0
        self.last_updated_timestamp = timestamp
        self.total_detected_score += score
        self.total_detections += 1
        current_ob.score = self.total_detected_score / self.total_detections


    def adjust_predicted_state(self, timestamp, previous_object, previous_state, current_predicted_state_lkf, object_distance, object_velocity, last_updated_timestamp, last_updated_step):
        """
        Adjust the predicted object state considering missed detections and noisy early detections mitigations.
        """
        if (timestamp - self.first_updated_timestamp) < self.config.max_security_window:
            adjustment_factor = 1 / self.config.max_security_window * (timestamp - self.first_updated_timestamp)
        else:
            adjustment_factor = 1
        if self.consecutive_missed_num == 0:
            pass
        else:
            if self.consecutive_missed_num < self.config.max_prediction_num:
                adjustment_factor *= (1 - self.consecutive_missed_num / self.config.max_prediction_num) * 0.8 + 0.1
            else:
                adjustment_factor *= 0.1
        current_predicted_state_lkf[0:3] = previous_state[0:3] + (current_predicted_state_lkf[0:3] - previous_state[0:3]) * adjustment_factor
        return current_predicted_state_lkf

