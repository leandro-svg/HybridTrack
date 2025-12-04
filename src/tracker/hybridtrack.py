import os
import torch
from typing import Optional, Tuple, List, Dict
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

        # Safe checkpoint loading: handle state_dict vs full model
        ckpt = torch.load(self.config.model_checkpoint, map_location=self.device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            missing, unexpected = self.learnableKF.load_state_dict(ckpt['state_dict'], strict=False)
            if missing:
                print(f"[HybridTrack] Warning: Missing keys in checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                print(f"[HybridTrack] Warning: Unexpected keys in checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        elif isinstance(ckpt, dict) and all(k.startswith('LKF_model') or k.startswith('module') for k in ckpt.keys()):
            missing, unexpected = self.learnableKF.load_state_dict(ckpt, strict=False)
            if missing:
                print(f"[HybridTrack] Warning: Missing keys in checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                print(f"[HybridTrack] Warning: Unexpected keys in checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
        else:
            # Assume full model was saved
            self.learnableKF = ckpt.to(self.device)

        # Log context configuration status
        ctx_cfg = getattr(self.config, 'CONTEXT', None)
        if ctx_cfg is not None:
            print(f"[HybridTrack] Context config: USE_CONTEXT={getattr(ctx_cfg, 'USE_CONTEXT', False)}, USE_HISTORY={getattr(ctx_cfg, 'USE_HISTORY', False)}")
        else:
            print("[HybridTrack] No CONTEXT config found - context disabled")

        self.learnableKF = self.learnableKF.to(self.device)
        self.learnableKF.LKF_model.init_hidden_LKF()
        ones_init = torch.ones((self.batch_size, self.track_dim, 1), device=self.device)
        self.learnableKF.LKF_model.InitSequence(ones_init, 0)
        self.learnableKF.LKF_model.m1y = ones_init.clone()
        self.learnableKF.LKF_model.m1x_prior = ones_init.clone()
        # Ensure internal batch size matches trajectory pool
        if hasattr(self.learnableKF, 'LKF_model'):
            self.learnableKF.LKF_model.update_batch_size(self.batch_size)

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
        lkf_model = self.learnableKF.LKF_model  
        self._lkf_prediction()
        for key, traj in self.active_trajectories.items():
            if traj.consecutive_missed_num >= self.config.max_prediction_num:
                dead_track_id.append(key)
                continue
            if len(traj) - traj.consecutive_missed_num == 1 and len(traj) >= self.config.max_prediction_num_for_new_object:
                dead_track_id.append(key)
            adjusted_state = traj.state_prediction(self.current_timestamp, lkf_model.m1x_prior[traj.label])
            lkf_model.m1x_prior[traj.label] = adjusted_state.unsqueeze(1)
            lkf_model.m1y[traj.label] = adjusted_state.unsqueeze(1)
        for id in dead_track_id:
            self.dead_trajectories[id] = self.active_trajectories.pop(id)

    def _lkf_prediction(self):
        with torch.no_grad():
            # Check if context is enabled
            ctx_cfg = getattr(self.config, 'CONTEXT', None)
            use_context = ctx_cfg is not None and getattr(ctx_cfg, 'USE_CONTEXT', False)
            
            # Build detection and history contexts + masks
            det_ctx, det_mask = self._prepare_detection_context()
            history_len = int(getattr(ctx_cfg, 'HISTORY_LEN', 5)) if ctx_cfg else 5
            hist_ctx, hist_mask = self._prepare_history_context(history_len=history_len)
            
            # EXTENSIVE DEBUG: Log what's being passed to LKF
            if not hasattr(self, '_lkf_debug_counter'):
                self._lkf_debug_counter = 0
            self._lkf_debug_counter += 1
            if self._lkf_debug_counter <= 10:
                print(f"\n[LKF PREDICTION INPUT] Frame {self.current_timestamp}, Call {self._lkf_debug_counter}")
                print(f"  use_context={use_context}")
                if det_ctx is not None:
                    print(f"  det_ctx: shape={det_ctx.shape}, mean={det_ctx.mean().item():.4f}, std={det_ctx.std().item():.4f}")
                    print(f"  det_mask: shape={det_mask.shape}, num_valid={(~det_mask).sum().item()}")
                else:
                    print(f"  det_ctx: None")
                if hist_ctx is not None:
                    print(f"  hist_ctx: shape={hist_ctx.shape}")
                else:
                    print(f"  hist_ctx: None")
            
            self.learnableKF.LKF_model.step_prior(
                det_context=det_ctx,
                det_mask=det_mask,
                hist_context=hist_ctx,
                hist_mask=hist_mask
            )

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
        detected_state_template = self.learnableKF.LKF_model.m1x_prior.squeeze(2)
        for i, label in enumerate(ids):
            box = self.current_bbs[i]
            score = self.current_scores[i]
            if label in self.active_trajectories and score > self.config.update_score:
                detected_state_template[label] = box
        self._lkf_update_step(detected_state_template)

    def _lkf_update_step(self, detected_state_template):
        with torch.no_grad():
            detected_state_template = detected_state_template.to(self.device)
            lkf_model = self.learnableKF.LKF_model
            lkf_model.y_previous = lkf_model.y_previous.to(self.device)
            lkf_model.step_KGain_est(detected_state_template)
            lkf_model.m1x_prior_previous = lkf_model.m1x_prior
            KF_gain = lkf_model.KGain
            dy = detected_state_template.unsqueeze(2) - lkf_model.m1y
            INOV = torch.bmm(KF_gain, dy)
            lkf_model.m1x_posterior_previous_previous = lkf_model.m1x_posterior_previous
            lkf_model.m1x_posterior_previous = lkf_model.m1x_posterior
            lkf_model.m1x_posterior = lkf_model.m1x_prior + INOV
            lkf_model.y_previous = detected_state_template.unsqueeze(-1)

    def _trajectorie_init(self, ids):
        assert len(ids) == len(self.current_bbs), "IDs length must match current bounding boxes length"
        valid_bbs, valid_ids = [], []
        lkf_model = self.learnableKF.LKF_model
        for i, label in enumerate(ids):
            box = self.current_bbs[i]
            features = self.current_features[i] if self.current_features is not None else None
            score = self.current_scores[i]
            if label in self.active_trajectories and score > self.config.update_score:
                track = self.active_trajectories[label]
                track.state_update(bb=box, updated_state=lkf_model.m1x_posterior[label], h_sigma=lkf_model.h_Sigma.squeeze(0)[label], features=features, score=score, timestamp=self.current_timestamp)
                valid_bbs.append(box)
                valid_ids.append(label)
            elif score > self.config.init_score:
                new_tra = ObjectPath(init_bb=box, init_features=features, init_score=score, init_timestamp=self.current_timestamp, label=label, tracking_features=self.tracking_features, bb_as_features=self.bb_as_features, config=self.config)
                self.active_trajectories[label] = new_tra
                valid_bbs.append(box)
                valid_ids.append(label)
                # Initialize LKF model state for new trajectory
                lkf_model.m1x_posterior_previous_previous[label] = (box - 3e-8).unsqueeze(-1)
                lkf_model.m1x_posterior_previous[label] = (box - 2e-9).unsqueeze(-1)
                lkf_model.m1x_posterior[label] = box.unsqueeze(-1)
                lkf_model.m1x_prior_previous[label] = (box - 1e-8).unsqueeze(-1)
                lkf_model.m1x_prior[label] = box.unsqueeze(-1)
                lkf_model.m1y[label] = box.unsqueeze(-1)
                lkf_model.y_previous[label] = lkf_model.m1x_prior_previous[label]
        if not valid_bbs:
            return torch.zeros(0, self.current_bbs.shape[1]), torch.zeros(0, dtype=torch.int64)
        return torch.stack(valid_bbs), torch.tensor(valid_ids, dtype=torch.int64)

    def post_processing(self, config):
        tra = {**self.dead_trajectories, **self.active_trajectories}
        return tra

    def _prepare_detection_context(self):
        """
        Prepare detection context tensor (B, N, 8) and mask (B, N) for current frame.
        
        NEW DESIGN: For each ego track slot in batch B, provide context about OTHER
        active tracks (not the ego itself). This gives meaningful interaction information.
        
        Context format: [dx, dy, dz, l, w, h, dry, score] where dx/dy/dz are RELATIVE
        to the ego track's current predicted position.
        
        Returns:
            context: (B, N, 8) tensor where B is track pool size
            mask: (B, N) boolean mask, True for padded positions
        """
        # DEBUG: Track why context is None
        if not hasattr(self, '_ctx_prep_debug_counter'):
            self._ctx_prep_debug_counter = 0
        self._ctx_prep_debug_counter += 1
        
        # Check if context is enabled
        ctx_cfg = getattr(self.config, 'CONTEXT', None)
        if ctx_cfg is None or not getattr(ctx_cfg, 'USE_CONTEXT', False):
            if self._ctx_prep_debug_counter <= 3:
                print(f"[CTX PREP DEBUG] Frame {self.current_timestamp}: Context DISABLED in config")
            return None, None
            
        if not self.active_trajectories:
            if self._ctx_prep_debug_counter <= 3:
                print(f"[CTX PREP DEBUG] Frame {self.current_timestamp}: NO active trajectories")
            return None, None

        import numpy as np
        
        lkf_model = self.learnableKF.LKF_model
        B = lkf_model.batch_size
        max_n = int(getattr(ctx_cfg, 'MAX_CONTEXT_OBJECTS', 16))
        # Reduced from 50m to 30m - distant objects are noise, not useful context
        dist_thresh = float(getattr(ctx_cfg, 'DISTANCE_THRESH', 30.0))
        
        # Get all active track IDs and their current predicted positions
        active_track_ids = list(self.active_trajectories.keys())
        
        if len(active_track_ids) < 2:
            # Need at least 2 tracks for meaningful context
            if self._ctx_prep_debug_counter <= 10:
                print(f"[CTX PREP DEBUG] Frame {self.current_timestamp}: Only {len(active_track_ids)} track(s), need >=2")
            return None, None
        
        # Build position dict: track_id -> current state (7,)
        # Use updated_state from PREVIOUS frame (not current!) since current frame's 
        # updated_state doesn't exist yet during prediction phase
        track_positions = {}
        for tid in active_track_ids:
            traj = self.active_trajectories[tid]
            # Get most recent timestamp before current frame that has updated_state
            timestamps = sorted([ts for ts in traj.trajectory.keys() if ts < self.current_timestamp])
            if len(timestamps) > 0:
                most_recent_ts = timestamps[-1]
                obj = traj.trajectory[most_recent_ts]
                if obj.updated_state is not None:
                    track_positions[tid] = obj.updated_state[:7].clone()
        
        if len(track_positions) < 2:
            if self._ctx_prep_debug_counter <= 10:
                print(f"[CTX PREP DEBUG] Frame {self.current_timestamp}: Only {len(track_positions)} valid position(s), need >=2")
            return None, None
        
        # Build context for each track slot
        # context[i] = relative positions of OTHER tracks to track i
        context_list = []
        mask_list = []
        
        for slot_idx in range(B):
            # Find which track (if any) is in this slot
            ego_tid = None
            ego_pos = None
            for tid, pos in track_positions.items():
                if tid == slot_idx:  # Track ID matches slot index
                    ego_tid = tid
                    ego_pos = pos
                    break
            
            if ego_tid is None or ego_pos is None:
                # No active track in this slot - provide zero context
                ctx = torch.zeros((max_n, 8), dtype=torch.float32, device=self.device)
                msk = torch.ones(max_n, dtype=torch.bool, device=self.device)
            else:
                # Build relative context: other tracks' positions relative to ego
                other_contexts = []
                for other_tid, other_pos in track_positions.items():
                    if other_tid == ego_tid:
                        continue  # Skip ego itself
                    
                    # Compute relative position
                    dx = other_pos[0] - ego_pos[0]  # relative x
                    dy = other_pos[1] - ego_pos[1]  # relative y
                    dz = other_pos[2] - ego_pos[2]  # relative z
                    
                    dist = torch.sqrt(dx**2 + dy**2)
                    if dist > dist_thresh:
                        continue  # Too far
                    
                    # Relative yaw
                    dry = other_pos[6] - ego_pos[6]
                    # Normalize to [-pi, pi]
                    dry = torch.atan2(torch.sin(dry), torch.cos(dry))
                    
                    # Get score (use average detection score, normalized to [0,1])
                    other_traj = self.active_trajectories.get(other_tid)
                    if other_traj and other_traj.total_detections > 0:
                        # Clip raw score to reasonable range and normalize
                        raw_score = other_traj.total_detected_score / other_traj.total_detections
                        score = min(max(raw_score, 0.0), 1.0)  # Clip to [0,1]
                    else:
                        score = 0.5
                    
                    # TEMPORARY: Disable velocity computation to match training (8-dim padded to 11-dim with zeros)
                    # TODO: Add velocity computation to training dataset, then re-enable this
                    dvx, dvy, dvz = 0.0, 0.0, 0.0  # Always zero to match training
                    
                    # VELOCITY COMPUTATION DISABLED - keeping code structure but not executing
                    if False:  # len(other_prev_ts) >= 2 and ego_traj:
                        # Check if updated_state exists for both timestamps
                        other_obj_curr = other_traj.trajectory[other_prev_ts[-1]]
                        other_obj_prev = other_traj.trajectory[other_prev_ts[-2]]
                        
                        if (other_obj_curr.updated_state is not None and 
                            other_obj_prev.updated_state is not None):
                            # Other vehicle velocity
                            other_pos_curr = other_obj_curr.updated_state[:3]
                            other_pos_prev = other_obj_prev.updated_state[:3]
                            other_velocity = (other_pos_curr - other_pos_prev).cpu()  # (3,)
                            
                            # Ego vehicle velocity
                            ego_prev_ts = sorted([t for t in ego_traj.trajectory.keys() 
                                                 if t < self.current_timestamp])
                            if len(ego_prev_ts) >= 2:
                                ego_obj_curr = ego_traj.trajectory[ego_prev_ts[-1]]
                                ego_obj_prev = ego_traj.trajectory[ego_prev_ts[-2]]
                                
                                if (ego_obj_curr.updated_state is not None and 
                                    ego_obj_prev.updated_state is not None):
                                    ego_pos_curr = ego_obj_curr.updated_state[:3]
                                    ego_pos_prev = ego_obj_prev.updated_state[:3]
                                    ego_velocity = (ego_pos_curr - ego_pos_prev).cpu()  # (3,)
                                    
                                    # Relative velocity
                                    rel_velocity = other_velocity - ego_velocity
                                    dvx, dvy, dvz = float(rel_velocity[0]), float(rel_velocity[1]), float(rel_velocity[2])
                    
                    # Context: [dx, dy, dz, l, w, h, dry, score, dvx, dvy, dvz]
                    ctx_entry = torch.tensor([
                        dx.item(), dy.item(), dz.item(),
                        other_pos[3].item(), other_pos[4].item(), other_pos[5].item(),
                        dry.item(), score
                    ], dtype=torch.float32, device=self.device)
                    
                    other_contexts.append((dist.item(), ctx_entry))
                
                # Sort by distance (closest first) and take top-N
                other_contexts.sort(key=lambda x: x[0])
                other_contexts = [c[1] for c in other_contexts[:max_n]]
                
                n_ctx = len(other_contexts)
                if n_ctx == 0:
                    ctx = torch.zeros((max_n, 8), dtype=torch.float32, device=self.device)
                    msk = torch.ones(max_n, dtype=torch.bool, device=self.device)
                else:
                    ctx = torch.stack(other_contexts, dim=0)  # (n_ctx, 8)
                    # Pad to max_n
                    if n_ctx < max_n:
                        pad = torch.zeros((max_n - n_ctx, 8), dtype=torch.float32, device=self.device)
                        ctx = torch.cat([ctx, pad], dim=0)
                    msk = torch.zeros(max_n, dtype=torch.bool, device=self.device)
                    msk[n_ctx:] = True
            
            context_list.append(ctx)
            mask_list.append(msk)
        
        context = torch.stack(context_list, dim=0)  # (B, max_n, 8)
        mask = torch.stack(mask_list, dim=0)  # (B, max_n)
        
        # EXTENSIVE DEBUG: Log context preparation
        if not hasattr(self, '_ctx_debug_counter'):
            self._ctx_debug_counter = 0
        self._ctx_debug_counter += 1
        if self._ctx_debug_counter <= 10:
            valid_per_slot = (~mask).sum(dim=1)
            total_valid = valid_per_slot.sum().item()
            print(f"\n{'='*80}")
            print(f"[CONTEXT PREP] Frame {self.current_timestamp}, Call {self._ctx_debug_counter}")
            print(f"{'='*80}")
            print(f"  Active tracks: {len(active_track_ids)}, IDs: {active_track_ids[:10]}")
            print(f"  Track positions available: {len(track_positions)}")
            print(f"  Context tensor shape: {context.shape}")
            print(f"  Total valid context entries: {total_valid}")
            print(f"  Valid per slot (first 10): {valid_per_slot[:10].tolist()}")
            print(f"  Context stats: mean={context.mean().item():.4f}, std={context.std().item():.4f}")
            print(f"  Context range: min={context.min().item():.4f}, max={context.max().item():.4f}")
            if total_valid > 0:
                # Show detailed info for first slot with context
                for i in range(min(3, B)):
                    if valid_per_slot[i] > 0:
                        n_valid = valid_per_slot[i].item()
                        print(f"\n  Slot {i} (ego track {i}): {n_valid} neighbors")
                        for j in range(min(3, n_valid)):
                            ctx = context[i, j]
                            print(f"    Neighbor {j}: dx={ctx[0]:.2f}, dy={ctx[1]:.2f}, dz={ctx[2]:.2f}, ")
                            print(f"               l={ctx[3]:.2f}, w={ctx[4]:.2f}, h={ctx[5]:.2f}, ")
                            print(f"               dry={ctx[6]:.3f}, score={ctx[7]:.3f}")
            print(f"{'='*80}\n")
            
        return context, mask

    def _prepare_history_context(self, history_len: int = 5):
        """
        Prepare history context tensor (B, K, H, 7) and mask (B, K).
        
        Uses the stored trajectory states for other active tracks.
        For each nearby track, extracts the last H posterior (updated) states.
        This gives the model information about how other vehicles have been moving,
        which helps with forecasting and interaction modeling.
        """
        # Check if history is enabled
        ctx_cfg = getattr(self.config, 'CONTEXT', None)
        if ctx_cfg is None or not getattr(ctx_cfg, 'USE_HISTORY', False):
            return None, None
            
        if not self.active_trajectories:
            return None, None
            
        max_k = int(getattr(ctx_cfg, 'MAX_CONTEXT_OBJECTS', 32))
        H = int(getattr(ctx_cfg, 'HISTORY_LEN', history_len))
        
        lkf_model = self.learnableKF.LKF_model
        B = lkf_model.batch_size
        
        active_track_ids = list(self.active_trajectories.keys())
        
        if len(active_track_ids) == 0:
            return None, None
        
        # Build history from trajectory objects (which store full history)
        entries = []
        for tid in active_track_ids:
            traj = self.active_trajectories[tid]
            
            if not hasattr(traj, 'trajectory') or len(traj.trajectory) == 0:
                continue
            
            # Get sorted frame indices
            frames = sorted(traj.trajectory.keys())
            
            # Extract last H updated_states (posterior states)
            seq_list = []
            for f in frames[-H:]:
                obj = traj.trajectory[f]
                # Prefer updated_state (posterior) over predicted_state
                if obj.updated_state is not None:
                    state = obj.updated_state[:7]
                elif obj.predicted_state is not None:
                    state = obj.predicted_state[:7]
                else:
                    continue
                seq_list.append(torch.as_tensor(state, dtype=torch.float32))
            
            if len(seq_list) == 0:
                continue
            
            seq = torch.stack(seq_list, dim=0)  # (t<=H, 7)
            
            # Compute distance from origin (current position)
            current_pos = seq[-1, :2]
            dist = torch.norm(current_pos)
            entries.append((tid, dist.item(), seq))
        
        if len(entries) == 0:
            return None, None
        
        # Sort by distance (closest first) and take up to max_k
        entries.sort(key=lambda x: x[1])
        selected = entries[:max_k]
        
        K = len(selected)
        E = 7  # State dimension
        
        # Build the history tensor
        hist_list = []
        for tid, _, seq in selected:
            # Pad to history length H if needed (pad at beginning for older missing frames)
            if seq.size(0) < H:
                pad = torch.zeros((H - seq.size(0), E), dtype=seq.dtype, device=self.device)
                seq = torch.cat([pad, seq.to(self.device)], dim=0)
            elif seq.size(0) > H:
                seq = seq[-H:].to(self.device)
            else:
                seq = seq.to(self.device)
            hist_list.append(seq)
        
        hist = torch.stack(hist_list, dim=0)  # (K, H, 7)
        
        # Expand to batch dimension
        hist = hist.unsqueeze(0).expand(B, -1, -1, -1)  # (B, K, H, 7)
        
        # Pad to max_k if needed
        if K < max_k:
            pad = torch.zeros((B, max_k - K, H, E), dtype=hist.dtype, device=hist.device)
            hist = torch.cat([hist, pad], dim=1)
            mask = torch.zeros((B, max_k), dtype=torch.bool, device=hist.device)
            mask[:, K:] = True
        else:
            mask = torch.zeros((B, K), dtype=torch.bool, device=hist.device)
        
        return hist, mask
