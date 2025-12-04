"""
"""
import os
import sys
import random
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from evaluator.eval_render import evaluate_heading, evaluate_trans
from .plotting import plot_bev, plot_all_batches
from .losses import euclidean_distance, calculate_speed_vectors, direction_consistency_loss, calculate_losses

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class TrainingPipeline:
    """
    """
    def __init__(self, Time, folder_name, model_name):
        self.Time = Time
        self.folder_name = os.path.join(folder_name, '')
        self.model_name = model_name
        self.device = torch.device('cpu')
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.xyz_loss = nn.L1Loss()
        self.scaler = MinMaxScaler()
        self.conf_loss = nn.BCELoss()
        self.state_size = 7
        self.logger = logging.getLogger(self.__class__.__name__)

    def save(self, path):
        torch.save(self, path)

    def set_ss_model(self, ss_model):
        self.ss_model = ss_model

    def set_model(self, model):
        self.model = model

    def set_training_params(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if cfg.TRAINER.USE_CUDA else 'cpu')
        self.N_steps = cfg.TRAINER.EPOCH
        self.N_B = cfg.TRAINER.BATCH_SIZE
        self.learning_rate = cfg.TRAINER.LR
        self.weight_decay = cfg.TRAINER.WD
        self.alpha = cfg.TRAINER.ALPHA
        self.N_T = cfg.TRAINER.N_TEST
        self.T_test = cfg.TRAINER.T_TEST
        self.relative_loss = cfg.TRAINER.RELATIVE_LOSS
        self.temporal_loss = cfg.TRAINER.TEMPORAL_LOSS
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.9, patience=25)

    def _calculate_losses(self, target_batch, state_out_batch, x_out_batch, attributes):
        # Use the shared calculate_losses from losses.py
        return calculate_losses(target_batch, state_out_batch, x_out_batch, attributes, self.xyz_loss)

    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, cfg,
                MaskOnState=False, randomInit=False, cv_init=None, train_init=None,
                train_lengthMask=None, cv_lengthMask=None,
                train_context=None, cv_context=None,
                train_history=None, cv_history=None):
        """
        Train LKF model.
        """
        root = cfg.DATASET.ROOT
        log_path = os.path.join(path_results, "logs")
        os.makedirs(os.path.join(root, log_path), exist_ok=True)
        logging.info(f"Results will be saved in: {log_path}")
        sys.stdout = open(os.path.join(log_path, "logs.txt"), "w")
        path_results_weight = os.path.join(path_results, 'weights')
        os.makedirs(os.path.join(root, path_results_weight), exist_ok=True)
        path_results_weight_per_epoch = os.path.join(path_results_weight, 'weights_per_epoch')
        os.makedirs(os.path.join(root, path_results_weight_per_epoch), exist_ok=True)
        data_path = os.path.join(root, path_results, "data_per_epoch")
        os.makedirs(data_path, exist_ok=True)
        self.N_E = len(train_input)
        self.N_CV = len(cv_input)
        self.robust_loss = nn.SmoothL1Loss()
        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])
        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_opt = float('inf')
        for ti in range(self.N_steps):
            self.optimizer.zero_grad()
            self.model.train()
            self.model.LKF_model.batch_size = self.N_B
            for param in self.model.parameters():
                assert not torch.isnan(param).any(), "Model parameter has NaN values."
            self.model.LKF_model.init_hidden_LKF()
            y_training_batch = torch.empty((self.N_B, self.state_size, self.ss_model.T))
            train_target_batch = torch.empty((self.N_B, self.state_size, self.ss_model.T))
            x_out_training_batch = torch.empty((self.N_B, self.state_size, self.ss_model.T))
            state_out_training_batch = torch.empty((self.N_B, self.state_size, self.ss_model.T))
            initializer = torch.empty((self.N_B, self.state_size, 1))
            n_e = random.sample(range(self.N_E), k=self.N_B)
            for ii, index in enumerate(n_e):
                train_elem = train_input[index][0]
                for idx, elem in enumerate(train_elem):
                    y_training_batch[ii, :, idx] = elem
                    if idx == 0:
                        initializer[ii, :, 0] = elem
                for idx, elem in enumerate(train_target[index][0]):
                    train_target_batch[ii, :, idx] = elem
            self.model.LKF_model.InitSequence(initializer, self.ss_model.T)
            for t in range(self.ss_model.T):
                # Optional detection context per batch/time
                det_ctx_batch = None
                det_mask_batch = None
                hist_ctx_batch = None
                hist_mask_batch = None
                
                use_ctx = getattr(cfg, 'CONTEXT', None)
                if bool(getattr(use_ctx, 'USE_CONTEXT', False)) and train_context is not None:
                    # Collect per-sample frame contexts and pad
                    per_sample_ctx = []
                    max_n = 0
                    for ii, index in enumerate(n_e):
                        try:
                            frames = train_context[index]
                            ctx = frames[t] if t < len(frames) else None
                            if ctx is None or (torch.is_tensor(ctx) and ctx.numel() == 0):
                                ctx = torch.zeros((0, 8), dtype=torch.float32)
                            elif not torch.is_tensor(ctx):
                                ctx = torch.tensor(ctx, dtype=torch.float32)
                        except Exception:
                            ctx = torch.zeros((0, 8), dtype=torch.float32)
                        per_sample_ctx.append(ctx)
                        max_n = max(max_n, ctx.size(0) if ctx.dim() > 0 else 0)
                    
                    if max_n > 0:
                        padded = []
                        mask = torch.zeros((self.N_B, max_n), dtype=torch.bool)
                        for ii, ctx in enumerate(per_sample_ctx):
                            n_ctx = ctx.size(0) if ctx.dim() > 0 and ctx.numel() > 0 else 0
                            if n_ctx < max_n:
                                pad = torch.zeros((max_n - n_ctx, 8), dtype=torch.float32)
                                if n_ctx > 0:
                                    ctx = torch.cat([ctx, pad], dim=0)
                                else:
                                    ctx = pad
                                mask[ii, n_ctx:] = True  # Mask padding positions
                            padded.append(ctx)
                        det_ctx_batch = torch.stack(padded, dim=0).to(self.device)
                        det_mask_batch = mask.to(self.device)
                        
                        # DEBUG: Log training context statistics
                        if not hasattr(self, '_train_ctx_debug_counter'):
                            self._train_ctx_debug_counter = 0
                        self._train_ctx_debug_counter += 1
                        if self._train_ctx_debug_counter <= 5:
                            print(f"\n[DEBUG TRAIN CTX] step {self._train_ctx_debug_counter}:")
                            print(f"  det_ctx_batch shape: {det_ctx_batch.shape}")
                            print(f"  det_ctx xyz mean: {det_ctx_batch[:,:,:3].mean().item():.4f}, std: {det_ctx_batch[:,:,:3].std().item():.4f}")
                            print(f"  det_ctx score mean: {det_ctx_batch[:,:,7].mean().item():.4f}")
                            print(f"  det_mask sum (padded): {det_mask_batch.sum().item()}")
                
                # Optional history context per batch/time
                if bool(getattr(use_ctx, 'USE_HISTORY', False)) and train_history is not None:
                    # Build history context: (B, K, H, 7) with mask (B, K)
                    per_sample_hist = []
                    per_sample_hist_mask = []
                    for ii, index in enumerate(n_e):
                        try:
                            hist_dict = train_history[index]
                            if hist_dict and 'history_per_frame' in hist_dict:
                                frame_hist = hist_dict['history_per_frame'][t]
                                histories = torch.tensor(frame_hist['histories'], dtype=torch.float32)  # (K, H, 7)
                                valid_masks = torch.tensor(frame_hist['valid_masks'], dtype=torch.bool)  # (K, H)
                                # Compute agent-level mask: True if all time steps are invalid (padded agent)
                                agent_mask = ~valid_masks.any(dim=1)  # (K,)
                            else:
                                # Empty history
                                histories = torch.zeros((1, 5, 7), dtype=torch.float32)
                                agent_mask = torch.ones(1, dtype=torch.bool)
                        except Exception:
                            histories = torch.zeros((1, 5, 7), dtype=torch.float32)
                            agent_mask = torch.ones(1, dtype=torch.bool)
                        per_sample_hist.append(histories)
                        per_sample_hist_mask.append(agent_mask)
                    
                    # Pad to max K across batch
                    max_k = max(h.size(0) for h in per_sample_hist)
                    H = per_sample_hist[0].size(1)
                    padded_hist = []
                    padded_mask = []
                    for histories, agent_mask in zip(per_sample_hist, per_sample_hist_mask):
                        K = histories.size(0)
                        if K < max_k:
                            pad_hist = torch.zeros((max_k - K, H, 7), dtype=histories.dtype)
                            histories = torch.cat([histories, pad_hist], dim=0)
                            pad_mask = torch.ones(max_k - K, dtype=torch.bool)
                            agent_mask = torch.cat([agent_mask, pad_mask], dim=0)
                        padded_hist.append(histories)
                        padded_mask.append(agent_mask)
                    
                    hist_ctx_batch = torch.stack(padded_hist, dim=0).to(self.device)  # (B, K, H, 7)
                    hist_mask_batch = torch.stack(padded_mask, dim=0).to(self.device)  # (B, K)
                
                if cfg.TRAINER.get('ONLINE_TRACKING', False):
                    model_output, state_prior, _ = self.model(y_training_batch[:, :, t],
                                                              det_context=det_ctx_batch, det_mask=det_mask_batch,
                                                              hist_context=hist_ctx_batch, hist_mask=hist_mask_batch)
                else:
                    model_output, state_prior, _ = self.model(y_training_batch[:, :, t], y_training_batch,
                                                              det_context=det_ctx_batch, det_mask=det_mask_batch,
                                                              hist_context=hist_ctx_batch, hist_mask=hist_mask_batch)
                x_out_training_batch[:, :, t] = model_output.squeeze(-1)
                state_out_training_batch[:, :, t] = state_prior.squeeze(-1)
            attributes = ['x', 'y', 'z', 'l', 'w', 'h', 'ry']
            if self.state_size == 11:
                attributes += ['bb1', 'bb2', 'bb3', 'bb4']
            losses = self._calculate_losses(train_target_batch, state_out_training_batch, x_out_training_batch, attributes)
            total_loss = sum(losses.values())
            self.MSE_train_linear_epoch[ti] = total_loss.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])
            torch.autograd.set_detect_anomaly(True)
            for param in self.model.parameters():
                assert not torch.isnan(param).any(), "Model parameter has NaN values."
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.8)
            self.optimizer.step()
            # Validation
            self.model.eval()
            self.model.batch_size = self.N_CV
            self.model.LKF_model.batch_size = self.N_CV
            self.model.init_hidden_LKF()
            with torch.no_grad():
                self.ss_model.T = self.T_test
                y_cv_batch = torch.empty((self.N_CV, self.state_size, self.ss_model.T))
                cv_target_batch = torch.empty((self.N_CV, self.state_size, self.ss_model.T))
                x_out_cv_batch = torch.empty((self.N_CV, self.state_size, self.ss_model.T))
                state_out_cv_batch = torch.empty((self.N_CV, self.state_size, self.ss_model.T))
                initializer_cv = torch.empty((self.N_CV, self.state_size, 1))
                for index in range(self.N_CV):
                    cv_elem = cv_input[index][0]
                    for idx, elem in enumerate(cv_elem):
                        y_cv_batch[index, :, idx] = elem
                        if idx == 0:
                            initializer_cv[index, :, 0] = elem
                    for idx, elem in enumerate(cv_target[index][0]):
                        cv_target_batch[index, :, idx] = elem
                self.model.LKF_model.InitSequence(initializer_cv, self.ss_model.T)
                for t in range(self.ss_model.T):
                    det_ctx_batch = None
                    det_mask_batch = None
                    hist_ctx_batch = None
                    hist_mask_batch = None
                    
                    use_ctx = getattr(cfg, 'CONTEXT', None)
                    if bool(getattr(use_ctx, 'USE_CONTEXT', False)) and cv_context is not None:
                        per_sample_ctx = []
                        max_n = 0
                        for index in range(self.N_CV):
                            try:
                                frames = cv_context[index]
                                ctx = frames[t] if t < len(frames) else None
                                if ctx is None or (torch.is_tensor(ctx) and ctx.numel() == 0):
                                    ctx = torch.zeros((0, 8), dtype=torch.float32)
                                elif not torch.is_tensor(ctx):
                                    ctx = torch.tensor(ctx, dtype=torch.float32)
                            except Exception:
                                ctx = torch.zeros((0, 8), dtype=torch.float32)
                            per_sample_ctx.append(ctx)
                            max_n = max(max_n, ctx.size(0) if ctx.dim() > 0 else 0)
                        
                        if max_n > 0:
                            padded = []
                            mask = torch.zeros((self.N_CV, max_n), dtype=torch.bool)
                            for ii, ctx in enumerate(per_sample_ctx):
                                n_ctx = ctx.size(0) if ctx.dim() > 0 and ctx.numel() > 0 else 0
                                if n_ctx < max_n:
                                    pad = torch.zeros((max_n - n_ctx, 8), dtype=torch.float32)
                                    if n_ctx > 0:
                                        ctx = torch.cat([ctx, pad], dim=0)
                                    else:
                                        ctx = pad
                                    mask[ii, n_ctx:] = True  # Mask padding positions
                                padded.append(ctx)
                            det_ctx_batch = torch.stack(padded, dim=0).to(self.device)
                            det_mask_batch = mask.to(self.device)
                    
                    # Optional history context for validation
                    if bool(getattr(use_ctx, 'USE_HISTORY', False)) and cv_history is not None:
                        per_sample_hist = []
                        per_sample_hist_mask = []
                        for index in range(self.N_CV):
                            try:
                                hist_dict = cv_history[index]
                                if hist_dict and 'history_per_frame' in hist_dict:
                                    frame_hist = hist_dict['history_per_frame'][t]
                                    histories = torch.tensor(frame_hist['histories'], dtype=torch.float32)
                                    valid_masks = torch.tensor(frame_hist['valid_masks'], dtype=torch.bool)
                                    agent_mask = ~valid_masks.any(dim=1)
                                else:
                                    histories = torch.zeros((1, 5, 7), dtype=torch.float32)
                                    agent_mask = torch.ones(1, dtype=torch.bool)
                            except Exception:
                                histories = torch.zeros((1, 5, 7), dtype=torch.float32)
                                agent_mask = torch.ones(1, dtype=torch.bool)
                            per_sample_hist.append(histories)
                            per_sample_hist_mask.append(agent_mask)
                        
                        max_k = max(h.size(0) for h in per_sample_hist)
                        H = per_sample_hist[0].size(1)
                        padded_hist = []
                        padded_mask = []
                        for histories, agent_mask in zip(per_sample_hist, per_sample_hist_mask):
                            K = histories.size(0)
                            if K < max_k:
                                pad_hist = torch.zeros((max_k - K, H, 7), dtype=histories.dtype)
                                histories = torch.cat([histories, pad_hist], dim=0)
                                pad_mask = torch.ones(max_k - K, dtype=torch.bool)
                                agent_mask = torch.cat([agent_mask, pad_mask], dim=0)
                            padded_hist.append(histories)
                            padded_mask.append(agent_mask)
                        
                        hist_ctx_batch = torch.stack(padded_hist, dim=0).to(self.device)
                        hist_mask_batch = torch.stack(padded_mask, dim=0).to(self.device)
                    
                    if cfg.TRAINER.get('ONLINE_TRACKING', False):
                        model_output, state_prior, _ = self.model(y_cv_batch[:, :, t],
                                                                  det_context=det_ctx_batch, det_mask=det_mask_batch,
                                                                  hist_context=hist_ctx_batch, hist_mask=hist_mask_batch)
                    else:
                        model_output, state_prior, _ = self.model(y_cv_batch[:, :, t], y_cv_batch,
                                                                  det_context=det_ctx_batch, det_mask=det_mask_batch,
                                                                  hist_context=hist_ctx_batch, hist_mask=hist_mask_batch)
                    x_out_cv_batch[:, :, t] = model_output.squeeze(-1)
                    state_out_cv_batch[:, :, t] = state_prior.squeeze(-1)
                losses_cv = self._calculate_losses(cv_target_batch, state_out_cv_batch, x_out_cv_batch, attributes)
                total_loss_cv = sum(losses_cv.values())
                self.MSE_cv_linear_epoch[ti] = total_loss_cv.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                if self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt:
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti
                    self.model.batch_size = 5000
                    self.model.LKF_model.batch_size = 5000
                    root_path = self.cfg.DATASET.ROOT
                    last_model_path = self.cfg.MODEL.SAVE_PATH
                    save_dir = os.path.join(root_path, last_model_path)
                    # Ensure save_dir is a directory, not a file path
                    if os.path.splitext(save_dir)[1]:
                        save_dir = os.path.dirname(save_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = os.path.join(save_dir, 'hybridtrack.pt')
                    torch.save(self.model, model_path)
            self.scheduler.step(self.MSE_cv_dB_opt)
            logging.info(f'Epoch {ti}: MSE Train: {self.MSE_train_dB_epoch[ti]:.4f} dB, MSE Val: {self.MSE_cv_dB_epoch[ti]:.4f} dB')
            if ti > 1:
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                logging.info(f"Delta MSE Train: {d_train:.4f} dB, Delta MSE Val: {d_cv:.4f} dB")
            logging.info(f"Optimal idx: {self.MSE_cv_idx_opt}, Optimal: {self.MSE_cv_dB_opt:.4f} dB")
            if ti % 100 == 0:
                torch.save(self.model, os.path.join(path_results_weight_per_epoch, f'hybridtrack_epoch_{ti}.pt'))
                # Evaluation
                x_out_cv_batch_ = x_out_cv_batch.permute(0, 2, 1).reshape(self.ss_model.T * self.N_CV, self.state_size)
                cv_target_batch_ = cv_target_batch.permute(0, 2, 1).reshape(self.ss_model.T * self.N_CV, self.state_size)
                state_out_cv_batch_ = state_out_cv_batch.permute(0, 2, 1).reshape(self.ss_model.T * self.N_CV, self.state_size)
                out_info = f'===== Target {len(cv_target_batch_)} vehicles\n'
                out_info += '====================== Target Domain performance =====================\n'
                out_info += evaluate_trans(np.array(x_out_cv_batch_[:, [0, 1, 2]]), np.array(cv_target_batch_[:, [0, 1, 2]]), mode='XYZ')
                acc, mederr, _ = evaluate_heading(np.array(x_out_cv_batch_[:, [6]]), np.array(cv_target_batch_[:, [6]]), n=30, only_verbose=0, details=0)
                out_info += f'Accuracy: {acc * 100.:.2f}%, Median Error: {mederr:.2f} degrees'
                logging.info(out_info)
        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel_test, cv_input, cv_target, path_results, path_results_weight, dataset_part, cfg,
               MaskOnState=False, randomInit=False, test_init=None, load_model=False, load_model_path=None, test_lengthMask=None):
        """
        Test the LKF model.
        """
        log_path = os.path.join(path_results, "logs")
        os.makedirs(log_path, exist_ok=True)
        if load_model:
            self.model = torch.load(load_model_path, map_location=self.device)
        else:
            self.model = torch.load(os.path.join(path_results_weight, 'hybridtrack.pt'), map_location=self.device)
        self.N_CV = cfg.TRAINER.N_TEST
        SysModel_test.T_test = self.T_test
        self.state_size = 7
        self.model.eval()
        self.model.batch_size = self.N_CV
        self.model.LKF_model.batch_size = self.N_CV
        self.model.init_hidden_LKF()
        with torch.no_grad():
            self.ss_model.T = self.T_test
            y_cv_batch = torch.empty((self.N_CV, self.state_size, self.ss_model.T))
            cv_target_batch = torch.empty((self.N_CV, self.state_size, self.ss_model.T))
            x_out_cv_batch = torch.empty((self.N_CV, self.state_size, self.ss_model.T))
            state_out_cv_batch = torch.empty((self.N_CV, self.state_size, self.ss_model.T))
            initializer_cv = torch.empty((self.N_CV, self.state_size, 1))
            for index in range(self.N_CV):
                cv_elem = cv_input[index][0]
                for idx, elem in enumerate(cv_elem):
                    y_cv_batch[index, :, idx] = elem
                    if idx == 0:
                        initializer_cv[index, :, 0] = elem
                for idx, elem in enumerate(cv_target[index][0]):
                    cv_target_batch[index, :, idx] = elem
            self.model.LKF_model.InitSequence(initializer_cv, self.ss_model.T)
            for t in range(self.ss_model.T):
                if cfg.TRAINER.get('ONLINE_TRACKING', False):
                    model_output, state_prior, _ = self.model(y_cv_batch[:, :, t])
                else:
                    model_output, state_prior, _ = self.model(y_cv_batch[:, :, t], y_cv_batch)
                x_out_cv_batch[:, :, t] = model_output.squeeze(-1)
                state_out_cv_batch[:, :, t] = state_prior.squeeze(-1)
            plot_all_batches(state_out_cv_batch, x_out_cv_batch, y_cv_batch)
            attributes = ['x', 'y', 'z', 'l', 'w', 'h', 'ry']
            if self.state_size == 11:
                attributes += ['bb1', 'bb2', 'bb3', 'bb4']
            losses_cv = self._calculate_losses(cv_target_batch, state_out_cv_batch, x_out_cv_batch, attributes)
        print(losses_cv)
        # Evaluation
        x_out_cv_batch_ = x_out_cv_batch.permute(0, 2, 1).reshape(self.ss_model.T * self.N_CV, self.state_size)
        cv_target_batch_ = cv_target_batch.permute(0, 2, 1).reshape(self.ss_model.T * self.N_CV, self.state_size)
        y_cv_batch_ = y_cv_batch.permute(0, 2, 1).reshape(self.ss_model.T * self.N_CV, self.state_size)
        state_out_cv_batch_ = state_out_cv_batch.permute(0, 2, 1).reshape(self.ss_model.T * self.N_CV, self.state_size)
        for batch, name in zip([y_cv_batch_, x_out_cv_batch_, state_out_cv_batch_], ['y_cv', 'x_out', 'state_out']):
            out_info = f'===== Target {len(cv_target_batch_)} vehicles\n'
            out_info += '====================== Target Domain performance =====================\n'
            out_info += evaluate_trans(np.array(batch[:, [0, 1, 2]]), np.array(cv_target_batch_[:, [0, 1, 2]]), mode='XYZ')
            acc, mederr, _ = evaluate_heading(np.array(batch[:, [6]]), np.array(cv_target_batch_[:, [6]]), n=30, only_verbose=0, details=0)
            out_info += f'Accuracy: {acc * 100.:.2f}%, Median Error: {mederr:.2f} degrees'
            acc10, mederr10, _ = evaluate_heading(np.array(batch[:, [6]]), np.array(cv_target_batch_[:, [6]]), n=10, only_verbose=0, details=0)
            out_info += f'Accuracy: {acc10 * 100.:.2f}%, Median Error: {mederr10:.2f} degrees'
            print(out_info)
        return [0, 0, 0, 0]


