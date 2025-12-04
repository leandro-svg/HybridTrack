# Solving the Context Problem: Training-Inference Gap

## Root Cause Analysis

You're **absolutely right** about the problem:

### Baseline (No Context)
- Input: Ego's own trajectory (position, velocity from Kalman filter)
- Works well because: Self-motion is smooth, predictable, well-modeled by learned Kalman gains

### Context Addition (Failed)
- **Training**: Uses ground truth positions of nearby cars (clean, accurate)
- **Inference**: Uses noisy detections + predictions (missing frames, drift, false positives)
- **Result**: Model learned to use perfect context, gets garbage context at test time

## Why This is Fundamental

KITTI characteristics:
1. **Highway driving**: Cars don't interact much (parallel lanes, sparse)
2. **Independent motion**: Each car follows its own trajectory
3. **Sparse scenarios**: 2-5 cars per frame typically
4. **Detection quality**: VirConv detections are good but not perfect

Your baseline already captures:
- ✅ Ego motion dynamics (velocity, acceleration)
- ✅ Measurement noise patterns
- ✅ Kalman gain adaptation

Context COULD help with:
- Lane change prediction (if car nearby changes lanes)
- Occlusion handling (if car blocks view)
- Collision avoidance (if cars interact)

But KITTI has **minimal** of these cases!

---

## Solution 1: Train with Realistic Noise (RECOMMENDED)

Make training context match inference conditions.

### A. Add Dropout to Context During Training

**File**: `src/dataset/training_dataset.py`

Add method to corrupt context:
```python
def corrupt_detection_context(self, det_context: np.ndarray, corruption_prob: float = 0.3) -> np.ndarray:
    \"\"\"
    Corrupt detection context to simulate tracking conditions.
    
    Args:
        det_context: (N, 8) array of detection context
        corruption_prob: Probability of dropping/corrupting each detection
    
    Returns:
        Corrupted context array
    \"\"\"
    if det_context is None or len(det_context) == 0:
        return det_context
    
    corrupted = det_context.copy()
    N = len(corrupted)
    
    # 1. Random dropout (simulates missed detections)
    dropout_mask = np.random.rand(N) > corruption_prob
    corrupted = corrupted[dropout_mask]
    
    if len(corrupted) == 0:
        return np.zeros((0, 8))
    
    # 2. Add position noise (simulates detection errors)
    pos_noise = np.random.randn(len(corrupted), 3) * 0.2  # 20cm std
    corrupted[:, :3] += pos_noise
    
    # 3. Add size noise
    size_noise = np.random.randn(len(corrupted), 3) * 0.1  # 10cm std
    corrupted[:, 3:6] += size_noise
    
    # 4. Add rotation noise
    rot_noise = np.random.randn(len(corrupted)) * 0.05  # ~3 degrees
    corrupted[:, 6] += rot_noise
    
    # 5. Corrupt scores (simulates confidence variation)
    corrupted[:, 7] *= (1 + np.random.randn(len(corrupted)) * 0.1)
    corrupted[:, 7] = np.clip(corrupted[:, 7], 0.0, 1.0)
    
    return corrupted
```

Then in `_extract_detection_context_sequence`, apply corruption:
```python
def _extract_detection_context_sequence(self, track_id_key: str, start_idx: int, seq_len: int, 
                                        pose_matrices: np.ndarray, ego_positions: np.ndarray) -> Dict[str, Any]:
    # ... existing code ...
    
    # NEW: Apply corruption if training mode
    if self.mode == 'train':
        for frame_idx in det_ctx_frames.keys():
            det_ctx_frames[frame_idx] = self.corrupt_detection_context(
                det_ctx_frames[frame_idx], 
                corruption_prob=0.3  # Drop 30% of context
            )
    
    return {"frames": det_ctx_frames, "masks": det_ctx_masks}
```

### B. Add to Config

**File**: `src/configs/training.yaml`
```yaml
CONTEXT:
  # ... existing ...
  TRAIN_NOISE:
    ENABLE: true
    DROPOUT_PROB: 0.3      # 30% chance to drop each detection
    POS_NOISE_STD: 0.2     # 20cm position noise
    SIZE_NOISE_STD: 0.1    # 10cm size noise  
    ROT_NOISE_STD: 0.05    # ~3 degree rotation noise
    SCORE_NOISE_STD: 0.1   # 10% score variation
```

---

## Solution 2: Use Relative Motion (Better Features)

Current context: `[x, y, z, l, w, h, ry, score]` - just positions

Better context: Add **relative motion** features!

### Enhance Context Format

**File**: `src/tracker/hybridtrack.py` in `_prepare_detection_context`:

Change from:
```python
ctx_entry = torch.tensor([dx, dy, dz, l, w, h, dry, score])
```

To:
```python
# Get velocities (if available from previous frame)
other_prev_ts = sorted([t for t in other_traj.trajectory.keys() if t < self.current_timestamp])
if len(other_prev_ts) >= 2:
    # Compute velocity from last 2 positions
    prev_pos = other_traj.trajectory[other_prev_ts[-1]].updated_state[:3]
    prev_prev_pos = other_traj.trajectory[other_prev_ts[-2]].updated_state[:3]
    velocity = (prev_pos - prev_prev_pos).cpu()  # (vx, vy, vz)
    
    # Relative velocity (how fast other car moves relative to ego)
    ego_prev_pos = ego_traj.trajectory[other_prev_ts[-1]].updated_state[:3]
    ego_prev_prev_pos = ego_traj.trajectory[other_prev_ts[-2]].updated_state[:3]
    ego_velocity = (ego_prev_pos - ego_prev_prev_pos).cpu()
    
    rel_velocity = velocity - ego_velocity
    dvx, dvy, dvz = rel_velocity[0], rel_velocity[1], rel_velocity[2]
else:
    dvx, dvy, dvz = 0.0, 0.0, 0.0

# New 11-dim context: [dx, dy, dz, l, w, h, ry, score, dvx, dvy, dvz]
ctx_entry = torch.tensor([dx, dy, dz, l, w, h, dry, score, dvx, dvy, dvz])
```

Update encoder:
```python
# In LearnableKF.py
self.det_enc = DetectionContextEncoder(in_dim=11, embed_dim=ctx_embed)  # Was 8
```

**Why this helps:**
- Relative velocity tells you if car is approaching/diverging
- Predicts lane changes (lateral velocity)
- Detects overtaking (longitudinal velocity difference)

---

## Solution 3: Selective Context (Only When Needed)

Don't use context always - only when it **actually helps**.

### Use Context Only for Ambiguous Cases

**File**: `src/model/LearnableKF.py`

Add adaptive gating based on **prediction uncertainty**:

```python
def step_prior(self, sequence=None, det_context=None, det_mask=None, ...):
    # ... existing prediction ...
    
    if self.use_context and det_context is not None:
        # Compute prediction uncertainty (from covariance)
        pred_uncertainty = torch.diagonal(self.Sigma, dim1=-2, dim2=-1).mean(dim=-1)  # (B,)
        
        # Only use context when uncertain (high covariance)
        uncertainty_threshold = 0.5  # Tune this
        use_ctx_mask = pred_uncertainty > uncertainty_threshold  # (B,)
        
        # Encode context as before
        det_kv = self.det_enc(det_context)
        # ... cross attention ...
        
        # Apply context only to uncertain predictions
        gate_term = self.ctx_alpha * ctx_proj
        gate_term = gate_term * use_ctx_mask.unsqueeze(-1).float()  # Zero out for certain predictions
        
        fused_feats = fused_feats + gate_term
```

**Why this helps:**
- Low uncertainty → trust baseline (no context needed)
- High uncertainty → use context (ambiguous cases: occlusion, multi-object)
- Prevents context from hurting good predictions

---

## Solution 4: Context as Constraint (Not Feature)

Instead of adding context to features, use it as a **soft constraint**.

### Collision Avoidance Loss

**File**: `src/tools/training.py`

Add auxiliary loss during training:

```python
def compute_collision_loss(self, predictions, det_context, det_mask):
    \"\"\"
    Penalize predictions that would collide with context vehicles.
    \"\"\"
    # predictions: (B, 7) - ego predicted position
    # det_context: (B, N, 8) - nearby vehicles
    
    ego_pos = predictions[:, :3]  # (B, 3)
    ctx_pos = det_context[:, :, :3]  # (B, N, 3)
    
    # Compute distances
    distances = torch.norm(
        ego_pos.unsqueeze(1) - ctx_pos,  # (B, 1, 3) - (B, N, 3) = (B, N, 3)
        dim=-1
    )  # (B, N)
    
    # Mask invalid context
    distances = distances.masked_fill(det_mask, float('inf'))
    
    # Penalize if ego too close to any car
    min_dist = distances.min(dim=1)[0]  # (B,)
    collision_threshold = 2.0  # 2 meters minimum distance
    
    collision_loss = F.relu(collision_threshold - min_dist).mean()
    return collision_loss

# In training loop:
total_loss = mse_loss + 0.1 * collision_loss  # Small weight
```

---

## Solution 5: Completely Different Approach - Graph Neural Network

Current: Context → MLP → Cross-attention → Fusion

Better: Model scene as graph!

### Use GNN for Multi-Agent Modeling

```python
class SceneGraphEncoder(nn.Module):
    def __init__(self, node_dim=8, edge_dim=4, hidden_dim=64):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        # Message passing
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # [node_i, node_j, edge_ij]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_net = nn.GRU(hidden_dim, hidden_dim)
    
    def forward(self, nodes, edges, edge_index):
        # nodes: (B, N, node_dim) - all vehicles
        # edges: (B, E, edge_dim) - relative distances/angles
        # edge_index: (B, E, 2) - connectivity
        
        h_nodes = self.node_encoder(nodes)  # (B, N, hidden_dim)
        h_edges = self.edge_encoder(edges)  # (B, E, hidden_dim)
        
        # Message passing
        for _ in range(3):  # 3 message passing steps
            messages = []
            for e in range(edge_index.shape[1]):
                i, j = edge_index[:, e, 0], edge_index[:, e, 1]
                h_i = h_nodes[torch.arange(h_nodes.size(0)), i]
                h_j = h_nodes[torch.arange(h_nodes.size(0)), j]
                h_ij = h_edges[:, e]
                
                msg = self.message_net(torch.cat([h_i, h_j, h_ij], dim=-1))
                messages.append(msg)
            
            # Aggregate and update
            aggregated = torch.stack(messages, dim=1).mean(dim=1)
            h_nodes, _ = self.update_net(aggregated.unsqueeze(1), h_nodes.transpose(0, 1))
            h_nodes = h_nodes.transpose(0, 1)
        
        return h_nodes[:, 0]  # Return ego node encoding
```

---

## My Recommendation

**Try Solution 1 + Solution 2 together:**

1. **Add noise to training context** (30% dropout, position noise)
2. **Add relative velocity features** (11-dim instead of 8-dim)
3. **Retrain model**

This addresses:
- ✅ Training-inference gap (realistic noise)
- ✅ Better features (motion, not just position)
- ✅ Minimal code changes

Expected improvement: **+0.5-1.5% HOTA**

Want me to implement this?
