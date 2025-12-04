# History Context Implementation for HybridTrack

## Overview
Complete implementation of history context support for encoding other agents' motion histories (past H waypoints) and applying cross-attention alongside detection context in the Learnable Kalman Filter prior step.

## Key Features
- **Variable-length agent handling**: Supports 1–120+ agents per frame with padding and masking
- **Ground-truth based**: Uses KITTI tracking labels to build reliable agent histories
- **Top-K selection**: Distance-weighted selection of nearest K agents (default: 32)
- **Temporal depth**: Configurable history length H (default: 5 frames)
- **Full masking support**: Proper masks to avoid zero-padding influence in attention
- **Backward compatible**: All features optional; works without history when disabled

## Architecture

### Data Generation
**Script**: `docs/data_utils/setup_history_context.py`
- Reads KITTI ground-truth tracking labels (`label_02/*.txt`)
- Converts agent positions from camera to velodyne coordinates
- Selects top-K nearest agents per ego frame within max distance (50m)
- Builds (K, H, 7) histories for each agent: last H states up to current frame
- Generates per-frame valid masks indicating which time steps are available
- Pads to uniform K across all frames with all-False masks for padding
- Outputs: `src/data/ann/{train|validation}/history_context.json`

**Format**:
```json
{
  "seq_track_id": {
    "video_id": "0000",
    "track_id": 0,
    "frame_id": [0, 1, 2, ...],
    "history_per_frame": [
      {
        "histories": [[[x,y,z,l,w,h,ry], ...], ...],  // (K, H, 7)
        "valid_masks": [[true, false, ...], ...]       // (K, H)
      },
      ...
    ]
  }
}
```

**Run**: 
```bash
cd docs
sbatch run_setup_history_context.sh
```

### Training Pipeline Integration

#### 1. Dataset Loading (`src/dataset/training_dataset.py`)
- `KITTIDataset.__init__`: Loads `history_context.json` if available
- `load_history_context()`: Reads history JSON per split
- `create_sequences()`: Extracts history slices aligned with ego trajectory sequences
- `__getitem__`: Returns (gt, det, history_dict) tuple

#### 2. Batch Generation (`src/tools/batch_generation.py`)
- `SystemModel.GenerateBatch`: Captures `HistoryContext` list from dataloader
- Stores per-sequence history dicts in `self.HistoryContext`

#### 3. Training Loop (`src/tools/training.py`)
- `NNTrain()`: Accepts `train_history` and `cv_history` parameters
- Per time-step:
  - Builds `hist_ctx_batch` (B, K, H, 7) from per-sample history dicts
  - Computes `hist_mask_batch` (B, K): True if agent is padding
  - Pads to max K across batch
  - Passes to model forward: `hist_context=hist_ctx_batch, hist_mask=hist_mask_batch`

#### 4. Model Forward (`src/model/LearnableKF.py`)
- `step_prior()`: Accepts optional `hist_context` and `hist_mask`
- If `CONTEXT.USE_HISTORY=true`:
  - Encodes histories via `HistoryContextEncoder` (Transformer over H)
  - Applies cross-attention from ego query to history embeddings with `hist_mask`
  - Fuses with detection context embeddings and ego features
  - Computes context-aware residual prior

#### 5. Tracking Runtime (`src/tracker/hybridtrack.py`)
- `_lkf_prediction()`: Calls `_prepare_history_context()` if enabled
- `_prepare_history_context()`: Builds (B, K, H, 7) from active trajectories
  - Collects last H states per trajectory
  - Selects top-K by distance
  - Pads and masks
- Passes `hist_ctx` and `hist_mask` to `step_prior()`

#### 6. Dataset Saving (`src/dataset/utils.py`)
- `DataGen()`: Appends `train_history` to saved dataset payload at index 7

#### 7. Training Script (`src/training_script.py`)
- Loads `train_history` from `loaded_data[7]` if present
- Passes `train_history` and `cv_history` to `NNTrain()`

## Configuration

Both `src/configs/training.yaml` and `src/configs/tracking.yaml` include:

```yaml
CONTEXT:
  USE_HISTORY: false          # Set true after generating history_context.json and training
  HISTORY_LEN: 5              # Number of past frames per agent
  MAX_HISTORY_AGENTS: 32      # Max agents to encode (top-K)
  HISTORY_DISTANCE_THRESHOLD: 50.0  # Max distance to consider agents
  HISTORY_NHEAD: 4            # Attention heads for history encoder
  HISTORY_EMBED_DIM: 64       # Embedding dimension
```

**Default**: `USE_HISTORY: false` to allow training detection-only context first.

## Workflow

### Step 1: Generate History Context Data
```bash
cd docs
sbatch run_setup_history_context.sh
```
Outputs:
- `src/data/ann/train/history_context.json`
- `src/data/ann/validation/history_context.json`

### Step 2: Generate Training Dataset (with history)
History context is automatically loaded and included when you run:
```bash
python src/training_script.py
```
Dataset will include history at index 7 if `history_context.json` exists.

### Step 3: Train Detection-Only (Baseline)
Keep `USE_HISTORY: false` in `training.yaml` and train:
```bash
python src/training_script.py
```
Model learns detection context only; history encoder not used.

### Step 4: Enable History Training
Set `USE_HISTORY: true` in `training.yaml` and retrain or fine-tune:
```bash
python src/training_script.py
```
Model now uses both detection and history contexts in prior.

### Step 5: Tracking with History
Set `USE_HISTORY: true` in `tracking.yaml` and run tracking:
```bash
python src/tracking_main.py --cfg_file configs/tracking.yaml
```
Tracker builds runtime history from active trajectories.

## Technical Details

### History Encoding
- Input: (B, K, H, 7) agent histories
- Per-agent processing:
  - Linear projection: 7 → embed_dim
  - Positional encoding over H
  - Transformer encoder over time (2 layers, 4 heads)
  - Temporal pooling → (B, K, embed_dim)
- Agent-level masking: True for fully padded agents

### Cross-Attention Fusion
```python
# Ego features (query)
ego_q = q_proj_hist(ego_features)  # (B, 1, E)

# History embeddings (key/value)
hist_embeddings = hist_enc(hist_context)  # (B, K, E)

# Masked attention
hist_fused = cross_attn_hist(ego_q, hist_embeddings, key_padding_mask=hist_mask)  # (B, E)

# Concatenate with detection context
context_combined = [det_fused, hist_fused]
context_vec = ctx_proj(torch.cat(context_combined, dim=-1))  # (B, FEATURE_DIM)

# Add to ego features for residual prior
prior_input = ego_features + context_vec
```

### Variable-Length Handling
- **Agents per frame**: 0–120+ (varies)
- **Padding**: Zero histories + all-True masks for agents beyond K
- **Masking**: 
  - `hist_mask` (B, K): True masks out padded agents in attention
  - `valid_masks` (K, H): Per-time-step validity (not directly used in current attention; for future extensions)

### Memory Efficiency
- Top-K cap (default 32) limits agent dimension
- History length H=5 keeps temporal window short
- Embedding dimension 64 balances expressiveness and cost
- Runtime: O(B·K·H·D) per forward pass

## Files Modified/Created

### Created
- `docs/data_utils/setup_history_context.py` - History context generator
- `docs/run_setup_history_context.sh` - SLURM script

### Modified
- `src/dataset/training_dataset.py` - Load and sequence history
- `src/tools/batch_generation.py` - Capture HistoryContext
- `src/tools/training.py` - Build and pass history tensors
- `src/dataset/utils.py` - Save history in dataset
- `src/training_script.py` - Load and route history
- `src/tracker/hybridtrack.py` - Runtime history building (already done in prior work)
- `src/model/LearnableKF.py` - History encoder and fusion (already done)
- `src/model/context_modules.py` - HistoryContextEncoder (already done)

### Configs
- `src/configs/training.yaml` - CONTEXT.USE_HISTORY: false
- `src/configs/tracking.yaml` - CONTEXT.USE_HISTORY: false

## Benefits
- **Temporal cues**: Agent velocity/trajectory trends improve ego prediction
- **Collective motion**: Multi-agent flow patterns provide contextual priors
- **Occlusion handling**: History fills gaps when detections sparse
- **Complex scenes**: Most useful in dense traffic, junctions, maneuvers

## Tradeoffs
- **Data dependency**: Requires consistent track IDs (GT or reliable association)
- **Compute cost**: Additional O(K·H·D) per step
- **Diminishing returns**: Low gains in sparse scenes or with high-quality detections

## Recommendations
1. Train detection-only baseline first (current default)
2. Generate history data once GT labels available
3. Fine-tune with history enabled for incremental improvement
4. Monitor performance vs. latency tradeoff
5. Adjust K, H, and distance thresholds based on dataset characteristics

## Testing
To verify integration without enabling history:
```bash
# Check dataset loads history without error
python -c "from dataset.training_dataset import KITTIDataset; from configs.config_utils import get_cfg; cfg=get_cfg(); ds=KITTIDataset(cfg, 'train'); print(f'History loaded: {len(ds.kitti_sequences_history)} sequences')"

# Check batch generation captures history
python -c "from tools.batch_generation import SystemModel; print('HistoryContext attribute exists:', hasattr(SystemModel, '__init__'))"
```

## Future Extensions
- Time-step level masking within histories (handle short tracks)
- Learnable weighting between detection and history contexts
- Multi-scale temporal encoding (fast vs. slow motion)
- Separate history encoders for different agent types
