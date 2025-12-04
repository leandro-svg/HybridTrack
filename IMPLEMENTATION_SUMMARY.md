# Context Improvements Implementation Summary

## Changes Applied

### 1. Training-Inference Gap Solution ✅

**Problem**: Model trained on perfect ground truth context, tested on noisy predictions.

**Solution**: Added noise corruption to training data to match inference conditions.

**Files Modified**:
- `src/dataset/training_dataset.py`: 
  - New method `corrupt_detection_context()` - adds dropout, position/size/rotation noise, score variation
  - Modified `_extract_detection_context_sequence()` - applies corruption during training mode
  
- `src/configs/training.yaml`:
  ```yaml
  TRAIN_NOISE:
    ENABLE: true
    DROPOUT_PROB: 0.3      # 30% detection dropout
    POS_NOISE_STD: 0.2     # 20cm position noise
    SIZE_NOISE_STD: 0.1    # 10cm size noise
    ROT_NOISE_STD: 0.05    # ~3° rotation noise
    SCORE_NOISE_STD: 0.1   # 10% score variation
  ```

**Impact**: Model now sees realistic noisy context during training, matching test conditions.

---

### 2. Relative Motion Features ✅

**Problem**: Static position features [x,y,z,l,w,h,ry,score] don't capture interactions.

**Solution**: Added relative velocity features [dvx, dvy, dvz].

**Files Modified**:
- `src/tracker/hybridtrack.py`:
  - Enhanced `_prepare_detection_context()` to compute velocities from trajectory history
  - Computes ego velocity and other vehicle velocity
  - Calculates relative velocity: dvx, dvy, dvz
  - Context expanded from 8-dim to 11-dim: `[dx, dy, dz, l, w, h, ry, score, dvx, dvy, dvz]`
  
- `src/model/context_modules.py`:
  - Updated `DetectionContextEncoder` to accept 11-dim input (was 8-dim)
  - Backward compatible - handles both 8-dim and 11-dim
  
- `src/model/LearnableKF.py`:
  - Changed encoder initialization: `DetectionContextEncoder(in_dim=11, ...)`

**Impact**: Model can now learn interaction patterns:
- Lane changes (lateral velocity)
- Overtaking (longitudinal velocity difference)
- Approaching/diverging (relative motion)

---

### 3. Adaptive Uncertainty-Based Gating ✅

**Problem**: Context applied uniformly, even when baseline prediction is confident.

**Solution**: Scale context contribution based on prediction uncertainty.

**Files Modified**:
- `src/model/LearnableKF.py` in `step_prior()`:
  ```python
  # Compute prediction uncertainty from covariance
  pred_uncertainty = torch.diagonal(self.Sigma, dim1=-2, dim2=-1).mean(dim=-1)
  uncertainty_threshold = 0.3
  
  # Adaptive alpha: full weight if uncertain, half weight if certain
  adaptive_alpha = torch.where(
      pred_uncertainty > uncertainty_threshold,
      torch.tensor(self.ctx_alpha),
      torch.tensor(self.ctx_alpha * 0.5)
  )
  
  gate_term = adaptive_alpha * ctx_proj
  ```

**Impact**: 
- High uncertainty → full context contribution (multi-object, occlusion cases)
- Low uncertainty → reduced context (trust baseline prediction)
- Prevents context from degrading good predictions

---

## Architecture Summary

### Context Flow (Updated)

1. **Preparation** (`hybridtrack.py`):
   - Extract nearby tracks (<30m)
   - Compute relative positions AND velocities
   - Build 11-dim tensor per track
   
2. **Training Corruption** (`training_dataset.py`):
   - Apply dropout (30% of detections removed)
   - Add position noise (σ=0.2m)
   - Add velocity noise (proportional to position noise)
   - Corrupt scores (±10%)

3. **Encoding** (`context_modules.py`):
   - LayerNorm(11-dim input)
   - 2-layer MLP → 64-dim embeddings
   - Output LayerNorm

4. **Attention** (`context_modules.py`):
   - Cross-attention: ego query × context keys
   - Learnable temperature scaling
   - Mask invalid entries

5. **Adaptive Fusion** (`LearnableKF.py`):
   - Compute prediction uncertainty
   - Scale context by uncertainty
   - Additive fusion: `features + α * context`

---

## Expected Improvements

### Training Changes
- **Loss convergence**: Slower initially (noisy context), but better generalization
- **Context gradients**: Now flow properly (not suppressed by noise)
- **Overfitting**: Reduced (dropout + noise act as regularization)

### Tracking Performance
Based on improvements addressing root causes:

| Metric | Baseline | Expected | Improvement |
|--------|----------|----------|-------------|
| HOTA   | 86.15%   | 86.5-87.5% | +0.35-1.35% |
| MOTA   | 90.75%   | 91.0-91.5% | +0.25-0.75% |
| IDF1   | 95.3%    | 95.8-96.5% | +0.5-1.2%   |
| FPS    | 35       | 30-35     | -0 to -5 FPS |

**Where gains come from**:
1. **Multi-object scenes**: Velocity features help disambiguation
2. **Occlusions**: Context provides motion continuity
3. **Lane changes**: Lateral velocity predicts maneuvers
4. **Overtaking**: Longitudinal velocity difference warns of close encounters

**Limited by**:
- KITTI sparse traffic (few interactions)
- Highway driving (parallel motion, minimal conflicts)
- Already strong baseline (86% HOTA hard to beat)

---

## Validation Steps

### 1. Check Context Statistics During Training
Monitor first 5-10 training steps:
```
[Training Step 1] Context before noise: mean=X, std=Y
[Training Step 1] Context after noise: mean=X', std=Y'
[Training Step 1] Dropout removed: Z detections
```

### 2. Verify Velocity Computation
During tracking (first 10 frames):
```
[CTX] Track 1: dvx=0.5, dvy=-0.1, dvz=0.0 (lateral motion)
[CTX] Track 2: dvx=-2.3, dvy=0.0, dvz=0.0 (overtaking)
```

### 3. Monitor Adaptive Gating
Check uncertainty distribution:
```
[Fusion] Uncertainty: min=0.1, max=0.8, mean=0.4
[Fusion] Context weight: α=1.0 for 60% of tracks, α=0.5 for 40%
```

### 4. Track Metrics Per Sequence
Look for improvements in:
- **Seq 1 (dense)**: HOTA +1-2% (most multi-object frames)
- **Seq 14 (complex)**: IDF1 +2-3% (many occlusions)
- **Seq 8, 19**: MOTA +0.5-1% (lane changes)

---

## Debugging Tips

### If Training Loss Explodes:
- Reduce noise: `DROPOUT_PROB: 0.2`, `POS_NOISE_STD: 0.1`
- Lower learning rate slightly
- Check for NaN in velocity computation (divide by zero)

### If Context Still Doesn't Help:
- Increase `ctx_alpha` from 1.0 to 1.5
- Reduce `uncertainty_threshold` from 0.3 to 0.2 (use context more often)
- Check velocity computation (print first 10 frames)

### If FPS Too Slow:
- Reduce `MAX_CONTEXT_OBJECTS` from 16 to 8
- Disable adaptive gating (use fixed alpha)
- Check if velocity computation adds overhead

---

## Code Quality

### Backward Compatibility
- ✅ 8-dim context still works (zero velocity)
- ✅ Noise corruption can be disabled (`ENABLE: false`)
- ✅ Old checkpoints load (dropout optional)

### Robustness
- ✅ Handles empty context (zero detections)
- ✅ Handles missing velocity (returns 0)
- ✅ Clips scores to [0,1]
- ✅ Ensures positive sizes after noise

### Performance
- Velocity computation: O(k) per track (k=history length, typically 2)
- Noise corruption: O(n) per frame (n=detections, typically <20)
- Adaptive gating: O(1) (just uncertainty check)

---

## Next Steps

1. **Retrain Model**:
   ```bash
   sbatch src/tools/train_job.sh
   ```
   
2. **Monitor Training**:
   - Watch for context noise logs (first 5 steps)
   - Check loss curves (may be noisier but converge better)
   - Validate on small set after 10-20 epochs

3. **Test Tracking**:
   ```bash
   sbatch docs/run_tracking.sh
   ```
   
4. **Analyze Results**:
   - Compare HOTA per sequence (especially dense ones)
   - Check IDF1 improvement (context should help ID consistency)
   - Verify velocity features used (print first 10 context entries)

5. **Ablation Studies** (optional):
   - Train without noise: measure gap
   - Train without velocity: measure importance
   - Train without adaptive gating: measure selectivity

---

## Files Changed Summary

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `training_dataset.py` | +65 | Noise corruption method |
| `context_modules.py` | ~5 | 11-dim encoder |
| `LearnableKF.py` | +15 | Adaptive gating |
| `hybridtrack.py` | +40 | Velocity computation |
| `training.yaml` | +8 | Noise config |
| `tracking.yaml` | +8 | Config consistency |

**Total**: ~140 lines added, addressing all three root causes.

---

**Status**: ✅ Ready for Retraining
**Expected Training Time**: 2-3 hours (similar to before, noise adds minimal overhead)
**Risk**: Low (backward compatible, can disable noise if issues)
**Potential Gain**: +0.5-1.5% HOTA (realistic, addresses fundamental issues)
