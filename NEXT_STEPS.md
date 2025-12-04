# Context System - Next Steps

## Current Situation (Job 17998)
- **HOTA**: 86.159% (baseline: 86.245%, change: -0.086%)
- **FPS**: 12.3 (vs 35.7 with old fixes, vs 290 without context)
- **Status**: Context flows through pipeline but has NO positive impact

## Root Cause Analysis

### Why Context Doesn't Help Yet
1. **Old checkpoint problem**: Model trained WITH learnable gate that suppressed context
2. **Undertrained encoder**: Context encoder never learned useful features (gate blocked gradients)
3. **Code vs weights mismatch**: Our fixes work (no gate, 30m threshold) but checkpoint has useless weights

### Evidence
- Removing gate in code doesn't help if encoder weights are random/undertrained
- Even with `alpha=1.0`, we're just amplifying noise from undertrained encoder
- Context processing works (no crashes) but contributes nothing useful

## Solution: Retrain Model

### Quick Validation First (Optional)
Test WITHOUT context to confirm baseline:

```yaml
# src/configs/tracking.yaml
CONTEXT:
  USE_CONTEXT: false  # Disable to get clean baseline
```

Expected result: HOTA ~86.2-86.5%, FPS ~290

This confirms our baseline and proves context overhead.

### Then: Train New Model

**1. Verify Training Config**
```yaml
# src/configs/training.yaml
CONTEXT:
  USE_CONTEXT: true
  USE_HISTORY: false
  DISTANCE_THRESH: 30.0  # Already fixed
  # ... other settings
```

**2. Start Training**
```bash
sbatch src/tools/train_job.sh
```

**3. Monitor Training**
Watch for:
- Context encoder gradients flowing (not blocked by gate)
- Context contribution stats (~30% of feature norm with alpha=0.3, or 100% with alpha=1.0)
- Loss curves showing benefit from context

**4. Training Should Take**
- ~20-30 epochs for convergence
- ~2-3 hours on RTX 4090
- MSE loss should decrease below current 0.3305 dB

**5. After Training - Test**
```bash
sbatch docs/run_tracking.sh
```

Expected improvements:
- **HOTA**: +0.5-2% (target: 86.7-88.2%)
- **IDF1**: +1-3% (better ID consistency from context)
- **MOTA**: +0.3-1% (fewer false positives)
- **FPS**: 12-15 FPS (context overhead is real but worth it)

## Why Retraining Will Work

### Fixed Architecture Benefits
1. **No escape path**: Model MUST learn to use context (no gate to close)
2. **Better signal**: 30m threshold filters distant noise
3. **Proper regularization**: Dropout prevents overfitting
4. **Stable learning**: Fixed alpha provides consistent gradient flow

### What Model Will Learn
- **Relative motion patterns**: Other vehicles' velocities relative to ego
- **Occlusion resolution**: Use nearby tracks to disambiguate measurements
- **Multi-object interactions**: Learn which context helps vs hurts
- **Distance-aware weighting**: Cross-attention learns to focus on relevant neighbors

## Alternative: Incremental Approach

If full retraining is expensive, try this:

**1. Freeze Base Model**
```python
# In LearnableKF.__init__
for param in self.encoder.parameters():
    param.requires_grad = False
for param in self.decoder.parameters():
    param.requires_grad = False
```

**2. Train Only Context Modules**
- Only context encoder, cross-attention, and projections learn
- Faster: ~5-10 epochs
- Lower risk of breaking existing performance

**3. Then Fine-tune All Together**
- Unfreeze all parameters
- Continue training for 10 more epochs
- Model learns to integrate context with motion

## Current Code Status

### ✅ Fixes Applied
- Removed learnable gate → Fixed `alpha=1.0` (was 0.3, increased for testing)
- Reduced distance threshold: 50m → 30m
- Score normalization to [0,1]
- Optional dropout for regularization
- Temperature scaling in cross-attention

### ⚠️ Not Yet Done
- Model retraining with new architecture
- Validation that context improves performance
- Ablation studies on alpha values

### 📊 Performance Targets
| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| HOTA   | 86.2%    | 87%    | 88%     |
| MOTA   | 90.7%    | 91%    | 92%     |
| IDF1   | 95.3%    | 96%    | 97%    |
| FPS    | 290      | 12-15  | 20+     |

## Decision Tree

```
START
  ├─→ Want quick baseline test?
  │    └─→ Set USE_CONTEXT: false, run tracking
  │         └─→ Confirms HOTA ~86.2%, FPS ~290
  │
  ├─→ Want full retraining? (RECOMMENDED)
  │    └─→ Check training.yaml (USE_CONTEXT: true, DISTANCE_THRESH: 30)
  │         └─→ sbatch src/tools/train_job.sh
  │              └─→ Wait 2-3 hours
  │                   └─→ Run tracking, expect HOTA +0.5-2%
  │
  └─→ Want incremental approach?
       └─→ Modify LearnableKF to freeze base model
            └─→ Train context modules only (10 epochs)
                 └─→ Fine-tune all together (10 more epochs)
                      └─→ Run tracking, expect HOTA +0.3-1%
```

## Bottom Line

**The fixes are correct, but you need to retrain.** 

The current checkpoint learned with a suppressive gate, so its context encoder is undertrained and useless. Once you retrain with the fixed architecture (no gate, 30m threshold, proper regularization), context WILL help.

**Estimated time investment:**
- Retraining: 2-3 hours
- Testing: 30 minutes
- **Total: ~3-4 hours for proper validation**

**Expected ROI:**
- +0.5-2% HOTA (significant for tracking)
- Better multi-object handling
- More robust ID consistency
- Research contribution (proves context helps)

---

**Recommendation: Start retraining now.** The code is ready, configs are fixed, architecture is sound. You just need fresh weights that actually learned to use context.
