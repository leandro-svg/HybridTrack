# Extensive Debug Logging - Context Processing Pipeline

## Overview
This document describes the comprehensive debug logging added to understand why context has ZERO impact on tracking results (HOTA 86.245% identical with/without context).

## Debug Points Added

### 1. Context Preparation (`hybridtrack.py::_prepare_detection_context`)

**Location**: Lines 418-446
**Frequency**: First 10 frames
**Output Format**:
```
================================================================================
[CONTEXT PREP] Frame <timestamp>, Call <counter>
================================================================================
  Active tracks: <count>, IDs: [...]
  Track positions available: <count>
  Context tensor shape: (B, 16, 8)
  Total valid context entries: <count>
  Valid per slot (first 10): [...]
  Context stats: mean=X.XXXX, std=X.XXXX
  Context range: min=X.XXXX, max=X.XXXX

  Slot <i> (ego track <i>): <n> neighbors
    Neighbor <j>: dx=X.XX, dy=X.XX, dz=X.XX,
                   l=X.XX, w=X.XX, h=X.XX,
                   dry=X.XXX, score=X.XXX
================================================================================
```

**Purpose**: Verify context is being prepared correctly with ego-relative coordinates, correct neighbor counts, and reasonable values.

---

### 2. LKF Prediction Input (`hybridtrack.py::_lkf_prediction`)

**Location**: Lines 459-471
**Frequency**: First 10 calls
**Output Format**:
```
[LKF PREDICTION INPUT] Frame <timestamp>, Call <counter>
  use_context=True
  det_ctx: shape=(B, 16, 8), mean=X.XXXX, std=X.XXXX
  det_mask: shape=(B, 16), num_valid=<count>
  hist_ctx: None
```

**Purpose**: Confirm what's being passed to the model (shape, statistics, valid entries).

---

### 3. Model Context Encoding & Attention (`LearnableKF.py::step_prior`)

**Location**: Lines 320-368
**Frequency**: First 10 steps
**Output Format**:
```
================================================================================
[MODEL CTX FUSION] Step <counter>
================================================================================
  INPUTS:
    det_context: shape=(B, 16, 8)
    det_context stats: mean=X.XXXX, std=X.XXXX
    det_context range: min=X.XXXX, max=X.XXXX
    det_context xyz: mean=X.XXXX, std=X.XXXX
    det_context scores: mean=X.XXXX, std=X.XXXX
    det_mask: <count> padded out of <total>
    Valid context entries: <count>

  ENCODING:
    det_kv (encoded): mean=X.XXXX, std=X.XXXX
    det_kv range: min=X.XXXX, max=X.XXXX
    ego_det_q (query): mean=X.XXXX, std=X.XXXX
    det_attn (after attn): mean=X.XXXX, std=X.XXXX

  FUSION:
    ctx_concat: mean=X.XXXX, std=X.XXXX
    ctx_proj: mean=X.XXXX, std=X.XXXX
    ctx_proj range: min=X.XXXX, max=X.XXXX
    ctx_gate (raw): X.XXXXXX
    ctx_gate (tanh): X.XXXXXX
    Context contribution magnitude: X.XXXXXX
    fused_feats (before ctx): mean=X.XXXX, std=X.XXXX
================================================================================
```

**Purpose**: 
- Check if context encoder produces meaningful embeddings (det_kv)
- Verify ego query is reasonable (ego_det_q)
- Check attention output (det_attn)
- **CRITICAL**: Check gate value (ctx_gate) - if near zero, context is suppressed
- Check context contribution magnitude

---

### 4. Cross-Attention Mechanism (`context_modules.py::CrossAttentionFusion`)

**Location**: Lines 134-158
**Frequency**: First 10 calls
**Output Format**:
```
[CROSS-ATTENTION] Call <counter>:
  ego_q: mean=X.XXXX, std=X.XXXX
  ctx_kv: mean=X.XXXX, std=X.XXXX
  Valid context: <count> / <total>
  Attention weights (avg over heads): mean=X.XXXXXX, std=X.XXXXXX
  Attention range: min=X.XXXXXX, max=X.XXXXXX
  First sample valid attention: [w1, w2, w3, w4, w5]
  Attention entropy: X.XXXX
  Output: mean=X.XXXX, std=X.XXXX
```

**Purpose**:
- Verify attention mechanism is working (not all uniform weights)
- Check if attention is focusing on specific neighbors or uniform
- Entropy shows concentration: low = focused, high = uniform
- Verify output has meaningful signal

---

### 5. Fusion Impact (`LearnableKF.py::step_prior` - after gating)

**Location**: Lines 373-381
**Frequency**: First 10 steps
**Output Format**:
```
  AFTER FUSION:
    gate_term: mean=X.XXXXXX, std=X.XXXXXX
    gate_term range: min=X.XXXXXX, max=X.XXXXXX
    fused_feats (after ctx): mean=X.XXXX, std=X.XXXX
    Absolute change in fused_feats: X.XXXXXX
    Relative change: X.XXXXXX
================================================================================
```

**Purpose**:
- **MOST CRITICAL**: Shows actual change in features after adding context
- If "Absolute change" is near zero → context is not affecting predictions
- If "Relative change" << 0.01 → negligible impact
- This directly answers "why no impact?"

---

## Expected Debug Flow (per prediction step)

```
[CONTEXT PREP] → [LKF PREDICTION INPUT] → [MODEL CTX FUSION]
                                              ↓
                                    [CROSS-ATTENTION]
                                              ↓
                                    [AFTER FUSION]
```

## Key Diagnostics

### If context has no impact, look for:

1. **Context preparation issues**:
   - All zeros in context tensor
   - No valid neighbors (valid_per_slot all zeros)
   - Extreme values (NaN, Inf)

2. **Encoding issues**:
   - det_kv near zero or uniform
   - Encoder not producing meaningful embeddings

3. **Attention issues**:
   - Attention weights uniform (all ~0.0625 for 16 neighbors)
   - High entropy (> 2.5 for 16 neighbors means nearly uniform)
   - det_attn output near zero

4. **Gating issues** ⭐ MOST LIKELY:
   - ctx_gate near zero (e.g., -5.0 raw → ~0.0 after tanh)
   - Context contribution magnitude < 1e-6
   - Relative change < 1e-4

5. **Projection issues**:
   - ctx_proj near zero despite non-zero attention
   - ctx_proj dimensions mismatched

## Running the Debug

```bash
sbatch run_tracking_debug.sh
```

This will:
- Run sequence 0000 only (fast, ~2 minutes)
- Generate extensive debug output in `hybridtrack_debug.<jobid>.out`
- Save results to `results/debug_context_enabled/`

## Next Steps After Debug

Based on the debug output, we can:

1. **If gate ≈ 0**: Need to initialize gate differently or remove gating
2. **If attention uniform**: Check context encoder initialization
3. **If context all zeros**: Fix context preparation logic
4. **If encoder output zeros**: Check LayerNorm/activation issues
5. **If everything looks good but no impact**: Architecture problem (context path bypassed)

## Files Modified

- `/src/tracker/hybridtrack.py`: Context prep + LKF input debug
- `/src/model/LearnableKF.py`: Model fusion debug
- `/src/model/context_modules.py`: Cross-attention debug
- `/run_tracking_debug.sh`: Debug tracking script
