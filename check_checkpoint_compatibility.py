#!/usr/bin/env python3
"""
Check if checkpoint has new context parameters (null_ctx_token, ctx_gate).
If missing, context features will be randomly initialized at load time.
"""
import torch
import sys

def check_checkpoint(ckpt_path):
    print(f"\n{'='*70}")
    print(f"Checking checkpoint: {ckpt_path}")
    print('='*70)
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Determine checkpoint structure
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                print("✓ Checkpoint format: Dictionary with 'state_dict' key")
            else:
                state_dict = ckpt
                print("✓ Checkpoint format: Direct state_dict")
        else:
            print("✓ Checkpoint format: Full model object")
            state_dict = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else None
        
        if state_dict is None:
            print("✗ Could not extract state_dict from checkpoint")
            return False
        
        # Check for new context parameters
        has_null_token = any('null_ctx_token' in k for k in state_dict.keys())
        has_ctx_gate = any('ctx_gate' in k for k in state_dict.keys())
        has_det_enc = any('det_enc' in k for k in state_dict.keys())
        has_context = any('ctx_proj' in k or 'cross_attn' in k for k in state_dict.keys())
        
        print(f"\nContext-Related Parameters:")
        print(f"  null_ctx_token (NEW):    {'✓ FOUND' if has_null_token else '✗ MISSING - will be random!'}")
        print(f"  ctx_gate (NEW):          {'✓ FOUND' if has_ctx_gate else '✗ MISSING - will be random!'}")
        print(f"  det_enc (context):       {'✓ FOUND' if has_det_enc else '✗ MISSING'}")
        print(f"  ctx_proj/cross_attn:     {'✓ FOUND' if has_context else '✗ MISSING'}")
        
        # Check total parameters
        total_params = sum(p.numel() for p in state_dict.values() if torch.is_tensor(p))
        print(f"\nTotal parameters: {total_params:,}")
        
        # List all keys for debugging
        print(f"\nAll parameter keys ({len(state_dict)} total):")
        for i, key in enumerate(sorted(state_dict.keys())):
            if torch.is_tensor(state_dict[key]):
                shape = tuple(state_dict[key].shape)
                print(f"  {i+1:3d}. {key:60s} {str(shape):20s}")
        
        # Verdict
        print(f"\n{'='*70}")
        if has_null_token and has_ctx_gate:
            print("✓ COMPATIBLE: Checkpoint has new context parameters")
            print("  → Safe to use with USE_CONTEXT=true")
        elif has_context:
            print("✗ INCOMPATIBLE: Checkpoint has old context code (missing null token)")
            print("  → Using USE_CONTEXT=true will degrade performance!")
            print("  → Recommendation: Retrain with new code OR disable context")
        else:
            print("○ NO CONTEXT: Checkpoint trained without context features")
            print("  → Must retrain to use context")
        print('='*70 + '\n')
        
        return has_null_token and has_ctx_gate
        
    except Exception as e:
        print(f"\n✗ Error loading checkpoint: {e}")
        return False

if __name__ == '__main__':
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'src/result/hybridtrack/online/model_checkpoint/hybridtrack.pt'
    check_checkpoint(ckpt_path)
