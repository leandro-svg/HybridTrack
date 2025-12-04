#!/usr/bin/env python3
"""
Inspect the learned context gate value from the trained checkpoint
"""

import torch
import sys
import pickle

checkpoint_path = 'src/result/hybridtrack/online/model_checkpoint/hybridtrack.pt'

try:
    # Load checkpoint with weights_only=False to skip class validation
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    print()
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Find context-related parameters
    print("Context-related parameters:")
    print("="*80)
    
    for key, value in state_dict.items():
        if 'ctx' in key.lower() or 'context' in key.lower():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    # Single value parameter
                    raw_value = value.item()
                    sigmoid_value = torch.sigmoid(value).item()
                    print(f"{key}:")
                    print(f"  Raw value: {raw_value:.6f}")
                    print(f"  Sigmoid value: {sigmoid_value:.6f}")
                else:
                    print(f"{key}: shape={value.shape}, mean={value.mean().item():.6f}, std={value.std().item():.6f}")
            print()
    
    # Check if ctx_gate exists
    if 'ctx_gate' in state_dict:
        gate_raw = state_dict['ctx_gate'].item()
        gate_sigmoid = torch.sigmoid(state_dict['ctx_gate']).item()
        print("\n" + "="*80)
        print("CRITICAL: Context Gate Value")
        print("="*80)
        print(f"Raw gate parameter: {gate_raw:.6f}")
        print(f"Sigmoid(gate): {gate_sigmoid:.6f}")
        print(f"\nInterpretation:")
        if gate_sigmoid < 0.1:
            print("  ⚠️  NEARLY ZERO - Context is being suppressed!")
        elif gate_sigmoid < 0.3:
            print("  ⚠️  VERY LOW - Context has minimal influence")
        elif gate_sigmoid < 0.7:
            print("  ✓ MODERATE - Context is contributing")
        else:
            print("  ✓ HIGH - Context is strongly influencing predictions")
    else:
        print("\n⚠️  WARNING: ctx_gate parameter not found in checkpoint!")
        print("This means the model was trained without the learnable gate.")
        
    # Check for null_ctx_token
    if 'null_ctx_token' in state_dict:
        null_token = state_dict['null_ctx_token']
        print(f"\nNull context token: mean={null_token.mean().item():.6f}, std={null_token.std().item():.6f}")
    
    # Check epoch/training info
    if 'epoch' in checkpoint:
        print(f"\nCheckpoint epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
