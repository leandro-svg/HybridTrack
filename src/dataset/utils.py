import numpy as np
def proj2Dto3D(pts2d, zs, k):
    fx = k[0,0]
    fy = k[1,1]
    cx = k[0,2]
    cy = k[1,2]
    X = (pts2d[0] - cx) * zs / fx
    Y = (pts2d[1] - cy) * zs / fy
    return np.array([X, Y, zs])


"""
Contains utility functions for data generation in simulations.
"""

import torch
from typing import List, Optional
from tools.batch_generation import SystemModel  # Assuming SystemModel is in this path

def DataGen(cfg, SysModel_data: SystemModel, train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader, fileName: str) -> None:
    """
    Generates training and cross-validation data and saves it to a file.

    Args:
        cfg: Configuration object (YACS CfgNode) containing parameters like TRAINER.N_E, TRAINER.N_CV, etc.
        SysModel_data: An instance of SystemModel used to generate data batches.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        fileName: The path to the file where the generated data will be saved.
    """
    # Generate training data
    SysModel_data.GenerateBatch(cfg, train_dataloader, cfg.TRAINER.N_E, randomInit=cfg.TRAINER.RANDOMINIT_TRAIN)
    train_input: List[torch.Tensor] = SysModel_data.Input
    train_target: List[torch.Tensor] = SysModel_data.Target
    train_init: Optional[torch.Tensor] = SysModel_data.m1x_0_batch

    # Generate cross-validation data
    SysModel_data.GenerateBatch(cfg, val_dataloader, cfg.TRAINER.N_CV, randomInit=cfg.TRAINER.RANDOMINIT_CV)
    cv_input: List[torch.Tensor] = SysModel_data.Input
    cv_target: List[torch.Tensor] = SysModel_data.Target
    cv_init: Optional[torch.Tensor] = SysModel_data.m1x_0_batch

    # Save the generated data
    # The saved tuple contains: (training inputs, training targets, validation inputs, validation targets, training initial states, validation initial states)
    torch.save([train_input, train_target, cv_input, cv_target, train_init, cv_init], fileName)
    print(f"Training and validation data saved to {fileName}")

def DataGen_eval(cfg, SysModel_data: SystemModel, test_dataloader: torch.utils.data.DataLoader, fileName: str) -> None:
    """
    Generates test data and saves it to a file.

    Args:
        cfg: Configuration object (YACS CfgNode) containing parameters like TRAINER.N_TEST, etc.
        SysModel_data: An instance of SystemModel used to generate data batches.
        test_dataloader: DataLoader for test data.
        fileName: The path to the file where the generated data will be saved.
                  The data is saved as a tuple: (test_input, test_target, test_init).
    """
    # Generate test data
    SysModel_data.GenerateBatch(cfg, test_dataloader, cfg.TRAINER.N_TEST, randomInit=cfg.TRAINER.RANDOMINIT_CV)
    test_input: List[torch.Tensor] = SysModel_data.Input
    test_target: List[torch.Tensor] = SysModel_data.Target
    test_init: Optional[torch.Tensor] = SysModel_data.m1x_0_batch

    # Save the generated data
    # The saved tuple contains: (test inputs, test targets, test initial states)
    torch.save([test_input, test_target, test_init], fileName)
    print(f"Test data saved to {fileName}")
