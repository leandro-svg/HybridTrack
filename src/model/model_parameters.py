"""This file contains the parameters for system modeling, focusing on translation and rotation.

Updated 2023-02-06: f and h now support batch processing for improved speed.
"""

import math
import torch
from torch import Tensor
from typing import Tuple, Union

# Define torch.pi if not already defined (it is in Python 3.8+ math, but good for torch tensors)
if not hasattr(torch, 'pi'):
    torch.pi = torch.acos(torch.zeros(1)).item() * 2

#########################
### Design Parameters ###
#########################
m: int = 7  # State size
n: int = 7  # Measurements size

# Initial state estimates (mean and covariance)
m1x_0: Tensor = torch.ones(m, 1)
m2x_0: Tensor = torch.zeros(m, m)  # Typically, a small non-zero covariance is used if uncertain

FRAME_RATE: float = 0.03  # Frame rate of the system
DT: float = FRAME_RATE     # Time step duration

# Identity matrices (useful for constructing state transition and observation matrices)
EYE3: Tensor = torch.eye(3)
EYE7: Tensor = torch.eye(7)
ZEROS3_3: Tensor = torch.zeros(3, 3)

# Rotation parameters (example values, can be set dynamically)
roll_deg: float = 0.0  # Default to no rotation
pitch_deg: float = 0.0
yaw_deg: float = 0.0

def get_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Tensor:
    """Computes the 3D rotation matrix from roll, pitch, yaw angles."""
    roll: float = math.radians(roll_deg)
    pitch: float = math.radians(pitch_deg)
    yaw: float = math.radians(yaw_deg)

    RX: Tensor = torch.tensor([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]])
    RY: Tensor = torch.tensor([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]])
    RZ: Tensor = torch.tensor([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]])
    return torch.mm(RZ, torch.mm(RY, RX))

# Default rotation matrix (no rotation)
RotMatrix: Tensor = get_rotation_matrix(roll_deg, pitch_deg, yaw_deg)

#################################################################
### State Evolution Function f (Constant Velocity Example)    ###
#################################################################
def f(x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    State evolution function (example: constant velocity model for 3D position and orientation).
    Assumes state x is [batch_size, state_dim, 1].
    State vector: [x, y, z, vx, vy, vz, ry] (example, adjust as needed)
    For a simple constant velocity model affecting only x, y, z based on vx, vy, vz:
    x_k+1 = x_k + vx_k * DT
    y_k+1 = y_k + vy_k * DT
    z_k+1 = z_k + vz_k * DT
    vx_k+1 = vx_k
    vy_k+1 = vy_k
    vz_k+1 = vz_k
    ry_k+1 = ry_k
    """
    batch_size, state_dim, _ = x.shape
    device = x.device

    # State transition matrix F for a constant velocity model (example)
    F_matrix = torch.eye(state_dim, device=device)
    if state_dim >= 6:  # Assuming pos and vel are the first 6 components
        F_matrix[0, 3] = DT  # x += vx * DT
        F_matrix[1, 4] = DT  # y += vy * DT
        F_matrix[2, 5] = DT  # z += vz * DT

    F_matrix_batch = F_matrix.reshape((1, state_dim, state_dim)).repeat(batch_size, 1, 1)
    x_next = torch.bmm(F_matrix_batch, x)

    if jacobian:
        return x_next, F_matrix_batch  # Jacobian of linear function is the matrix itself
    return x_next

######################################################
### Observation Function h (Direct State Observation) ###
######################################################
H_matrix_direct: Tensor = torch.eye(n, m)  # Assuming n <= m

def h(x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Observation function.
    Assumes direct observation of the first n components of the state if n < m,
    or direct observation of the full state if n = m.
    Input x: [batch_size, state_dim, 1]
    Output y: [batch_size, obs_dim, 1]
    """
    batch_size = x.shape[0]
    device = x.device

    H_obs = H_matrix_direct.to(device)
    H_obs_batch = H_obs.reshape((1, n, m)).repeat(batch_size, 1, 1)
    y = torch.bmm(H_obs_batch, x)

    if jacobian:
        return y, H_obs_batch
    return y

def hRotate(x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Observation function with rotation applied to the observation matrix (if applicable).
    Currently, it behaves like h (direct observation) as RotMatrix is identity by default
    and the original code used EYE7 for H.
    Input x: [batch_size, state_dim, 1]
    Output y: [batch_size, obs_dim, 1]
    """
    batch_size = x.shape[0]
    device = x.device

    H_obs_rotated_batch = H_matrix_direct.to(device).reshape((1, n, m)).repeat(batch_size, 1, 1)
    y = torch.bmm(H_obs_rotated_batch, x)

    if jacobian:
        return y, H_obs_rotated_batch
    return y

###############################################
### Process Noise Q and Observation Noise R ###
###############################################
Q_diag_values = torch.ones(m) * 1.0  # Default: moderate process noise
Q_structure: Tensor = torch.diag(Q_diag_values) * 1e-4  # Scaled down from original 100

R_diag_values = torch.ones(n) * 1.0  # Default: moderate observation noise
R_structure: Tensor = torch.diag(R_diag_values) * 1e-3  # Same as original