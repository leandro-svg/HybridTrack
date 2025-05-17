import torch
import torch.nn as nn

def euclidean_distance(predictions, targets):
    if torch.isnan(predictions).any() or torch.isnan(targets).any():
        raise ValueError("NaN values detected in predictions or targets")
    if torch.isinf(predictions).any() or torch.isinf(targets).any():
        raise ValueError("Inf values detected in predictions or targets")
    result = torch.sqrt(((predictions - targets) ** 2).sum(1) + 1e-8).mean()
    if torch.isnan(result) or torch.isinf(result):
        raise ValueError("NaN or Inf in result")
    return result

def calculate_speed_vectors(positions):
    return positions[:, :, 1:] - positions[:, :, :-1]

def direction_consistency_loss(speed_vectors):
    directions = speed_vectors / (torch.norm(speed_vectors, dim=1, keepdim=True) + 1e-6)
    direction_diffs = directions[:, :, 1:] - directions[:, :, :-1]
    return torch.norm(direction_diffs, dim=1).mean()

def calculate_losses(target_batch, state_out_batch, x_out_batch, attributes, xyz_loss):
    losses = {}
    scaling = dict(x=1.0, y=1.0, z=1.0, ry=1.0)
    for i, attr in enumerate(attributes):
        scale = scaling.get(attr, 1.0)
        losses[f'{attr}_state'] = xyz_loss(target_batch[:, i, :], state_out_batch[:, i, :]) * scale
        losses[attr] = xyz_loss(target_batch[:, i, :], x_out_batch[:, i, :]) * scale
    losses['mse_state'] = xyz_loss(target_batch[:, 0:3, :], state_out_batch[:, 0:3, :])
    losses['mse'] = xyz_loss(target_batch[:, 0:3, :], x_out_batch[:, 0:3, :])
    losses['temporal'] = xyz_loss(state_out_batch[:, 0:3, 1:], state_out_batch[:, 0:3, :-1])
    speed_vectors = calculate_speed_vectors(state_out_batch[:, 0:3, :])
    losses['speed_consistency'] = direction_consistency_loss(speed_vectors)
    return losses
