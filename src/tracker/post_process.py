import numpy as np
import math
import torch

def smoothstep(t):
    """
    Smoothstep function that maps t in [0, 1] to a smooth value also in [0, 1].
    It accelerates slowly at the beginning and decelerates at the end.
    """
    return 3 * t**2 - 2 * t**3

def interpolate_state(prev_state, next_state, t):
    """
    Interpolates between two vehicle states using smoothstep.
    Parameters:
    - prev_state: array-like of vehicle parameters [x, y, z, w, h, l] at start.
    - next_state: array-like of vehicle parameters [x, y, z, w, h, l] at end.
    - t: float between 0 and 1, the interpolation factor.
    Returns:
    - interpolated_state: NumPy array representing the interpolated state.
    """
    prev_state = np.array(prev_state, dtype=float)
    next_state = np.array(next_state, dtype=float)
    t = np.clip(t, 0.0, 1.0)
    smooth_t = smoothstep(t)
    interpolated_state = prev_state + (next_state - prev_state) * smooth_t
    return interpolated_state

def filter_and_interpolate_trajectory(trajectory):
    """
    Filters and interpolates the trajectory:
    - Smooth scores
    - Interpolate missing updated_state between detections
    - Optionally fill remaining gaps with predicted_state
    """
    def interpolate_angle(a1, a2, ratio):
        diff = (a2 - a1 + 180) % 360 - 180  
        return a1 + diff * ratio
    detected_num = 0.00001
    score_sum = 0.0
    sorted_keys = sorted(trajectory.keys())
    prev_key = None
    prev_state = None
    i = 0
    while i < len(sorted_keys):
        key = sorted_keys[i]
        ob = trajectory[key]
        if ob.updated_state is not None:
            prev_key = key
            prev_state = ob.updated_state
            i += 1
            continue
        j = i
        next_key = None
        next_state = None
        while j < len(sorted_keys):
            lookahead_key = sorted_keys[j]
            lookahead_ob = trajectory[lookahead_key]
            if lookahead_ob.updated_state is not None:
                next_key = lookahead_key
                next_state = lookahead_ob.updated_state
                break
            j += 1
        if next_state is None:
            break
        if prev_state is None:
            i = j
            continue
        total_steps = next_key - prev_key
        if total_steps <= 0:
            i = j
            continue
        def interpolate_heading(prev_angle, next_angle, ratio):
            prev_angle = (prev_angle + math.pi) % (2 * math.pi) - math.pi
            next_angle = (next_angle + math.pi) % (2 * math.pi) - math.pi
            diff = next_angle - prev_angle
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            angle_interpolated = prev_angle + ratio * diff
            angle_interpolated = (angle_interpolated + math.pi) % (2 * math.pi) - math.pi
            return angle_interpolated
        for interp_index in range(i, j):
            interp_key = sorted_keys[interp_index]
            step = interp_key - prev_key
            ratio = step / total_steps
            prev_score = trajectory[prev_key].score
            next_score = trajectory[next_key].score
            if prev_score is not None and next_score is not None:
                interp_score = (1 - ratio) * prev_score + ratio * next_score
            else:
                interp_score = next_score if prev_score is None else prev_score
            interp_linear = torch.tensor(interpolate_state(prev_state[:-1],  next_state[:-1], ratio))
            interp_heading = interpolate_heading(
                prev_state[-1],
                next_state[-1],
                ratio
            )
            interp_state = list(interp_linear) + [interp_heading]
            interp_state = torch.tensor(interp_state)
            trajectory[interp_key].updated_state = interp_state
            trajectory[interp_key].score = interp_score
        i = j
    first_valid_key = next((k for k in sorted_keys if trajectory[k].updated_state is not None), None)
    if first_valid_key is not None:
        for k in sorted_keys:
            if k < first_valid_key and trajectory[k].updated_state is None:
                trajectory[k].updated_state = trajectory[first_valid_key].updated_state
    for key in sorted_keys:
        ob = trajectory[key]
        if ob.updated_state is None and ob.predicted_state is not None:
            ob.updated_state = ob.predicted_state
    return trajectory
