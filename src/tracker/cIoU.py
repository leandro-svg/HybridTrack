import torch
import math

def _get_minmax_corners(boxes):
    """
    Convert each oriented 3D box into its axis-aligned bounding box (AABB).
    boxes: (N, 7) = [x, y, z, width, height, length, heading]
    Returns AABBs of shape (N, 2, 3), where [:, 0, :] is the min corner and [:, 1, :] is the max corner.
    """

    corner_offsets = torch.tensor([
        [ 0.5,  0.5,  0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5],
    ], dtype=boxes.dtype, device=boxes.device)  

    centers = boxes[:, :3]               
    widths  = boxes[:, 3]               
    heights = boxes[:, 4]
    lengths = boxes[:, 5]
    headings= boxes[:, 6]               

    dims = torch.stack([widths, heights, lengths], dim=1)  
    corners_local = corner_offsets.unsqueeze(0) * dims.unsqueeze(1)  

    cos_h = torch.cos(headings)  
    sin_h = torch.sin(headings)  
    x_loc = corners_local[:, :, 0]
    y_loc = corners_local[:, :, 1]
    z_loc = corners_local[:, :, 2]  
    x_rot = x_loc * cos_h.unsqueeze(1) - y_loc * sin_h.unsqueeze(1)
    y_rot = x_loc * sin_h.unsqueeze(1) + y_loc * cos_h.unsqueeze(1)

    corners_world = torch.stack([x_rot, y_rot, z_loc], dim=-1) + centers.unsqueeze(1)  

    min_corners = corners_world.min(dim=1).values  
    max_corners = corners_world.max(dim=1).values  
    return torch.stack([min_corners, max_corners], dim=1)  


def _aabb_volume(aabbs):
    """
    Compute volume of each axis-aligned bounding box.
    aabbs: (N, 2, 3)
    Returns: (N,) volume
    """
    diff = aabbs[:, 1] - aabbs[:, 0]  
    return diff[:, 0] * diff[:, 1] * diff[:, 2]


def _intersection_volume_broadcast(aabbs1, aabbs2):
    """
    Broadcasted intersection volume of two sets of AABBs.
    aabbs1: (N, 2, 3)
    aabbs2: (M, 2, 3)
    Returns: (N, M) intersection volume
    """
    intersection_min = torch.max(aabbs1[:, None, 0], aabbs2[None, :, 0])
    intersection_max = torch.min(aabbs1[:, None, 1], aabbs2[None, :, 1])
    intersection_dim = torch.clamp(intersection_max - intersection_min, min=0)
    return intersection_dim[:, :, 0] * intersection_dim[:, :, 1] * intersection_dim[:, :, 2]


def _enclosing_box_broadcast(aabbs1, aabbs2):
    """
    Compute the smallest enclosing AABB that encloses each pair of AABBs.
    aabbs1: (N, 2, 3)
    aabbs2: (M, 2, 3)
    Returns: (N, M, 2, 3)
    """
    min_corner = torch.min(aabbs1[:, None, 0], aabbs2[None, :, 0])  
    max_corner = torch.max(aabbs1[:, None, 1], aabbs2[None, :, 1])  
    return torch.stack([min_corner, max_corner], dim=2)  


def _aspect_ratio_correction(dims1, dims2, iou):
    """
    Compute the aspect ratio term used in the CIoU formula, broadcasted for all pairs.
    dims1: (N, 3) => width1, height1, length1
    dims2: (M, 3) => width2, height2, length2
    iou:   (N, M)
    Returns: (N, M) aspect ratio correction
    """
    w1, _, l1 = dims1[:, 0], dims1[:, 1], dims1[:, 2]  
    w2, _, l2 = dims2[:, 0], dims2[:, 1], dims2[:, 2]  

    ratio1 = torch.atan(w1 / (l1 + 1e-7))  
    ratio2 = torch.atan(w2 / (l2 + 1e-7)) 

    ratio_diff = ratio1.unsqueeze(1) - ratio2.unsqueeze(0)  
    v = (4.0 / (math.pi ** 2)) * ratio_diff * ratio_diff

    alpha = v / (1e-7 + (1 - iou) + v)
    return alpha * v  


def ciou_3d(boxes1, boxes2):
    """
    Compute the 3D Complete IoU (CIoU) and regular IoU between two sets of oriented 3D boxes.
    boxes1, boxes2: (N, 7), (M, 7) => [x, y, z, width, height, length, heading]
    Returns:
        CIoU: (N, M)
        IoU:  (N, M)
    """

    aabb1 = _get_minmax_corners(boxes1)  
    aabb2 = _get_minmax_corners(boxes2)  

    inter_vol = _intersection_volume_broadcast(aabb1, aabb2)  
    vol1 = _aabb_volume(aabb1)  
    vol2 = _aabb_volume(aabb2) 
    union_vol = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol
    iou = inter_vol / (union_vol + 1e-7)  

    centers1 = boxes1[:, :3]  
    centers2 = boxes2[:, :3]  
    center_diff = centers1.unsqueeze(1) - centers2.unsqueeze(0)  
    center_dist = torch.sum(center_diff * center_diff, dim=2) 

    enc_box = _enclosing_box_broadcast(aabb1, aabb2) 
    diag = enc_box[:, :, 1, :] - enc_box[:, :, 0, :]  
    diag_length_sq = torch.sum(diag * diag, dim=2)    

    dims1 = aabb1[:, 1] - aabb1[:, 0]  
    dims2 = aabb2[:, 1] - aabb2[:, 0]  
    ar_corr = _aspect_ratio_correction(dims1, dims2, iou) 

    ciou = iou.clone()
    mask = (iou >= 0.5)
    # If IoU >= 0.5 => subtract center-dist & aspect-ratio
    ciou[mask] = ciou[mask] - (center_dist[mask] / (diag_length_sq[mask] + 1e-7) + ar_corr[mask])
    # Otherwise => subtract only center-dist
    ciou[~mask] = ciou[~mask] - (center_dist[~mask] / (diag_length_sq[~mask] + 1e-7))

    return ciou, iou
