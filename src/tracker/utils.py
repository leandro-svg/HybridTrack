import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from src.dataset.tracking_dataset_utils import read_calib
from ...model.hybridtrack import Trajectory
from .box_op_2d import *
import numpy as np
import torch
from networkx.algorithms.matching import max_weight_matching, min_weight_matching
import networkx as nx
from scipy.optimize import linear_sum_assignment
from .cIoU import  ciou_3d


def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

def find_square_points( center, exterior):
    x, y, z = center
    x1, y1, z1 = exterior
    
    dx, dy, dz = x1 - x, y1 - y, z1 - z
    
    side_length = torch.sqrt(dx**2 + dy**2 + dz**2) * np.sqrt(2)
    
    ux = torch.tensor([1, 0, 0])
    uy = torch.tensor([0, 0, 1])
    
    half_side = side_length / 2
    
    P1 = center + half_side * (ux + uy)
    P2 = center + half_side * (-ux + uy)
    P3 = center + half_side * (-ux - uy)
    P4 = center + half_side * (ux - uy)
    
    return [P1, P2, P3, P4]
def calculate_area( box):
    """Calculate the area of a bounding box (BEV)."""
    return box[2] * box[3]

def intersection_area( box1, box2):
    """Calculate the intersection area between two bounding boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[0] + box1[2], box2[0] + box2[2])
    yB = min(box1[1] + box1[3], box2[1] + box2[3])

    if xA < xB and yA < yB:
        return (xB - xA) * (yB - yA)
    return 0

def union_area( box1, box2, inter_area):
    """Calculate the union area of two bounding boxes given intersection area."""
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    return area1 + area2 - inter_area

def enclosing_rectangle( box1, box2):
    """Calculate the minimum enclosing rectangle for two bounding boxes."""
    xA = min(box1[0], box2[0])
    yA = min(box1[1], box2[1])
    xB = max(box1[0] + box1[2], box2[0] + box2[2])
    yB = max(box1[1] + box1[3], box2[1] + box2[3])
    return np.array([xA, yA, xB - xA, yB - yA])

def euclidean_distance( box1, box2):
    """Calculate the Euclidean distance between the center points of two boxes."""
    center1 = np.array([box1[0] + box1[2] / 2, box1[1] + box1[3] / 2])
    center2 = np.array([box2[0] + box2[2] / 2, box2[1] + box2[3] / 2])
    return np.linalg.norm(center1 - center2)

def diagonal_distance( rect):
    """Calculate the diagonal distance of the minimum enclosing rectangle."""
    return np.sqrt(rect[2]**2 + rect[3]**2)

def Ro_IoU( Bd_bev, Bt_bev):
    """Calculate Ro-IoU."""
    inter_area = intersection_area(Bd_bev, Bt_bev)
    union_area_val = union_area(Bd_bev, Bt_bev, inter_area)
    return inter_area / union_area_val

def Ro_GDIoU( Bd_bev, Bt_bev, omega1=1.0, omega2=1.0):
    """Calculate Ro_GDIoU."""
    iou = Ro_IoU(Bd_bev, Bt_bev)
    rect = enclosing_rectangle(Bd_bev, Bt_bev)
    c = euclidean_distance(Bd_bev, Bt_bev)
    d = diagonal_distance(rect)

    gd_iou = iou - omega1 * (c / d) - omega2 * (c ** 2 / d ** 2)
    return gd_iou
def velo_to_cam_numpy( cloud,vtc_mat):
    mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
    mat[:,0:3]=cloud[:,0:3]
    mat=np.mat(mat)
    normal=np.mat(vtc_mat)
    normal=normal[0:3,0:4]
    transformed_mat = normal * mat.T
    T=np.array(transformed_mat.T,dtype=np.float32)
    return T
def velo_to_cam( cloud, vtc_mat): 
    mat = torch.ones((cloud.shape[0], 4), dtype=torch.float32) 
    mat[:, 0:3] = cloud[:, 0:3] 
    if not isinstance(vtc_mat, torch.Tensor): 
        vtc_mat = torch.tensor(vtc_mat, dtype=torch.float32) 
        normal = vtc_mat[0:3, 0:4] 
        transformed_mat = torch.matmul(normal, mat.T) 
        T = transformed_mat.T.contiguous() 
        
    return T
def real_world_to_image_domain_and_2d( box_template, v2c, p2, current_pose):
    entered_boxe = torch.clone(box_template)
    box = register_bbs_initial(entered_boxe,torch.inverse(torch.tensor(current_pose)))

    #box =   box_template 

    box[:, 6] = -box[:, 6] - torch.pi / 2
    box[:, 2] -= box[:, 5] / 2

    box[:,0:3] = velo_to_cam(box[:,0:3],v2c)[:,0:3]
    # box = box[0]
    box2d = torch.zeros((np.shape(box)[0]),4)
    for index, pseudo_box in enumerate(box):
        pseudo_box2d = bb3d_2_bb2d(pseudo_box,torch.tensor(p2[0:3,:]))
        box2d[index] = pseudo_box2d
    return box, box2d

def real_world_to_velodyne_domain( box_template, v2c, p2, current_pose):
    entered_boxe = torch.clone(box_template)

    box = register_bbs_initial(entered_boxe,torch.inverse(torch.tensor(current_pose)))

    # box = torch.clone(box_template)

    box[:, 6] = -box[:, 6] - torch.pi / 2
    box[:, 2] -= box[:, 5] / 2


    return box
def compute_angle( v1, v2):
    v1_norm = v1 / torch.norm(v1)
    v2_norm = v2 / torch.norm(v2)
    
    dot_product = torch.dot(v1_norm, v2_norm)
    
    angle = torch.acos(dot_product)
    
    return angle

def calculate_area( detections):
    """
    Calculate the area of the bounding box that encompasses all detections.
    """
    if len(detections) == 0:
        return 1.0  

    min_x = min(det[0] for det in detections)
    max_x = max(det[2] for det in detections)
    min_y = min(det[1] for det in detections)
    max_y = max(det[3] for det in detections)

    area = (max_x - min_x) * (max_y - min_y)
    return max(area, 1.0)
def calculate_local_density( detection, all_detections, radius=15.0):
    """
    Calculate the local density for a given detection based on neighboring detections within a radius.
    """
    count = 0
    for det in all_detections:
        diffs = detection[0:3] - det[0:3]
        if torch.sqrt((diffs ** 2).sum(-1)) < radius:
            count += 1
    
    area = torch.pi * radius ** 2
    return count / area

def min_max_normalize( data, min_val=None, max_val=None):
    """
    Normalize the data to be in the range [0, 1].
    Args:
        data: tensor of shape (..., 3), where the last dimension represents the coordinates (x, y, z).
    Returns:
        normalized_data: tensor of the same shape as input, normalized along the last dimension.
    """
    if min_val is None:
        min_val = data.min(dim=0, keepdim=True).values
    if max_val is None:
        max_val = data.max(dim=0, keepdim=True).values
    return (data - min_val) / (max_val - min_val), min_val, max_val

def cam_to_velo(cloud, vtc_mat):
    vtc_mat = torch.tensor(vtc_mat, dtype=torch.float32)
    
    ones = torch.ones((cloud.shape[0], 1), dtype=torch.float32)
    mat = torch.cat((cloud[:, :3], ones), dim=1)
    
    normal = torch.inverse(vtc_mat)[:3, :4]
    
    transformed_mat = torch.matmul(normal, mat.T)
    
    T = transformed_mat.T[:, :3]
    
    return T