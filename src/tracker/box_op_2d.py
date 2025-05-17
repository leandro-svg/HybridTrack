import numpy as np
import torch


def convert_bbs_type(boxes, input_box_type):
    boxes = torch.tensor(boxes, dtype=torch.float32)

    assert input_box_type in ["Kitti", "OpenPCDet", "Waymo"], 'unsupported input box type!'

    if input_box_type in ["OpenPCDet", "Waymo"]:
        return boxes

    if input_box_type == "Kitti":  # (h,w,l,x,y,z,yaw, bb1, bb2, bb3, bb4) -> (x,y,z,l,w,h,yaw, bb1 ,bb2, bb3, bb4)
        t_id = boxes.shape[1] // 7
        new_boxes = torch.zeros_like(boxes)  # Initialize a new tensor with the same shape and type as boxes
        new_boxes[:, :] = boxes[:, :]
        for i in range(t_id):
            b_id = i * 7
            new_boxes[:, b_id:b_id + 3] = boxes[:, b_id + 3:b_id + 6]
            new_boxes[:, b_id + 3] = boxes[:, b_id + 2]
            new_boxes[:, b_id + 4] = boxes[:, b_id + 1]
            new_boxes[:, b_id + 5] = boxes[:, b_id + 0]
            # new_boxes[:, b_id + 6] = (torch.pi - boxes[:, b_id + 6]) + torch.pi / 2
            # new_boxes[:, b_id + 2] += boxes[:, b_id + 0] / 2
        return new_boxes

def convert_bbs_type_initial(boxes, input_box_type):
    boxes = torch.tensor(boxes, dtype=torch.float32)

    assert input_box_type in ["Kitti", "OpenPCDet", "Waymo"], 'unsupported input box type!'

    if input_box_type in ["OpenPCDet", "Waymo"]:
        return boxes

    if input_box_type == "Kitti":  # (h,w,l,x,y,z,yaw, bb1, bb2, bb3, bb4) -> (x,y,z,l,w,h,yaw, bb1 ,bb2, bb3, bb4)
        t_id = boxes.shape[1] // 7
        new_boxes = torch.zeros_like(boxes)  # Initialize a new tensor with the same shape and type as boxes
        new_boxes[:, :] = boxes[:, :]
        for i in range(t_id):
            b_id = i * 7
            new_boxes[:, b_id:b_id + 3] = boxes[:, b_id + 3:b_id + 6]
            new_boxes[:, b_id + 3] = boxes[:, b_id + 2]
            new_boxes[:, b_id + 4] = boxes[:, b_id + 1]
            new_boxes[:, b_id + 5] = boxes[:, b_id + 0]
            new_boxes[:, b_id + 6] = (torch.pi - boxes[:, b_id + 6]) + torch.pi / 2
            new_boxes[:, b_id + 2] += boxes[:, b_id + 0] / 2
        return new_boxes

def convert_back_bbs_type(boxes, output_box_type):
    boxes = torch.tensor(boxes, dtype=torch.float32)
    boxes = boxes.unsqueeze(0)
    assert output_box_type in ["Kitti", "OpenPCDet", "Waymo"], 'unsupported output box type!'

    if output_box_type in ["OpenPCDet", "Waymo"]:
        return boxes

    if output_box_type == "Kitti":  # (x,y,z,l,w,h,yaw, bb1, bb2, bb3, bb4) -> (h,w,l,x,y,z,yaw, bb1 ,bb2, bb3, bb4)
        t_id = boxes.shape[1] // 7
        original_boxes = torch.zeros_like(boxes)  # Initialize a new tensor with the same shape and type as boxes
        original_boxes[:, :] = boxes[:, :]
        for i in range(t_id):
            b_id = i * 7
            original_boxes[:, b_id + 3:b_id + 6] = boxes[:, b_id:b_id + 3]
            original_boxes[:, b_id + 2] = boxes[:, b_id + 3]
            original_boxes[:, b_id + 1] = boxes[:, b_id + 4]
            original_boxes[:, b_id + 0] = boxes[:, b_id + 5]
            # original_boxes[:, b_id + 6] = (torch.pi - boxes[:, b_id + 6]) + torch.pi / 2
            # original_boxes[:, b_id + 0] = boxes[:, b_id + 2] - boxes[:, b_id + 0] / 2
        return original_boxes
    
def convert_bbs_type_numpy(boxes, input_box_type):
    boxes = np.array(boxes)

    assert input_box_type in ["Kitti", "OpenPCDet", "Waymo"], 'unsupported input box type!'

    if input_box_type in ["OpenPCDet", "Waymo"]:
        return boxes

    if input_box_type == "Kitti":  # (h,w,l,x,y,z,yaw) -> (x,y,z,l,w,h,yaw)

        t_id = boxes.shape[1] // 7
        new_boxes = np.zeros(shape=boxes.shape)
        new_boxes[:, :] = boxes[:, :]
        for i in range(t_id):
            b_id = i * 7
            new_boxes[:, b_id + 0:b_id + 3] = boxes[:, b_id + 3:b_id + 6]
            new_boxes[:, b_id + 3] = boxes[:, b_id + 2]
            new_boxes[:, b_id + 4] = boxes[:, b_id + 1]
            new_boxes[:, b_id + 5] = boxes[:, b_id + 0]
            # new_boxes[:, b_id + 6] = (np.pi - boxes[:, b_id + 6]) + np.pi / 2
            # new_boxes[:, b_id + 2] += boxes[:, b_id + 0] / 2
        return new_boxes

def get_registration_angle(mat):
    cos_theta = mat[0, 0]
    sin_theta = mat[1, 0]

    cos_theta = torch.clamp(cos_theta, -1, 1)

    theta_cos = torch.acos(cos_theta)

    if sin_theta >= 0:
        return theta_cos
    else:
        return 2 * torch.pi - theta_cos


def register_bbs(boxes, pose):
    if pose is None:
        return boxes

    boxes = boxes 
    pose = torch.tensor(pose, dtype=torch.float32)   

    ang = get_registration_angle(pose)

    t_id = boxes.shape[1] // 7

    ones = torch.ones((boxes.shape[0], 1))
    for i in range(t_id):
        b_id = i * 7
        box_xyz = boxes[:, b_id:b_id + 3]
        box_xyz1 = torch.cat([box_xyz, ones], dim=-1)  

        box_world = torch.matmul(box_xyz1, pose.T)

        # boxes[:, b_id:b_id + 3] = box_world[:, :3]
        # boxes[:, b_id + 6] += ang
    return boxes
def register_bbs_initial(boxes, pose):
    if pose is None:
        return boxes
    boxes = boxes 
    pose = torch.tensor(pose, dtype=torch.float32)   

    ang = get_registration_angle(pose)

    t_id = boxes.shape[1] // 7

    ones = torch.ones((boxes.shape[0], 1))
    for i in range(t_id):
        b_id = i * 7
        box_xyz = boxes[:, b_id:b_id + 3]
        box_xyz1 = torch.cat([box_xyz, ones], dim=-1)  

        box_world = torch.matmul(box_xyz1, pose.T)
        boxes[:, b_id:b_id + 3] = box_world[:, :3]
        boxes[:, b_id + 6] += ang
    return boxes

def get_registration_angle_numpy(mat):
    cos_theta = mat[0, 0]
    sin_theta = mat[1, 0]

    if cos_theta < -1:
        cos_theta = -1
    if cos_theta > 1:
        cos_theta = 1

    theta_cos = np.arccos(cos_theta)

    if sin_theta >= 0:
        return theta_cos
    else:
        return 2 * np.pi - theta_cos


def register_bbs_numpy(boxes, pose):

    if pose is None:
        return boxes

    ang = get_registration_angle_numpy(pose)

    t_id = boxes.shape[1] // 7

    ones = np.ones(shape=(boxes.shape[0], 1))
    for i in range(t_id):
        b_id = i * 7
        box_xyz = boxes[:, b_id:b_id + 3]
        box_xyz1 = np.concatenate([box_xyz, ones], -1)

        box_world = np.matmul(box_xyz1, pose.T)

        # boxes[:, b_id:b_id + 3] = box_world[:, 0:3]
        # boxes[:, b_id + 6] += ang
    return boxes

def register_bbs_numpy_initial(boxes, pose):

    if pose is None:
        return boxes

    ang = get_registration_angle_numpy(pose)

    t_id = boxes.shape[1] // 7

    ones = np.ones(shape=(boxes.shape[0], 1))
    for i in range(t_id):
        b_id = i * 7
        box_xyz = boxes[:, b_id:b_id + 3]
        box_xyz1 = np.concatenate([box_xyz, ones], -1)

        box_world = np.matmul(box_xyz1, pose.T)

        boxes[:, b_id:b_id + 3] = box_world[:, 0:3]
        boxes[:, b_id + 6] += ang
    return boxes


def corners3d_to_img_boxes(P2, corners3d):
    """
    Converts 3D bounding box corners into 2D image boxes using a projection matrix.

    :param P2: Projection matrix (3, 4)
    :param corners3d: (N, 8, 3) corners in rect coordinate
    :return: img_boxes: (N, 4) [x1, y1, x2, y2] in rgb coordinate
    :return: boxes_corner: (N, 8, 2) [xi, yi] in rgb coordinate
    """
    sample_num = corners3d.shape[0]
    
    corners3d_hom = torch.cat((corners3d, torch.ones(sample_num, 8, 1)), dim=2)  # (N, 8, 4)

    img_pts = torch.matmul(corners3d_hom.to(torch.float32), P2.t().to(torch.float32))  # (N, 8, 3)

    x = img_pts[:, :, 0] / img_pts[:, :, 2]
    y = img_pts[:, :, 1] / img_pts[:, :, 2]
    x1, y1 = torch.min(x, dim=1)[0], torch.min(y, dim=1)[0]
    x2, y2 = torch.max(x, dim=1)[0], torch.max(y, dim=1)[0]

    img_boxes = torch.stack((x1, y1, x2, y2), dim=1)
    boxes_corner = torch.stack((x, y), dim=2)  # (N, 8, 2)

    img_boxes[:, 0] = torch.clamp(img_boxes[:, 0], 0, 1242 - 1)
    img_boxes[:, 1] = torch.clamp(img_boxes[:, 1], 0, 375 - 1)
    img_boxes[:, 2] = torch.clamp(img_boxes[:, 2], 0, 1242 - 1)
    img_boxes[:, 3] = torch.clamp(img_boxes[:, 3], 0, 375 - 1)

    return img_boxes, boxes_corner



def corners3d_to_img_boxes_numpy(P2, corners3d):
    """
    :param corners3d: (N, 8, 3) corners in rect coordinate
    :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
    :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
    """
    sample_num = corners3d.shape[0]
    corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

    img_pts = np.matmul(corners3d_hom, P2.T)  # (N, 8, 3)

    x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
    x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
    x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

    img_boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
    boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, 1242 - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, 375 - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, 1242 - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, 375 - 1)

    return img_boxes, boxes_corner


def bb3d_2_bb2d(bb3d, P2):
    x, y, z, l, w, h, yaw = bb3d

    pt1 = torch.tensor([l / 2, 0, w / 2, 1])
    pt2 = torch.tensor([l / 2, 0, -w / 2, 1])
    pt3 = torch.tensor([-l / 2, 0, w / 2, 1])
    pt4 = torch.tensor([-l / 2, 0, -w / 2, 1])
    pt5 = torch.tensor([l / 2, -h, w / 2, 1])
    pt6 = torch.tensor([l / 2, -h, -w / 2, 1])
    pt7 = torch.tensor([-l / 2, -h, w / 2, 1])
    pt8 = torch.tensor([-l / 2, -h, -w / 2, 1])
    pts = torch.stack([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]).unsqueeze(0) 

    transpose = torch.tensor([[torch.cos(torch.pi - yaw), 0, -torch.sin(torch.pi - yaw), x],
                              [0, 1, 0, y],
                              [torch.sin(torch.pi - yaw), 0, torch.cos(torch.pi - yaw), z],
                              [0, 0, 0, 1]])

    pts = torch.matmul(pts, transpose.t())  

    box, _ = corners3d_to_img_boxes(P2, pts[:, :, :3])

    return box

def bb3d_2_bb2d_numpy(bb3d,P2):

    x,y,z,l,w,h,yaw = bb3d[0],bb3d[1],bb3d[2],bb3d[3],bb3d[4],bb3d[5],bb3d[6]

    pt1 = [l / 2, 0, w / 2, 1]
    pt2 = [l / 2, 0, - w / 2, 1]
    pt3 = [- l / 2, 0, w / 2, 1]
    pt4 = [- l / 2, 0, - w / 2, 1]
    pt5 = [l / 2, - h, w / 2, 1]
    pt6 = [l / 2, - h, - w / 2, 1]
    pt7 = [- l / 2, - h, w / 2, 1]
    pt8 = [- l / 2, - h, - w / 2, 1]
    pts = np.array([[pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]])
    transpose = np.array([[np.cos(np.pi - yaw), 0, -np.sin(np.pi - yaw), x],
                          [0, 1, 0, y],
                          [np.sin(np.pi - yaw), 0, np.cos(np.pi - yaw), z],
                          [0, 0, 0, 1]])
    pts = np.matmul(pts, transpose.T)
    box, _ = corners3d_to_img_boxes_numpy(P2, pts[:, :, 0:3])

    return box