#####################
# Based on https://github.com/hongzhenwang/RRPN-revise
# Licensed under The MIT License
# Author: yanyan, scrin@foxmail.com
#####################
import math

import numba
import numpy as np
from numba import cuda


def div_up(m, n):
    return m // n + (m % n > 0)

def triangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0

def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(triangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4], int_pts[2 * i + 4:2 * i + 6]))
    return area_val


def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = np.mean(int_pts.reshape(-1, 2), axis=0)
        vs = []
        for i in range(num_of_inter):
            v = int_pts[2 * i:2 * i + 2] - center
            d = np.linalg.norm(v)
            v /= d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs.append(v[0])
        vs = np.array(vs)
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx, ty = int_pts[2 * i], int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = pts1[2 * i:2 * i + 2]
    B = pts1[2 * ((i + 1) % 4):2 * ((i + 1) % 4) + 2]
    C = pts2[2 * j:2 * j + 2]
    D = pts2[2 * ((j + 1) % 4):2 * ((j + 1) % 4) + 2]

    BA = B - A
    DA = D - A
    CA = C - A
    acd = DA[1] * CA[0] > CA[1] * DA[0]
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        CB = C - B
        abc = CA[1] * BA[0] > BA[1] * CA[0]
        abd = DA[1] * BA[0] > BA[1] * DA[0]
        if abc != abd:
            DC = D - C
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA[1] * DC[0] - BA[0] * DC[1]
            Dx = ABBA * DC[0] - BA[0] * CDDC
            Dy = ABBA * DC[1] - BA[1] * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0

def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = np.zeros(2)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


def rbbox_to_corners(corners, rbbox):
    rbbox = rbbox.detach().numpy()

    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    
    corners_x = np.array([-x_d / 2, -x_d / 2, x_d / 2, x_d / 2])
    corners_y = np.array([-y_d / 2, y_d / 2, y_d / 2, -y_d / 2])

    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y




def rotate_iou_eval(rbox1, rbox2, criterion=-1):
    corners1 = np.zeros(8)
    corners2 = np.zeros(8)
    inter_corners = np.zeros(16)

    rbbox_to_corners(corners1, rbox1)
    rbbox_to_corners(corners2, rbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2, inter_corners)
    sort_vertex_in_convex_polygon(inter_corners, num_intersection)

    return area(inter_corners, num_intersection)

def devRotateIoUEval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = rotate_iou_eval(rbox1, rbox2, criterion)
    
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


def rotate_iou_cpu_eval(boxes, query_boxes, criterion=-1):
    N, K = boxes.shape[0], query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)

    for i in range(N):
        for j in range(K):
            iou[i, j] = rotate_iou_eval(boxes[i], query_boxes[j], criterion)

    return iou
