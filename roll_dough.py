##################################################################################################################################
# Robotic Dough Shaping - Roll Dough GUI Application
#   Project for Robot Manipulation (CS 6751), Cornell University, Spring 2022
#   Group members: Di Ni, Xi Deng, Zeqi Gu, Henry Zheng, Jan (Janko) Ondras
##################################################################################################################################
# Authors: 
#   Jan (Janko) Ondras (jo951030@gmail.com)
#   Zeqi Gu (contributed to circle detection and color filtering)
#   Henry Zheng (helped with testing)
##################################################################################################################################
# Instructions:
#   In one terminal:
#     source ~/interbotix_ws/devel/source.bash
#     roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=wx250s use_pointcloud_tuner_gui:=true
#   In another terminal:
#     cd <directory-this-file-is-located-in>
#     python3 roll_dough.py -dr -vo -vw -m play-doh -sm highest-point -em target
##################################################################################################################################

import json
import os
import csv
import cv2
import argparse
import numpy as np
from scipy.linalg import inv
from time import time, sleep
from datetime import datetime
from types import GeneratorType
from pynput.keyboard import Key, Controller

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from sklearn.neighbors import KDTree

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface


WINDOW_ID = 'Robotic Dough Shaping'
IMG_SHAPE = (480, 640) # pixels
# Region of interest
# Note: x and y is swapped in array representation as compared to cv2 image visualization representation!
# Here x and y are in the cv2 image visualization representation and (0,0) is in the top left corner
ROI = {
    'x_min': 170,   # pixels
    'y_min': 0,     # pixels
    'x_max': 540,   # pixels
    'y_max': 320    # pixels
}
# Target shape detection parameters
MIN_TARGET_CIRCLE_RADIUS = 40   # pixels for 3.5 inch (50 pixels => only circles with diameters 4 inch or more will be detected)
MAX_TARGET_CIRCLE_RADIUS = 180  # pixels
# Current shape detection parameters
MIN_COLOR_INTENSITY = 70
MIN_CONTOUR_AREA = 1000 # pixels^2
INCHES_PER_PIXEL = 0.03926950683443001

# Fraction of dough height reached at the roll start point
DOUGH_HEIGHT_START_POINT_CONTRACTION_RATIO = -0.4
DOUGH_HEIGHT_END_POINT_CONTRACTION_RATIO = -0.4
# Manually measured correction added to the z coordinate in the point cloud to robot coordinate transformation
Z_CORRECTION = 0.005468627771547538
# Minimum z value to move the robot to
Z_MIN = 0.06 # prev contact value = 0.065 meters
# Maximum depth in point cloud coordinates
MAX_DEPTH = 0.60100262777 # TODO: might need to check 0.60642844 inverse

ROLLING_PIN_WIDTH = 0.061 - 0.002  # meters (compensate for looseness)
ROLLING_PIN_RADIUS = 0.017 - 0.004 - 0.003 # meters (compensate for looseness and pin curvature)
SHRINK_HEIGHT_OFFSET = 0.02 # meters
SHRINK_PIXEL_TOLERANCE = 4  # 2 pixels (corresponds to about 2 mm which is the width of the target shape outline)

# Transformation matrix: camera_depth_optical_frame -> wx250s/base_link
T_pc2ro = np.array([[ 0.08870469, -0.9955393 , -0.03213995,  0.175572                ],
                    [-0.99605734, -0.08862224, -0.0039837 ,  0.0246511               ],
                    [ 0.00111761,  0.03236661, -0.99947544,  0.595534 + Z_CORRECTION ],
                    [ 0.        ,  0.        ,  0.        ,  1.                      ]])
# Transformation matrix: camera_color_optical_frame -> wx250s/base_link
T_co2ro = np.array([[ 0.07765632, -0.99658459, -0.02808279,  0.17434133],
                    [-0.99697964, -0.07765513, -0.00113465,  0.0391628 ],
                    [-0.00105   ,  0.02808608, -0.99960496,  0.59613603],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
# Transformation matrix: image 2D -> point cloud 2D
T_im2pc = np.array([[ 0.0009652963665596975, 0.                   , -0.32951992162237304 ],
                    [ 0.                   , 0.0009512554830209385, -0.22369287797508308 ],
                    [ 0.                   , 0.                   ,  1.                  ]])
T_pc2im = inv(T_im2pc)

TERMINATION_TIME_UPPER_BOUND = 500 # seconds

# Global variables
RGB_IMG = None
POINT_CLOUD = None


def rgb_img_callback(ros_msg):
    global RGB_IMG
    RGB_IMG = cv2.imdecode(np.frombuffer(ros_msg.data, np.uint8), cv2.IMREAD_COLOR)


def point_cloud_callback(ros_msg):
    global POINT_CLOUD
    POINT_CLOUD = point_cloud2.read_points_list(ros_msg, skip_nans=True, field_names=('x', 'y', 'z'))


def get_ROI_img(img):
    return img[ROI['y_min']:ROI['y_max'], ROI['x_min']:ROI['x_max']]


def capture_target_shape(visual_output, keyboard, visual_wait):
    if type(RGB_IMG) != np.ndarray or RGB_IMG.shape != (*IMG_SHAPE, 3):
        raise ValueError(f'No valid RGB image data received from camera!')

    # Detect circle
    gray_img = cv2.cvtColor(RGB_IMG, cv2.COLOR_BGR2GRAY)
    ROI_gray_img = get_ROI_img(gray_img)
    circles = cv2.HoughCircles(ROI_gray_img, 
                               method=cv2.HOUGH_GRADIENT, 
                               dp=1, 
                               minDist=MAX_TARGET_CIRCLE_RADIUS,
                               param1=50,
                               param2=50,
                               minRadius=MIN_TARGET_CIRCLE_RADIUS,
                               maxRadius=MAX_TARGET_CIRCLE_RADIUS)
    if circles is None:
        raise ValueError(f'Failed to detect circular target shape!')
    if len(circles[0]) > 1:
        print(f'Warning: multiple circles detected when capturing target shape! Taking the largest one.')

    # Take the largest circle
    largest_circle = sorted(circles[0], key=lambda c: c[2])[-1]
    largest_circle = np.round(largest_circle)

    # Undo ROI transform
    x = int(largest_circle[0]) + ROI['x_min']
    y = int(largest_circle[1]) + ROI['y_min']
    r = int(largest_circle[2])

    if visual_output:
        debug_img = RGB_IMG.copy()
        # Draw the region of interest
        cv2.rectangle(debug_img, (ROI['x_min'], ROI['y_min']), (ROI['x_max'], ROI['y_max']), color=(0, 255, 0), thickness=1)
        # Draw the target shape
        cv2.circle(debug_img, (x, y), r, color=(0, 0, 255), thickness=1)
        # Draw text
        cv2.rectangle(debug_img, (5, 5), (160, 70), color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(debug_img, 'ROI', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, 'Target shape', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow(WINDOW_ID, debug_img)
        cv2.setWindowTitle(WINDOW_ID, WINDOW_ID + ' - Target shape')
        if not visual_wait: keyboard.press(Key.space)
        cv2.waitKey(0)

    return {
        'type': 'circle',
        'params': {
            'center': (x, y),
            'radius': r
        }
    }


def capture_current_shape(visual_output, keyboard, visual_wait):
    ROI_rgb_img = get_ROI_img(RGB_IMG)

    # Color filter
    color_mask = np.zeros((*ROI_rgb_img.shape[:2], 3)).astype('uint8')
    color_mask[ROI_rgb_img < MIN_COLOR_INTENSITY] = 255
    overall_color_mask = cv2.bitwise_or(cv2.bitwise_or(color_mask[:, :, 0], color_mask[:, :, 1]), color_mask[:, :, 2])
    
    # Detect contours
    contours, _ = cv2.findContours(overall_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        raise ValueError(f'No contours detected for the current shape!')

    # Take the largest contour
    current_shape_contour = sorted(contours, key=lambda c: cv2.contourArea(c))[-1]

    current_shape_area = cv2.contourArea(current_shape_contour)
    if current_shape_area < MIN_CONTOUR_AREA:
        print(f'Warning: the area of the current shape is {current_shape_area} which is less than {MIN_CONTOUR_AREA}')

    # Undo ROI transform
    current_shape_contour[:, 0] += np.array([ROI['x_min'], ROI['y_min']])

    if visual_output:
        debug_img = RGB_IMG.copy()
        # Draw the region of interest
        cv2.rectangle(debug_img, (ROI['x_min'], ROI['y_min']), (ROI['x_max'], ROI['y_max']), color=(0, 255, 0), thickness=1)
        # Draw the current shape
        # drawContours will fail if the contour is not closed (i.e. when a part of the dough is out of ROI)
        # cv2.drawContours(debug_img, [current_shape_contour], color=(255, 0, 0), thickness=1)
        cv2.polylines(debug_img, [current_shape_contour], isClosed=True, color=(255, 0, 0), thickness=1)
        # Draw text
        cv2.rectangle(debug_img, (5, 5), (160, 100), color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(debug_img, 'ROI', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, 'Current shape', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow(WINDOW_ID, debug_img)
        cv2.setWindowTitle(WINDOW_ID, WINDOW_ID + ' - Current dough shape')
        if not visual_wait: keyboard.press(Key.space)
        cv2.waitKey(0)
    
    return current_shape_contour


def calculate_centroid_2d(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        raise ValueError('Failed to calculate the centroid of the current dough shape!')
    return np.array([ int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) ])


def calculate_centroid_3d(roi=None):
    if len(POINT_CLOUD) == 0:
        raise ValueError(f'No valid point cloud data received from camera!')

    if roi is None:
        return np.mean(POINT_CLOUD, axis=0)

    # If the point cloud filter is not tuned, need to filter out points outside of region of interest (current shape contour)
    #     Create boolean mask from current_shape_contour
    #     For each point in point cloud: 
    #         Transform to image space
    #         Check if it is inside
    roi_mask = np.zeros(IMG_SHAPE, dtype="uint8")
    cv2.fillPoly(roi_mask, [roi], color=255)
    num_points = 0
    centroid = np.zeros(3)
    for pt in POINT_CLOUD:
        x, y = np.dot(T_pc2im, np.array([pt[0], pt[1], 1]))[:2]
        x, y = round(x), round(y)
        if x >= 0 and y >= 0 and y < roi_mask.shape[0] and x < roi_mask.shape[1] and roi_mask[y, x] == 255:
            num_points += 1
            centroid += np.array(pt)
    return centroid / num_points


def calculate_circle_line_intersection(cc, r, p1, p2):
    # Args: circle center, circle radius, 
    #       line point 1 (roll start point), line point 2 (candidate end point on dough boundary)
    # Points must be numpy arrays

    # Translate to origin
    p1_orig = p1 - cc
    p2_orig = p2 - cc

    x1, y1 = p1_orig
    x2, y2 = p2_orig

    # General form line equation parameters
    a = y2 - y1
    b = x1 - x2
    c = y1 * x2 - x1 * y2

    # Below inspired by https://cp-algorithms.com/geometry/circle-line-intersection.html
    EPS = 1e-6
    a2b2 = a*a + b*b
    x0 = - a*c / a2b2
    y0 = - b*c / a2b2
    delta = c*c - r*r*a2b2
    if delta > EPS:
        # No intersecion points
        return None

    elif abs(delta) < EPS:
        # One intersection point
        # Undo initial translation to origin
        return np.array([x0, y0]) + cc

    else:
        # Two intersection points
        mult = np.sqrt((r*r - c*c/a2b2) / a2b2)
        i1 = np.array([x0 + b*mult, y0 - a*mult])
        i2 = np.array([x0 - b*mult, y0 + a*mult])

        # Get only the intersections in the direction p1->p2 of the intended roll and further from p1 than |p1,p2|
        # If both intersections satisfy this condition, take the one further from p1
        dist_p1p2 = np.linalg.norm(p2_orig - p1_orig)
        intersection = None
        max_dist_p1i = 0
        for i in [i1, i2]:
            if (i[0] - x1) * (x2 - x1) > 0 and (i[1] - y1) * (y2 - y1) > 0:
                dist_p1i = np.linalg.norm(i - p1_orig)
                if dist_p1i > dist_p1p2 and dist_p1i > max_dist_p1i:
                    max_dist_p1i = dist_p1i
                    intersection = i

        # Undo initial translation to origin
        return intersection + cc if intersection is not None else None


def get_dough_depth(pt):
    # Get dough depth in point cloud coordinates in the proximity of a given point in 2D point cloud coordinates

    if len(POINT_CLOUD) == 0:
        raise ValueError(f'No valid point cloud data received from camera!')
    
    # Find k closest points in the point cloud (in terms of x and y) and get their average distance from camera
    point_cloud_arr = np.array(POINT_CLOUD)
    _, idx = KDTree(point_cloud_arr[:, :2]).query([pt], k=3)
    depth = np.mean(point_cloud_arr[idx][0, :, 2])

    return depth


def calculate_max_dough_height_and_point():
    if len(POINT_CLOUD) == 0:
        raise ValueError(f'No valid point cloud data received from camera!')

    # Get highest dough point in 3D point cloud coordinates
    highest_dough_pt_pc = sorted(POINT_CLOUD, key=lambda pt: pt[2])[0]

    max_dough_height = np.dot(T_pc2ro, np.array([*highest_dough_pt_pc, 1]))[2]
    if max_dough_height < 0:
        print(f'Warning: The estimated maximum dough height is {max_dough_height:.4f} m. Setting to 0 m.')
        max_dough_height = 0
    print(f'Maximum dough height: {max_dough_height:.4f} m')

    return max_dough_height, highest_dough_pt_pc


def calculate_roll_start_and_end(params, current_shape_contour, iou, max_dough_height, highest_dough_pt_pc, iteration, pcl, keyboard):
    # pcl_clusters_detected, pcl_clusters = pcl.get_cluster_positions(ref_frame="wx250s/base_link", sort_axis="x", reverse=True)
    # if pcl_clusters_detected:
    #     S = pcl_clusters[0]['position']
    # else:
    #     S = (*calculate_centroid_2d(current_shape_contour), 0.01)
    H_pc = highest_dough_pt_pc
    C = np.array(params['target_shape']['params']['center'])
    R = params['target_shape']['params']['radius']

    # Calculate the roll start point S (in 2D)
    if params['start_method'] == 'centroid-2d':
        S_im = calculate_centroid_2d(current_shape_contour)

        # Transform from image to 2D point cloud coordinates
        S_pc = np.dot(T_im2pc, np.array([*S_im, 1]))[:2]
        # print(S_im, '- im2pc ->', S_pc)

        # Get dough depth in point cloud coordinates in the proximity of a given point in point cloud coordinates
        depth_pc = get_dough_depth(S_pc)
        # print('DEPTH', depth)
        
    elif params['start_method'] == 'centroid-3d':
        # If the point cloud filter is not tuned, need to filter out points outside of current_shape_contour
        *S_pc, depth_pc = calculate_centroid_3d(roi=current_shape_contour)

        # Transform from 2D point cloud to image coordinates
        S_im = np.dot(T_pc2im, np.array([*S_pc, 1])).astype(int)[:2]
        # print(S_pc, '- pc2im ->', S_im)

    elif params['start_method'] == 'highest-point':
        *S_pc, depth_pc = H_pc

        # Transform from 2D point cloud to image coordinates
        S_im = np.dot(T_pc2im, np.array([*S_pc, 1])).astype(int)[:2]
        # print(S_pc, '- pc2im ->', S_im)

    # When shrink action is enabled
    do_shrink = False
    if params['enable_shrink']:
        # Find a current shape point that is outside of the target shape and furthest from the target shape center
        furthest_outside_pt = None
        furthest_outside_pt_dist_squared = 0
        for P in current_shape_contour[:, 0]:
            dist_CP_squared = sum(np.square(P - C))
            if dist_CP_squared > (R + SHRINK_PIXEL_TOLERANCE)**2:
                if furthest_outside_pt_dist_squared < dist_CP_squared:
                    furthest_outside_pt = P
                    furthest_outside_pt_dist_squared = dist_CP_squared

        # If a point outside of the target shape was found:
        #     Calculate the roll end point E (in 2D)
        #     Correct the start point S (in 2D)
        if furthest_outside_pt is not None:
            do_shrink = True
            print(f'Current dough shape exceeds the target shape! => Applying the SHRINK action')
            
            # Set the end point E_im to a closest point on the target shape in the direction furthest_outside_pt->C
            # This approach should work with the highest-point start method
            u = (C - furthest_outside_pt) / np.linalg.norm(C - furthest_outside_pt)
            dist = np.sqrt(furthest_outside_pt_dist_squared) - (R - SHRINK_PIXEL_TOLERANCE)
            E_im = furthest_outside_pt + dist*u
            # Alternative: Set the end point E_im to a point on the line furthest_outside_pt->C at the distance |S_im,C| from furthest_outside_pt
            # u = (C - furthest_outside_pt) / np.linalg.norm(C - furthest_outside_pt)
            # dist = np.linalg.norm(C - S_im)
            # E_im = furthest_outside_pt + dist*u
            # Alternative: Set the end point E_im in the direction S_im->C at the distance |S_im,C| from furthest_outside_pt
            # E_im = furthest_outside_pt + (C - S_im)
            E_im = E_im.astype(int)

            # Correct the start point for shrink action
            S_im = furthest_outside_pt

            # Transform from image to 2D point cloud coordinates
            S_pc = np.dot(T_im2pc, np.array([*furthest_outside_pt, 1]))[:2]
            depth_pc = MAX_DEPTH

    # Calculate the roll end point E (in 2D) when shrink action is disabled or current shape is entirely inside of the target shape
    if not do_shrink:
        E_im = None
        # Find the largest gap between the current and target dough shape
        max_gap = 0
        # [For debugging] Save circle-line intersection points
        # intersection_pts = []
        for P in current_shape_contour[:, 0]:
            intersection = calculate_circle_line_intersection(C, R, S_im, P)
            if intersection is not None:
                # [For debugging] Save circle-line intersection points
                # intersection_pts.append(intersection.astype(int))
                gap = np.linalg.norm(P - intersection)
                if gap > max_gap:
                    max_gap = gap
                    if params['end_method'] == 'current':
                        E_im = P
                    elif params['end_method'] == 'target':
                        E_im = intersection.astype(int)

        if E_im is None:
            print(f'No suitable roll end point E found! The dough might be already covering the whole target shape.')
            return None

    # Transform from image to 2D point cloud coordinates
    E_pc = np.dot(T_im2pc, np.array([*E_im, 1]))[:2]
    # print(E_im, '- im2pc ->', E_pc)

    # Transform from point cloud to robot coordinates
    S_ro = np.dot(T_pc2ro, np.array([S_pc[0], S_pc[1], depth_pc, 1]))[:3]
    E_ro = np.dot(T_pc2ro, np.array([E_pc[0], E_pc[1], depth_pc, 1]))[:3]
    # print(S_pc, '- pc2ro ->', S_ro)
    # print(E_pc, '- pc2ro ->', E_ro)

    dough_height_at_S = S_ro[2]
    if dough_height_at_S < 0:
        print(f'Warning: The estimated dough height at the roll start point S is {dough_height_at_S:.4f} m. Setting to 0 m.')
        dough_height_at_S = 0
    print(f'Dough height at roll start point S: {dough_height_at_S:.4f} m')

    height = dough_height_at_S
    if do_shrink:
        # Set z coordinate to touch the base
        height = SHRINK_HEIGHT_OFFSET

        # Offset the start and end point by the rolling pin radius/width
        # This is not shown in the visual output!
        u = (E_ro[:2] - S_ro[:2]) / np.linalg.norm(E_ro[:2] - S_ro[:2])
        dist = ROLLING_PIN_WIDTH/2 if params['side_shrink'] else ROLLING_PIN_RADIUS
        S_ro[:2] = S_ro[:2] - dist*u
        E_ro[:2] = E_ro[:2] - dist*u

    # Set z to the fraction of dough height that is reached at the roll start point
    # Offset by the minimum z value robot can be moved to
    S_z_ro = Z_MIN + DOUGH_HEIGHT_START_POINT_CONTRACTION_RATIO * height
    E_z_ro = Z_MIN + DOUGH_HEIGHT_END_POINT_CONTRACTION_RATIO * height

    S_ro = (S_ro[0], S_ro[1], S_z_ro)
    E_ro = (E_ro[0], E_ro[1], E_z_ro)

    # Calculate the angle of the direction S -> E
    # No need to use arctan2 due to symmetry
    yaw_SE = np.arctan((E_ro[1] - S_ro[1]) / (E_ro[0] - S_ro[0]))

    # Change rotation of the rolling pin by 90 degrees if shrinking with edge
    if do_shrink and params['side_shrink']:
        yaw_SE = yaw_SE - (np.pi/2) if yaw_SE > 0 else yaw_SE + (np.pi/2)

    print(f'Calculated roll start point S: {[f"{i:.3f}" for i in S_ro]} m')
    print(f'Calculated roll end point E: {[f"{i:.3f}" for i in E_ro]} m')
    print(f'Calculated the angle of the direction S -> E: {yaw_SE * 180 / np.pi:.3f}' + u'\xb0')

    if params['visual_output']:
        debug_img = RGB_IMG.copy()
        # Draw the region of interest
        cv2.rectangle(debug_img, (ROI['x_min'], ROI['y_min']), (ROI['x_max'], ROI['y_max']), color=(0, 255, 0), thickness=1)
        # Draw the target shape
        cv2.circle(debug_img, tuple(C), R, color=(0, 0, 255), thickness=1)
        # Draw the current shape
        # drawContours will fail if the contour is not closed (i.e. when a part of the dough is out of ROI)
        # cv2.drawContours(debug_img, [current_shape_contour], color=(255, 0, 0), thickness=1)
        cv2.polylines(debug_img, [current_shape_contour], isClosed=True, color=(255, 0, 0), thickness=1)
        # [For debugging] Draw intersection points
        # for i in intersection_pts:
        #     cv2.circle(debug_img, i, 1, color=(0, 255, 0), thickness=2)
        # Draw the planned roll path
        cv2.arrowedLine(debug_img, tuple(S_im), tuple(E_im), color=(0, 0, 0), thickness=2, tipLength=0.3)
        # Draw text
        cv2.rectangle(debug_img, (5, 5), (165, 460), color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(debug_img, f'Method:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'- {params["start_method"]}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'- end at {params["end_method"]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'- {("side shrink" if params["side_shrink"] else "forward shrink") if params["enable_shrink"] else "shrink disabled"}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'Iteration: {iteration:6d}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'Time:  {(time() - params["start_time"]):7.1f} s', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(debug_img, (10, 190), (155, 190), color=(150, 150, 150), thickness=1)
        cv2.putText(debug_img, 'ROI', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, 'Target shape', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, 'Current shape', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'IoU = {iou:.3f}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'Dough height:', (10, 330), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'- max: {max_dough_height:.3f} m', (10, 360), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'- at S: {dough_height_at_S:.3f} m', (10, 390), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, 'Roll angle', (10, 420), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'{yaw_SE * 180 / np.pi:11.2f} deg', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow(WINDOW_ID, debug_img)
        cv2.setWindowTitle(WINDOW_ID, WINDOW_ID + ' - Planned roll trajectory')
        if not params['visual_wait']: keyboard.press(Key.space)
        cv2.waitKey(0)
    
    return dough_height_at_S, S_ro, E_ro, yaw_SE


def calculate_iou(target_shape, current_shape_contour, visual_output, keyboard, visual_wait):
    # Calculate target shape mask
    target_shape_mask = np.zeros(IMG_SHAPE, dtype="uint8")
    if target_shape['type'] != 'circle':
        raise ValueError(f'Unknown target shape {target_shape["type"]}')
    cv2.circle(target_shape_mask, center=target_shape['params']['center'], radius=target_shape['params']['radius'], color=255, thickness=cv2.FILLED)

    # Calculate current shape mask
    current_shape_mask = np.zeros(IMG_SHAPE, dtype="uint8")
    # drawContours would fail if the contour is not closed (i.e. when a part of the dough is out of ROI)
    cv2.fillPoly(current_shape_mask, [current_shape_contour], color=255)

    # Calculate intersection over union
    intersection = cv2.bitwise_and(current_shape_mask, target_shape_mask)
    union = cv2.bitwise_or(current_shape_mask, target_shape_mask)
    iou = np.sum(intersection) / np.sum(union)

    if visual_output:
        debug_img = RGB_IMG.copy()
        # Draw the region of interest
        cv2.rectangle(debug_img, (ROI['x_min'], ROI['y_min']), (ROI['x_max'], ROI['y_max']), color=(0, 255, 0), thickness=1)
        # Draw the target shape
        cv2.circle(debug_img, center=target_shape['params']['center'], radius=target_shape['params']['radius'], color=(0, 0, 255), thickness=1)
        # Draw the current shape
        # drawContours will fail if the contour is not closed (i.e. when a part of the dough is out of ROI)
        # cv2.drawContours(debug_img, [current_shape_contour], color=(255, 0, 0), thickness=1)
        cv2.polylines(debug_img, [current_shape_contour], isClosed=True, color=(255, 0, 0), thickness=1)
        # Draw text
        cv2.rectangle(debug_img, (5, 5), (160, 130), color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(debug_img, 'ROI', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, 'Target shape', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, 'Current shape', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, f'IoU = {iou:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow(WINDOW_ID, debug_img)
        cv2.setWindowTitle(WINDOW_ID, WINDOW_ID + ' - Intersection over union')
        if not visual_wait: keyboard.press(Key.space)
        cv2.waitKey(0)

    return iou


def evaluate_and_plan(params, iteration, pcl, keyboard):
        # Calculate current shape contour
        current_shape_contour = capture_current_shape(visual_output=False, keyboard=keyboard, visual_wait=params['visual_wait'])
        # Calculate current IoU
        iou = calculate_iou(params['target_shape'], current_shape_contour, visual_output=False, keyboard=keyboard, visual_wait=params['visual_wait'])
        max_dough_height, highest_dough_pt_pc = calculate_max_dough_height_and_point()
        # Calculate the roll start point S, the roll end point E, and the angle of the direction S -> E
        plan_result = calculate_roll_start_and_end(params, current_shape_contour, iou, max_dough_height, highest_dough_pt_pc, iteration, pcl, keyboard)

        return  iou, max_dough_height, plan_result


def main():
    #########################
    # Parse args
    #########################

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--material', type=str, required=True, choices=['play-doh', 'plasticine', 'kinetic-sand'], help='Choose material to be rolled.')
    parser.add_argument('-sm', '--start-method', type=str, required=True, choices=['centroid-2d', 'centroid-3d', 'highest-point'],
                        help='Choose the roll start point calculation method from: "centroid-2d", "centroid-3d", and "highest-point"')
    parser.add_argument('-em', '--end-method', type=str, required=True, choices=['current', 'target'],
                        help='Choose the roll end point calculation method from: "current" (current shape outline), "target" (target shape outline)')
    parser.add_argument('-es', '--enable-shrink', action='store_true', default=False, help='Shrink dough to the target shape area when it expands outside of the target shape.')
    parser.add_argument('-ss', '--side-shrink', action='store_true', default=False, help='Perform the shrink action with the side edge of the rolling pin. Otherwise, use the forward edge as for the expand action. Applicable only when shrinking is enabled.')               
    parser.add_argument('-tc', '--termination-condition', type=str, default='time', choices=['time', 'iou'], help='Choose either "time" or "iou" termination condition.')
    parser.add_argument('-tv', '--termination-value', type=float, default=10., help='Either maximum time in seconds or minimum IoU based on the termination-condition argument.')
    parser.add_argument('-ld', '--log-dir', type=str, default='./logs', help='Path to directory where to save logs (relative to this file).') 
    parser.add_argument('-za', '--z-above', type=float, default=0.15812, help='Height above the base/table immediately before and after rolling.')
    parser.add_argument('-dr', '--disable-robot', action='store_true', default=False, help='Will not send any commands to the robot when set to True.')
    parser.add_argument('-vo', '--visual-output', action='store_true', default=False, help='Show visual output when set to True.')
    parser.add_argument('-vw', '--visual-wait', action='store_true', default=False, help='Wait for key press to proceed when visual output is enabled.')
    args = parser.parse_args()
    params = vars(args)

    if args.termination_condition == 'time':
        def terminate(t, iou):
            return t >= args.termination_value
    elif args.termination_condition == 'iou':
        def terminate(t, iou):
            return iou >= args.termination_value or t > TERMINATION_TIME_UPPER_BOUND
    else:
        raise ValueError(f'Unknown termination condition {args.termination_condition}')

    keyboard = Controller()

    #########################
    # Initialize vision
    #########################
    
    pcl = InterbotixPointCloudInterface()
    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, rgb_img_callback,  queue_size=1)
    rospy.Subscriber('/pc_filter/pointcloud/filtered', PointCloud2, point_cloud_callback, queue_size=1)
    # Do not initialize ROS node here, as this is done in InterbotixManipulatorXS (node name /wx250s_robot_manipulation)
    # rospy.init_node('roll_dough', anonymous=True)
    
    #########################
    # Initialize robot
    #########################

    bot = InterbotixManipulatorXS("wx250s", accel_time=0.1)#, moving_time=2, accel_time=0.3)

    def go_to_ready_pose(): 
        bot.arm.set_ee_pose_components(x=0.057, y=0, z=0.15812, pitch=np.pi/2, moving_time=0.5)

    #########################
    # Preparation
    #########################

    iteration = 0
    start_time = time()
    params['start_time'] = start_time

    # Move to ReadyPose
    if not args.disable_robot:
        go_to_ready_pose()

    # Capture the target dough shape
    target_shape = capture_target_shape(visual_output=False, keyboard=keyboard, visual_wait=args.visual_wait)
    params['target_shape'] = target_shape

    #########################
    # Setup logging
    #########################

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    target_shape_diameter_estimate = 2 * target_shape['params']['radius'] * INCHES_PER_PIXEL
    log_filename = f'log_{args.material}_{target_shape_diameter_estimate:.1f}_{args.start_method}_{args.end_method}_{"shrink-enabled" if args.enable_shrink else "shrink-disabled"}_{args.termination_condition}_{args.termination_value}_{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}'
    with open(f'{args.log_dir}/{log_filename}.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Logging
        time_elapsed = time() - start_time
        params['preparation_time'] = time_elapsed
        print('=' * 120)
        print(f'Params:\n{json.dumps(params, indent=2)}')
        print(f'Preparation: \t Time: {time_elapsed:5.3f} s')
        print('=' * 120)
        # First line is a dictionary of parameters
        csv_writer.writerow([json.dumps(params)])
        # Second line is the header
        csv_writer.writerow(['Iteration', 'Time (s)', 'IoU', 'Max dough height (m)', 'Dough height at S (m)'])

        # Evaluate and plan: calculate current shape contour, evaluate IoU, calculate maximum dough height, and calculate the roll start point S, the roll end point E, and the angle of the direction S -> E
        iou, max_dough_height, plan_result = evaluate_and_plan(params, iteration, pcl, keyboard)
        if plan_result is None:
            raise ValueError(f'Failed to make a plan!')
        dough_height_at_S, S, E, yaw_SE = plan_result

        # Logging
        time_elapsed = time() - start_time
        print(f'Iteration: {iteration:7d}\t Time: {time_elapsed:5.3f} s\t IoU: {iou:.3f}\t Maximum dough height: {max_dough_height:.4f} m\t Dough height at S: {dough_height_at_S:.4f} m')
        print('=' * 120)
        csv_writer.writerow([iteration, time_elapsed, iou, max_dough_height, dough_height_at_S])

        #########################
        # Iterative procedure
        #########################
        iteration = 1
        while plan_result is not None and not terminate(time() - start_time, iou):

            if not args.disable_robot:

                # Move to AboveBeforePose
                bot.arm.set_ee_pose_components(x=S[0], y=S[1], z=args.z_above, pitch=np.pi/2, yaw=yaw_SE, moving_time=0.5)

                # Move to TouchPose
                bot.arm.set_ee_pose_components(x=S[0], y=S[1], z=S[2], pitch=np.pi/2, yaw=yaw_SE, moving_time=1.5)
                sleep(0.5)

                # Perform roll to point E
                bot.arm.set_ee_pose_components(x=E[0], y=E[1], z=E[2], pitch=np.pi/2, yaw=yaw_SE, moving_time=2.5)

                # Move to AboveAfterPose
                bot.arm.set_ee_pose_components(x=E[0], y=E[1], z=args.z_above, pitch=np.pi/2, yaw=yaw_SE, moving_time=1)

                # Move to ReadyPose
                go_to_ready_pose()

            # Evaluate and plan: calculate current shape contour, evaluate IoU, calculate maximum dough height, and calculate the roll start point S, the roll end point E, and the angle of the direction S -> E
            iou, max_dough_height, plan_result = evaluate_and_plan(params, iteration, pcl, keyboard)
            if plan_result is not None:
                dough_height_at_S, S, E, yaw_SE = plan_result

            # Logging
            time_elapsed = time() - start_time
            print(f'Iteration: {iteration:7d}\t Time: {time_elapsed:5.3f} s\t IoU: {iou:.3f}\t Maximum dough height: {max_dough_height:.4f} m\t Dough height at S: {dough_height_at_S:.4f} m')
            print('=' * 120)
            csv_writer.writerow([iteration, time_elapsed, iou, max_dough_height, dough_height_at_S])

            iteration += 1

    cv2.waitKey()
    cv2.destroyAllWindows() 


if __name__=='__main__':
    main()
