######################################################################################################
# Robotic Dough Shaping - Main file
######################################################################################################
# Instructions:
#   First, run: roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=wx250s
#   Then change to this directory and run: python3 roll_dough.py
######################################################################################################

import json
import os
import csv
import cv2
import argparse
import numpy as np
from time import time
from datetime import datetime

import rospy
from sensor_msgs.msg import CompressedImage

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface


# In pixels
IMG_SHAPE = (480, 640)
MIN_TARGET_CIRCLE_RADIUS = 50
MAX_TARGET_CIRCLE_RADIUS = 180
# Region of interest
# Note: x and y is swapped in array representation as compared to cv2 image visualization representation!
# Here x and y are in the cv2 image visualization representation
# (0,0) is in the top left corner
ROI = {
    'x_min': 170,
    'y_min': 0,
    'x_max': 540,
    'y_max': 320
}
MIN_COLOR_INTENSITY = 70
MIN_CONTOUR_AREA = 1000


def get_ROI_img(img):
    return img[ROI['y_min']:ROI['y_max'], ROI['x_min']:ROI['x_max']]


def get_current_shape():
    ROI_rgb_img = get_ROI_img(rgb_img)

    # Color filter
    color_mask = np.zeros((*ROI_rgb_img.shape[:2], 3)).astype('uint8')
    color_mask[ROI_rgb_img < MIN_COLOR_INTENSITY] = 255
    overall_color_mask = cv2.bitwise_or(cv2.bitwise_or(color_mask[:, :, 0], color_mask[:, :, 1]), color_mask[:, :, 2])
    
    # Detect contours
    contours, _ = cv2.findContours(overall_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        raise ValueError(f'No contours detected for the current shape!')

    # Take largest contour
    current_shape_contour = sorted(contours, key=lambda c: cv2.contourArea(c))[-1]

    current_shape_area = cv2.contourArea(current_shape_contour)
    if current_shape_area < MIN_CONTOUR_AREA:
        print(f'Warning: the area of the current shape is {current_shape_area} which is less than {MIN_CONTOUR_AREA}')

    # Undo ROI transform
    current_shape_contour[:, 0] += np.array([ROI['x_min'], ROI['y_min']])
    return current_shape_contour


def calculate_iou(target_shape):
    # Calculate target shape mask
    target_shape_mask = np.zeros(IMG_SHAPE)
    if target_shape['type'] != 'circle':
        raise ValueError(f'Unknown target shape {target_shape["type"]}')
    cv2.circle(target_shape_mask, center=target_shape['params']['center'], radius=target_shape['params']['radius'], color=255, thickness=cv2.FILLED)

    # Calculate current shape mask
    current_shape_mask = np.zeros(IMG_SHAPE)
    # drawContours would fail if the contour is not closed (i.e. when a part of the dough is out of ROI)
    cv2.fillPoly(current_shape_mask, [get_current_shape()], color=255)

    # Calculate intersection over union
    intersection = cv2.bitwise_and(current_shape_mask, target_shape_mask)
    union = cv2.bitwise_or(current_shape_mask, target_shape_mask)
    return np.sum(intersection) / np.sum(union)


def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        raise ValueError('Failed to calculate the centroid of the current dough shape!')
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def image2robot_coords(pts):
    # Calculated transform parameters from two points 
    # Ix1, Iy1 = 320, 323
    # Ix2, Iy2 = 384, 326
    # Rx1, Ry1 = 0.0772, 0.0322
    # Rx2, Ry2 = 0.0756, -0.0306
    # A = (Rx2 - Rx1) / (Ix2 - Ix1)
    # B = Rx1 - A*Ix1
    # C = (Ry2 - Ry1) / (Iy2 - Iy1)
    # D = Ry1 - C*Iy1
    # print(f'{A}, {B}, {C}, {D}')
    A, B, C, D = -2.5000000000000066e-05, 0.08520000000000003, -0.02093333333333333, 0.04027500000000002
    return [ (A*x + B, C*y + D) for x, y in pts ]


def calculate_circle_line_intersection(cc, r, p1, p2):
    # Args: circle center, circle radius, 
    #       line point 1 (start point), line point 2 (candidate goal point on dough boundary)
    # Points must be numpy arrays

    # Translate to origin
    x1, y1 = p1 - cc
    x2, y2 = p2 - cc

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
        # Get intersection closer to p2 (i.e. in the direction p1->p2 of the intended roll)
        intersection = i2 if np.linalg.norm(i1 - p2) > np.linalg.norm(i2 - p2) else i1
        # Undo initial translation to origin
        return intersection + cc


def calculate_roll_start_and_goal(method, target_shape, pcl):
    current_shape_contour = get_current_shape()

    # Calculate roll start point S
    pcl_clusters_detected, pcl_clusters = pcl.get_cluster_positions(ref_frame="wx250s/base_link", sort_axis="x", reverse=True)
    if pcl_clusters_detected:
        S = pcl_clusters[0]['position']
    else:
        S = (*calculate_centroid(current_shape_contour), 0.01)

    # Calculate roll goal point G
    G = None
    C = np.array(target_shape['params']['center'])
    R = target_shape['params']['radius']

    if method == 'basic':
        # Find the largest gap between the current and target dough shape
        max_gap = 0
        for P in current_shape_contour[:, 0]:
            intersection = calculate_circle_line_intersection(C, R, S[:2], P)
            if intersection is not None:
                gap = np.linalg.norm(P - intersection)
                if gap > max_gap:
                    max_gap = gap
                    G = P

    if G is None:
        raise ValueError(f'No suitable goal point G found!')

    # Transform from image to robot coordinates
    G = image2robot_coords(G)
    # For now, the goal point has the same z location as the start point
    G = (G[0], G[1], S[2])
    return S, G


def capture_target_shape():
    if not rgb_img or rgb_img.shape != (*IMG_SHAPE, 3):
        raise ValueError(f'No valid RGB image data received from camera!')

    # Detect circle
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
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
        print(f'Warning: multiple circles detected when capturing target shape!')

    circles = np.round(circles[0, :]).astype("int")
    # Undo ROI transform
    x = circles[0][0] + ROI['x_min']
    y = circles[0][1] + ROI['y_min']
    r = circles[0][2]

    target_shape = {
        'type': 'circle',
        'params': {
            'center': (x, y),
            'radius': r
        }
    }
    return target_shape


def rgb_img_callback(ros_msg):
    global rgb_img
    rgb_img = cv2.imdecode(np.fromstring(ros_msg.data, np.uint8), cv2.IMREAD_COLOR)


def main():
    #########################
    # Parse args
    #########################

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, default='basic', help='Choose the method from: "basic", ...')
    parser.add_argument('-tc', '--termination-condition', type=str, default='time', help='Choose either "time" or "iou" termination condition.')
    parser.add_argument('-tv', '--termination-value', type=float, default=10., help='Either maximum time in seconds or minimum IoU based on the termination-condition argument.')
    parser.add_argument('-ld', '--log-dir', type=str, default='~/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_perception/scripts/robotic-dough-shaping/roll_dough.py /logs', help='Path to directory where to save logs.') 
    parser.add_argument('-za', '--z-above', type=float, default=0.15812, help='Vertical distance above the dough immediately before and after rolling.')
    parser.add_argument('-dr', '--disable-robot', type=bool, default=False, help='Will not send any commands to the robot when set to True.')
    parser.add_argument('-dv', '--debug-vision', type=bool, default=False, help='Show vision output when set to True.')
    args = parser.parse_args()
    params = vars(args)

    if args.termination_condition == 'time':
        def terminate(t, iou):
            return t >= args.termination_value
    elif args.termination_condition == 'iou':
        def terminate(t, iou):
            return iou >= args.termination_value
    else:
        raise ValueError(f'Unknown termination condition {args.termination_condition}')

    #########################
    # Initialize robot
    #########################

    bot = InterbotixManipulatorXS("wx250s")#, moving_time=5, accel_time=5)

    def go_to_ready_pose():
        bot.arm.set_ee_pose_components(x=0.0567, y=0, z=0.15812, pitch=np.pi/2)


    #########################
    # Initialize vision
    #########################
        
    pcl = InterbotixPointCloudInterface()
    rospy.init_node('roll_dough', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, rgb_img_callback,  queue_size=1)

    #########################
    # Setup logging
    #########################

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_filename = f'log_{args.termination_condition}_{args.termination_value}_{args.method}_{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}'
    with open(f'{args.log_dir}/{log_filename}.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        #########################
        # Preparation
        #########################

        iteration = 0
        start_time = time()
        iou = 0.

        # Move to ReadyPose
        if not args.disable_robot:
            go_to_ready_pose()

        # Capture the target dough shape
        target_shape = capture_target_shape()
        params.update(target_shape)

        # Calculate current IoU
        iou = calculate_iou(target_shape)

        # Logging
        time_elapsed = time() - start_time
        print('=' * 120)
        print(json.dumps(vars(args), indent=2))
        print(f'Preparation: \t Time: {time_elapsed:5.3f} s\t IoU: {iou:.3f}')
        print('=' * 120)
        # First line is a dictionary of parameters
        csv_writer.writerow([json.dumps(params)])
        # Second line is the header
        csv_writer.writerow(['Iteration', 'Time', 'IoU'])
        csv_writer.writerow([iteration, time_elapsed, iou])

        #########################
        # Iterative procedure
        #########################
        iteration = 1
        while not terminate(time() - start_time, iou):

            # Calculate the roll start point S and roll goal point G
            S, G = calculate_roll_start_and_goal(args.method, target_shape, pcl)
            print(f'Calculated roll start point S: {S}')
            print(f'Calculated roll goal point G: {G}')

            # Calculate the angle of the direction S -> G
            # No need to use arctan2 due to symmetry
            yaw_SG = np.arctan((G[1] - S[1]) / (G[0] - S[0]))
            print(f'Calculated the angle of the direction S -> G: {yaw_SG * 180 / np.pi}')

            if not args.disable_robot:

                # Move to AboveBeforePose
                bot.arm.set_ee_pose_components(x=S[0], y=S[1], z=S[2] + args.z_above, pitch=np.pi/2, yaw=yaw_SG)

                # Move to TouchPose
                bot.arm.set_ee_pose_components(x=S[0], y=S[1], z=S[2], pitch=np.pi/2, yaw=yaw_SG)

                # Perform roll to point G
                bot.arm.set_ee_pose_components(x=G[0], y=G[1], z=G[2], pitch=np.pi/2, yaw=yaw_SG)

                # Move to AboveAfterPose
                bot.arm.set_ee_pose_components(x=G[0], y=G[1], z=G[2] + args.z_above, pitch=np.pi/2, yaw=yaw_SG)

                # Move to ReadyPose
                go_to_ready_pose()

            # Calculate current IoU
            iou = calculate_iou(target_shape)

            # Logging
            time_elapsed = time() - start_time
            print(f'Iteration: {iteration:7d}\t Time: {time_elapsed:5.3f} s\t IoU: {iou:.3f}')
            print('=' * 120)
            csv_writer.writerow([iteration, time_elapsed, iou])

            iteration += 1


if __name__=='__main__':
    main()
