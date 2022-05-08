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


def calculate_goal_point(S, target_shape):
    # dict(zip(['x', 'y', 'z'], [?, ?, ?]))
    current_shape_contour = get_current_shape()
    #TODO
    return {
        'x': x,
        'y': y,
        'z': 0
    }


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
    pcl = InterbotixPointCloudInterface()

    def go_to_ready_pose():
        bot.arm.set_ee_pose_components(x=0.0567, y=0, z=0.15812, pitch=np.pi/2)


    #########################
    # Initialize vision
    #########################
        
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

            # Detect the roll starting point S
            #TODO try alternative?
            clusters_detected, clusters = pcl.get_cluster_positions(ref_frame="wx250s/base_link", sort_axis="x", reverse=True)
            if clusters_detected:
                S = dict(zip(['x', 'y', 'z'], clusters[0]['position']))
                print(f'Detected roll start point S: {S} from pointcloud')
            else:
                S = None

            # Calculate the roll goal point G
            G = calculate_goal_point(S, target_shape)
            G['z'] = S['z']
            print(f'Calculated roll goal point G: {G}')

            # Calculate the angle of the direction S -> G
            # No need to use arctan2 due to symmetry
            yaw_SG = np.arctan((G['y'] - S['y']) / (G['x'] - S['x']))
            print(f'Calculated the angle of the direction S -> G: {yaw_SG * 180 / np.pi}')

            if not args.disable_robot:

                # Move to AboveBeforePose
                bot.arm.set_ee_pose_components(x=S['x'], y=S['y'], z=S['z'] + args.z_above, pitch=np.pi/2, yaw=yaw_SG)

                # Move to TouchPose
                bot.arm.set_ee_pose_components(x=S['x'], y=S['y'], z=S['z'], pitch=np.pi/2, yaw=yaw_SG)

                # Perform roll to point G
                bot.arm.set_ee_pose_components(x=G['x'], y=G['y'], z=G['z'], pitch=np.pi/2, yaw=yaw_SG)

                # Move to AboveAfterPose
                bot.arm.set_ee_pose_components(x=G['x'], y=G['y'], z=G['z'] + args.z_above, pitch=np.pi/2, yaw=yaw_SG)

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
