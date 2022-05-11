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
from types import GeneratorType

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from sklearn.neighbors import KDTree

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface


WINDOW_TITLE_PREFIX = 'Robotic Dough Shaping - '
IMG_SHAPE = (480, 640)
# Region of interest (in pixels)
# Note: x and y is swapped in array representation as compared to cv2 image visualization representation!
# Here x and y are in the cv2 image visualization representation and (0,0) is in the top left corner
ROI = {
    'x_min': 170,
    'y_min': 0,
    'x_max': 540,
    'y_max': 320
}
# Target shape detection parameters
# Only circles with diameters 4 inch or more will be detected
MIN_TARGET_CIRCLE_RADIUS = 50
MAX_TARGET_CIRCLE_RADIUS = 180
# Current shape detection parameters
MIN_COLOR_INTENSITY = 70
MIN_CONTOUR_AREA = 1000
# Fraction of dough height reached at the roll start point
DOUGH_HEIGHT_CONTRACTION_RATIO = 0.5
# In meters
# Z_OFFSET = 0.61

RGB_IMG = None
POINT_CLOUD = None


def rgb_img_callback(ros_msg):
    global RGB_IMG
    RGB_IMG = cv2.imdecode(np.fromstring(ros_msg.data, np.uint8), cv2.IMREAD_COLOR)


def point_cloud_callback(ros_msg):
    # Generator
    global POINT_CLOUD
    POINT_CLOUD = point_cloud2.read_points(ros_msg, skip_nans=True, field_names=('x', 'y', 'z'))


def get_ROI_img(img):
    return img[ROI['y_min']:ROI['y_max'], ROI['x_min']:ROI['x_max']]


def capture_target_shape(debug_vision):
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

    if debug_vision:
        debug_img = RGB_IMG.copy()
        # Draw the region of interest
        cv2.rectangle(debug_img, (ROI['x_min'], ROI['y_min']), (ROI['x_max'], ROI['y_max']), color=(0, 255, 0), thickness=1)
        # Draw the target shape
        cv2.circle(debug_img, (x, y), r, color=(0, 0, 255), thickness=1)
        # Draw text
        cv2.rectangle(debug_img, (5, 5), (160, 70), color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(debug_img, 'ROI', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, 'Target shape', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow(WINDOW_TITLE_PREFIX + 'Target shape', debug_img)
        cv2.waitKey(0)

    return {
        'type': 'circle',
        'params': {
            'center': (x, y),
            'radius': r
        }
    }


def capture_current_shape(debug_vision):
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

    if debug_vision:
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
        cv2.imshow(WINDOW_TITLE_PREFIX + 'Current dough shape', debug_img)
        cv2.waitKey(0)
    
    return current_shape_contour


def calculate_centroid_2d(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        raise ValueError('Failed to calculate the centroid of the current dough shape!')
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def calculate_circle_line_intersection(cc, r, p1, p2):
    # Args: circle center, circle radius, 
    #       line point 1 (roll start point), line point 2 (candidate end point on dough boundary)
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


def image2robot_coords(pt):
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
    A, B = -2.5000000000000066e-05, 0.08520000000000003
    C, D = -0.02093333333333333, 0.04027500000000002
    return (A*pt[0] + B, C*pt[1] + D)


def image2pointcloud_coords(pt):
    A, B = 0.0009652963665596975, -0.32951992162237304
    C, D = 0.0009512554830209385, -0.22369287797508308
    return (A*pt[0] + B, C*pt[1] + D)


def get_dough_height(pt):
    # Transform to point cloud coordinates
    pc_pt = image2pointcloud_coords(pt)

    if type(POINT_CLOUD) != GeneratorType:
        raise ValueError(f'No valid point cloud data received from camera!')
    
    # Find k closest points in the point cloud (in terms of x and y) and get their average distance from camera
    # point_cloud_arr = np.fromiter(POINT_CLOUD, dtype=np.dtype(float, (3,)))
    point_cloud_arr = np.array(list(POINT_CLOUD))
    _, idx = KDTree(point_cloud_arr[:, :2]).query([pc_pt], k=3)
    depth = np.mean(point_cloud_arr[idx][0, :, 2])

    max_depth = max(point_cloud_arr[:, 2])
    return max_depth - depth


def calculate_roll_start_and_end(start_method, end_method, target_shape, pcl, debug_vision):
    # Calculate the roll start point S (in 2D)
    current_shape_contour = capture_current_shape(debug_vision=False)
    # pcl_clusters_detected, pcl_clusters = pcl.get_cluster_positions(ref_frame="wx250s/base_link", sort_axis="x", reverse=True)
    # if pcl_clusters_detected:
    #     S = pcl_clusters[0]['position']
    # else:
    #     S = (*calculate_centroid_2d(current_shape_contour), 0.01)
    if start_method == 'centroid-2d':
        S = calculate_centroid_2d(current_shape_contour)
        
    elif start_method == 'centroid-3d':
        #TODO: take point cloud data, filter out by ROI, and then take average
        assert False

    elif start_method == 'highest-point':
        #TODO: take point cloud data, filter out by ROI, and then sort by height
        assert False

    # Calculate the roll end point E (in 2D)
    E = None
    C = np.array(target_shape['params']['center'])
    R = target_shape['params']['radius']
    # Find the largest gap between the current and target dough shape
    max_gap = 0
    for P in current_shape_contour[:, 0]:
        intersection = calculate_circle_line_intersection(C, R, S, P)
        if intersection is not None:
            gap = np.linalg.norm(P - intersection)
            if gap > max_gap:
                max_gap = gap
                if end_method == 'current':
                    E = P
                elif end_method == 'target':
                    E = intersection

    if E is None:
        raise ValueError(f'No suitable roll end point E found!')

    if debug_vision:
        debug_img = RGB_IMG.copy()
        # Draw the region of interest
        cv2.rectangle(debug_img, (ROI['x_min'], ROI['y_min']), (ROI['x_max'], ROI['y_max']), color=(0, 255, 0), thickness=1)
        # Draw the target shape
        cv2.circle(debug_img, tuple(C), R, color=(0, 0, 255), thickness=1)
        # Draw the current shape
        # drawContours will fail if the contour is not closed (i.e. when a part of the dough is out of ROI)
        # cv2.drawContours(debug_img, [current_shape_contour], color=(255, 0, 0), thickness=1)
        cv2.polylines(debug_img, [current_shape_contour], isClosed=True, color=(255, 0, 0), thickness=1)
        # Draw the planned roll path
        cv2.arrowedLine(debug_img, tuple(S), tuple(E), color=(0, 0, 255), thickness=1)
        # Draw text
        cv2.rectangle(debug_img, (5, 5), (160, 100), color=(255, 255, 255), thickness=cv2.FILLED)
        cv2.putText(debug_img, 'ROI', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, 'Target shape', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(debug_img, 'Current shape', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.65, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow(WINDOW_TITLE_PREFIX + 'Roll trajectory', debug_img)
        cv2.waitKey(0)

    # Transform from image to robot coordinates
    S = image2robot_coords(S)
    E = image2robot_coords(E)

    # Calculate the z coordinates of the points S and E
    z = DOUGH_HEIGHT_CONTRACTION_RATIO * get_dough_height(S)
    # For now, the end point has the same z location as the start point
    S = (S[0], S[1], z)
    E = (E[0], E[1], z)

    return S, E


def calculate_iou(target_shape, debug_vision):
    # Calculate target shape mask
    target_shape_mask = np.zeros(IMG_SHAPE, dtype="uint8")
    if target_shape['type'] != 'circle':
        raise ValueError(f'Unknown target shape {target_shape["type"]}')
    cv2.circle(target_shape_mask, center=target_shape['params']['center'], radius=target_shape['params']['radius'], color=255, thickness=cv2.FILLED)

    # Calculate current shape mask
    current_shape_contour = capture_current_shape(debug_vision=False)
    current_shape_mask = np.zeros(IMG_SHAPE, dtype="uint8")
    # drawContours would fail if the contour is not closed (i.e. when a part of the dough is out of ROI)
    cv2.fillPoly(current_shape_mask, [current_shape_contour], color=255)

    # Calculate intersection over union
    intersection = cv2.bitwise_and(current_shape_mask, target_shape_mask)
    union = cv2.bitwise_or(current_shape_mask, target_shape_mask)
    iou = np.sum(intersection) / np.sum(union)

    if debug_vision:
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
        cv2.imshow(WINDOW_TITLE_PREFIX + 'Intersection over union', debug_img)
        cv2.waitKey(0)

    return iou


def main():
    #########################
    # Parse args
    #########################

    parser = argparse.ArgumentParser()
    parser.add_argument('-sm', '--start-method', type=str, default='centroid-2d', choices=['centroid-2d', 'centroid-3d', 'highest-point'],
                        help='Choose the roll start point calculation method from: "centroid-2d", "centroid-3d", and "highest-point"')
    parser.add_argument('-em', '--end-method', type=str, default='current', choices=['current', 'target'],
                        help='Choose the roll end point calculation method from: "current" (current shape outline), "target" (target shape outline)')
    parser.add_argument('-tc', '--termination-condition', type=str, default='time', choices=['time', 'iou'], help='Choose either "time" or "iou" termination condition.')
    parser.add_argument('-tv', '--termination-value', type=float, default=10., help='Either maximum time in seconds or minimum IoU based on the termination-condition argument.')
    parser.add_argument('-ld', '--log-dir', type=str, default='/home/RoboG1/interbotix_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_perception/scripts/robotic-dough-shaping/logs', help='Path to directory where to save logs.') 
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

    bot = InterbotixManipulatorXS("wx250s")#, moving_time=5, accel_time=5)

    def go_to_ready_pose():
        bot.arm.set_ee_pose_components(x=0.0567, y=0, z=0.15812, pitch=np.pi/2)

    #########################
    # Setup logging
    #########################

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_filename = f'log_{args.termination_condition}_{args.termination_value}_{args.start_method}_{args.end_method}_{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}'
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
        target_shape = capture_target_shape(debug_vision=False)
        params.update(target_shape)

        # Calculate current IoU
        iou = calculate_iou(target_shape, args.debug_vision)

        # Logging
        time_elapsed = time() - start_time
        print('=' * 120)
        print(f'Params:\n{json.dumps(params, indent=2)}')
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

            # Calculate the roll start point S and roll end point E
            S, E = calculate_roll_start_and_end(args.start_method, args.end_method, target_shape, pcl, args.debug_vision)
            print(f'Calculated roll start point S: {[f"{i:.3f}" for i in S]} m')
            print(f'Calculated roll end point E: {[f"{i:.3f}" for i in E]} m')

            # Calculate the angle of the direction S -> E
            # No need to use arctan2 due to symmetry
            yaw_SE = np.arctan((E[1] - S[1]) / (E[0] - S[0]))
            print(f'Calculated the angle of the direction S -> E: {yaw_SE * 180 / np.pi:.3f} deg')
            return

            if not args.disable_robot:

                # Move to AboveBeforePose
                bot.arm.set_ee_pose_components(x=S[0], y=S[1], z=S[2] + args.z_above, pitch=np.pi/2, yaw=yaw_SE)

                # Move to TouchPose
                bot.arm.set_ee_pose_components(x=S[0], y=S[1], z=S[2], pitch=np.pi/2, yaw=yaw_SE)

                # Perform roll to point E
                bot.arm.set_ee_pose_components(x=E[0], y=E[1], z=E[2], pitch=np.pi/2, yaw=yaw_SE)

                # Move to AboveAfterPose
                bot.arm.set_ee_pose_components(x=E[0], y=E[1], z=E[2] + args.z_above, pitch=np.pi/2, yaw=yaw_SE)

                # Move to ReadyPose
                go_to_ready_pose()

            # Calculate current IoU
            iou = calculate_iou(target_shape, args.debug_vision)

            # Logging
            time_elapsed = time() - start_time
            print(f'Iteration: {iteration:7d}\t Time: {time_elapsed:5.3f} s\t IoU: {iou:.3f}')
            print('=' * 120)
            csv_writer.writerow([iteration, time_elapsed, iou])

            iteration += 1

    cv2.destroyAllWindows() 


if __name__=='__main__':
    main()
