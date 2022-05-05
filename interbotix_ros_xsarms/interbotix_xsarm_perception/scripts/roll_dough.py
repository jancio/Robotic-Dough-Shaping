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
import argparse
import numpy as np
from time import time
from datetime import datetime
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface


def calculate_iou():
    #TODO
    return iou


def main():
    #########################
    # Parse args
    #########################

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, default='basic', help='Choose the method from: "basic", ...')
    parser.add_argument('-tc', '--termination-condition', type=str, default='time', help='Choose either "time" or "iou" termination condition.')
    parser.add_argument('-tv', '--termination-value', type=float, default=10., help='Either maximum time in seconds or minimum IoU based on the termination-condition argument.')
    parser.add_argument('-ld', '--log-dir', type=str, default='./logs', help='Path to directory where to save logs.') 
    parser.add_argument('-za', '--z-above', type=float, default=0.15812, help='Vertical distance above the dough immediately before and after rolling.')    
    args = parser.parse_args()

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

    #########################
    # Setup logging
    #########################

    print('=' * 120)
    print(json.dumps(vars(args), indent=2))
    print('=' * 120)

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_filename = f'log_{args.termination_condition}_{args.termination_value}_{args.method}_{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}'
    with open(f'{args.log_dir}/{log_filename}.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # First line is a dictionary of parameters
        csv_writer.writerow([json.dumps(vars(args))])
        # Second line is the header
        csv_writer.writerow(['Iteration', 'Time', 'IoU'])

        #########################
        # Preparation
        #########################
        iteration = 0
        start_time = time()
        iou = 0.

        # Move to ReadyPose
        bot.arm.set_ee_pose_components(x=0.0567, y=0, z=0.15812, pitch=np.pi/2)

        # Record the target dough shape
        #TODO

        # Calculate current IoU
        #TODO
        # iou = calculate_iou()

        # Logging
        time_elapsed = time() - start_time
        print(f'Preparation: \t Time: {time_elapsed:5.3f} s\t IoU: {iou:.3f}')
        print('=' * 120)
        csv_writer.writerow([iteration, time_elapsed, iou])

        #########################
        # Iterative procedure
        #########################
        iteration = 1
        while not terminate(time() - start_time, iou):

            # Detect the roll starting point S
            #TODO
            _, cluster = pcl.get_cluster_positions(ref_frame="wx250s/base_link", sort_axis="x", reverse=True)
            S = cluster['position']
            print(f'Detected roll start point S: x, y, z =')

            # Calculate the roll goal point G
            #TODO
            print(f'Calculated roll goal point G: x, y, z =')

            # Calculate the angle of the direction S -> G
            # No need to use arctan2 due to symmetry
            yaw_SG = np.arctan((Gy - Sy) / (Gx - Sx))
            print(f'Calculated the angle of the direction S -> G: {yaw_SG * 180 / np.pi}')

            # Move to AboveBeforePose
            bot.arm.set_ee_pose_components(x=Sx, y=Sy, z=Sz + args.z_above, pitch=np.pi/2, yaw=yaw_SG)

            # Move to TouchPose
            bot.arm.set_ee_pose_components(x=Sx, y=Sy, z=Sz, pitch=np.pi/2, yaw=yaw_SG)

            # Perform roll to point G
            bot.arm.set_ee_pose_components(x=Gx, y=Gy, z=Gz, pitch=np.pi/2, yaw=yaw_SG)

            # Move to AboveAfterPose
            bot.arm.set_ee_pose_components(x=Gx, y=Gy, z=Gz + args.z_above, pitch=np.pi/2, yaw=yaw_SG)

            # Move to ReadyPose
            bot.arm.set_ee_pose_components(x=0.0567, y=0, z=0.15812, pitch=np.pi/2)

            # Calculate current IoU
            #TODO
            # iou = calculate_iou()

            # Logging
            time_elapsed = time() - start_time
            print(f'Iteration: {iteration:7d}\t Time: {time_elapsed:5.3f} s\t IoU: {iou:.3f}')
            print('=' * 120)
            csv_writer.writerow([iteration, time_elapsed, iou])

            iteration += 1


if __name__=='__main__':
    main()
