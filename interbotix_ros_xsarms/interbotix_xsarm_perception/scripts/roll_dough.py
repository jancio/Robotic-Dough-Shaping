######################################################################################################
# Robotic Dough Shaping - Main file
######################################################################################################
# Instructions:
#   First, run: roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=wx250s
#   Then change to this directory and run: python3 roll_dough.py
######################################################################################################

from time import time
import math.pi as PI
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface




def main():
    # Initialize the arm module along with the pointcloud module
    bot = InterbotixManipulatorXS("wx250s")#, moving_time=5, accel_time=5)
    pcl = InterbotixPointCloudInterface()

    #########################
    # Settings
    #########################

    method = 'basic'
    # Choose one termination condition
    termination_condition = 'max_time'
    termination_condition = 'min_iou'
    max_time = 10 # in seconds
    min_iou = 0.5

    if termination_condition == 'max_time':
        def terminate(t, iou):
            return t >= max_time
    elif termination_condition == 'min_iou':
        def terminate(t, iou):
            return iou >= min_iou
    else:
        raise ValueError(f'Unknown terminationcondition {termination_condition}')

    # Logging
    print('=' * 120)
    print(f'Method: {method}')
    print(f'Termination condition: {termination_condition}')
    if termination_condition == 'max_time':
        print(f'\t when time >= {max_time} s')
    elif termination_condition == 'min_iou':
        print(f'\t when  IoU >= {min_iou}')
    print('=' * 120)

    #########################
    # Preparation
    #########################
    start_time = time()
    iou = 0.

    # Move to ReadyPose
    bot.arm.set_ee_pose_components(x=0.0567, y=0, z=0.15812, pitch=PI/2)

    # Record the target dough shape
    #TODO

    # Calculate current IoU
    #TODO
    # iou = calculate_iou()

    # Logging
    print(f'Preparation time: {time() - start_time} s')
    print('=' * 120)

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

        # Move to AboveBeforePose
        bot.arm.set_ee_pose_components(x=Sx, y=Sy, z=Sz+d, pitch=PI/2, yaw=angle(S->G))

        # Move to TouchPose
        bot.arm.set_ee_pose_components(x=Sx, y=Sy, z=Sz, pitch=PI/2, yaw=angle(S->G))

        # Perform roll to point G
        bot.arm.set_ee_pose_components(x=Gx, y=Gy, z=Gz, pitch=PI/2, yaw=angle(S->G))

        # Move to AboveAfterPose
        bot.arm.set_ee_pose_components(x=Gx, y=Gy, z=Gz+d, pitch=PI/2, yaw=angle(S->G))

        # Move to ReadyPose
        bot.arm.set_ee_pose_components(x=0.0567, y=0, z=0.15812, pitch=PI/2)

        # Calculate current IoU
        #TODO
        # iou = calculate_iou()

        # Logging
        print(f'Iteration: {iteration:7d}\t Time: {time() - start_time:5.3f} s\t IoU: {iou:.3f}')
        print('=' * 120)

        iteration += 1


if __name__=='__main__':
    main()
