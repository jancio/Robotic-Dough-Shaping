##################################################################################################################################
# Robotic Dough Shaping - Force sensing
#   Project for Robot Manipulation (CS 6751), Cornell University, Spring 2022
##################################################################################################################################
# Author: 
#   Janko Ondras (jo951030@gmail.com)
##################################################################################################################################
# Instructions:
#   First, run: roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=wx250s
#   Then change to this directory and run: python3 sense_force.py
##################################################################################################################################


import argparse
import numpy as np
from interbotix_xs_modules.arm import InterbotixManipulatorXS


def main():
    #########################
    # Parse args
    #########################

    parser = argparse.ArgumentParser()
    parser.add_argument('-za', '--z-above', type=float, default=0.15812, help='Vertical distance above the dough immediately before and after rolling.')
    parser.add_argument('-ni', '--num-iterations', type=int, default=5, help='Number of iterations to run.')
    args = parser.parse_args()

    #########################
    # Initialize robot
    #########################

    bot = InterbotixManipulatorXS("wx250s", moving_time=5, accel_time=5)

    # Calibrated minimum height
    z_min = 0.065
    x = 0.2
    y = 0.

    #########################
    # Move up and down
    #########################
    for i in range(1, args.num_iterations + 1):

        # Move above the dough
        bot.arm.set_ee_pose_components(x=x, y=y, z=z_min + args.z_above, pitch=np.pi/2)

        # Wait for user trigger
        input('Press ENTER to touch the dough! ...')

        # Touch the dough
        bot.arm.set_ee_pose_components(x=x, y=y, z=z_min, pitch=np.pi/2)

        print(f'Iteration: {i:7d}\t Touched the dough!')

    # Move back to start position
    bot.arm.set_ee_pose_components(x=x, y=y, z=z_min + args.z_above, pitch=np.pi/2)
    bot.arm.go_to_sleep_pose()


if __name__=='__main__':
    main()
