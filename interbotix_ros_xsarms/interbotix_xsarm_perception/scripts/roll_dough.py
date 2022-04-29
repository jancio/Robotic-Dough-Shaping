import time
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_perception_modules.pointcloud import InterbotixPointCloudInterface

# First, run: roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=wx250s
# Then change to this directory and type: python3 roll_dough.py


def main():
    # Initialize the arm module along with the pointcloud module
    bot = InterbotixManipulatorXS("wx250s")#, moving_time=5, accel_time=5)
    pcl = InterbotixPointCloudInterface()

    # Move to ReadyPose
    bot.arm.set_ee_pose_components(x=?, y=0, z=?, pitch=1.57)
    # Record the target dough shape
    todo

    # Iterative procedure
    while not terminate:

        # Detect the roll starting point S
        _, cluster = pcl.get_cluster_positions(ref_frame="wx250s/base_link", sort_axis="x", reverse=True)
        S = cluster['position']

        # Calculate the roll goal point G
        todo

        # Move to AboveBeforePose
        bot.arm.set_ee_pose_components(x=Sx, y=Sy, z=Sz+d, pitch=1.57, yaw=angle(S->G))

        # Move to TouchPose
        bot.arm.set_ee_pose_components(x=Sx, y=Sy, z=Sz, pitch=1.57, yaw=angle(S->G))

        # Perform roll to point G
        bot.arm.set_ee_pose_components(x=Gx, y=Gy, z=Gz, pitch=1.57, yaw=angle(S->G))

        # Move to AboveAfterPose
        bot.arm.set_ee_pose_components(x=Gx, y=Gy, z=Gz+d, pitch=1.57, yaw=angle(S->G))

        # Move to ReadyPose
        bot.arm.set_ee_pose_components(x=?, y=0, z=?, pitch=1.57)






    # set initial arm and gripper pose
    # bot.arm.set_ee_pose_components(x=0.3, z=0.2)
    # bot.gripper.open()

    # get the ArmTag pose
    # bot.arm.set_ee_pose_components(y=-0.3, z=0.2)
    # time.sleep(0.5)
    # armtag.find_ref_to_arm_base_transform()
    bot.arm.go_to_sleep_pose()
    # bot.arm.go_to_home_pose()
    # bot.arm.set_ee_cartesian_trajectory(x=-0.4, z=-0.2, pitch=1.57)
    bot.arm.set_ee_pose_components(x=0.0567, z=0.15812, pitch=1.57)
    # bot.arm.set_ee_pose_components(pitch=1.57)
    # bot.arm.go_to_sleep_pose()

    # get the cluster positions
    # sort them from max to min 'x' position w.r.t. the 'wx200/base_link' frame
    success, clusters = pcl.get_cluster_positions(ref_frame="wx250s/base_link", sort_axis="x", reverse=True)
    print("-------number of clusters:  ", len(clusters), clusters)

    # pick up all the objects and drop them in a virtual basket in front of the robot
    # for cluster in clusters:
    #     x, y, z = cluster["position"]
    #     print(x,y,z)
    #     # for i in range(10):
    #     # bot.arm.set_ee_pose_components(x=x-0.03, y=y, z=z+0.1, pitch=1.57)
    #     # bot.arm.set_ee_pose_components(x=x-0.03, y=y, z=z+0.05, pitch=1.57)
    #     bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.1, pitch=1.57)
    #     bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.05, pitch=1.57)
    #     # bot.gripper.close()
    #     bot.arm.set_ee_pose_components(x=x+0.02, y=y, z=z+0.05, pitch=1.57)

    #     # --------------------

    #     bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.1, pitch=1.57, yaw=1.57)
    #     bot.arm.set_ee_pose_components(x=x, y=y-0.02, z=z+0.1, pitch=1.57, yaw=1.57)
    #     # bot.arm.set_ee_pose_components(x=0.3, z=0.2, pitch=1.57)
    #     # bot.gripper.open()
    # bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()
