#!/usr/bin/env python
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.
"""
__author__ =  'Simon Haller <simon.haller at uibk.ac.at>'
__version__=  '0.1'
__license__ = 'BSD'
# Python libs
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
# import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import PointCloud2
# import sensor_msgs.msg.point_cloud as pc2
from sensor_msgs import point_cloud2
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE=False

class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        # self.image_pub = rospy.Publisher("/output/image_raw/compressed",
        #     CompressedImage, queue_size=10)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/pc_filter/pointcloud/filtered",
            PointCloud2, self.callback,  queue_size = 1)
        if VERBOSE:
            print ("subscribed to /pc_filter/pointcloud/filtered")
        self.once = True

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print ('received image of type: "%s"' % ros_data.format)

        if self.once:
            
            data = point_cloud2.read_points_list(ros_data, skip_nans=True, field_names = ("x", "y", "z"))
            # x, y, z, ?
            p = np.array(data)
            print(p.shape)
            # print(len(data))
            np.save("./target.npy", p)
            print("save succeed")
            self.once = False


def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
