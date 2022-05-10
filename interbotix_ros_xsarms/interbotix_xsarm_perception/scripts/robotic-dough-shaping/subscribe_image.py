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
import cv2
from PIL import Image

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE=False

class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
            CompressedImage, queue_size=10)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/camera/color/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE:
            print ("subscribed to /camera/color/image_raw/compressed")


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print ('received image of type: "%s"' % ros_data.format)

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        image_np_copy = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        np.save('./janko/d8.npy', image_np_copy)
        image_copy = Image.fromarray(image_np_copy)
        image_copy.save('./janko/d8.png')
        print('saved!')
        return

        # bounding box
        top_left = (250,40)
        bottom_right = (450,250)

        bb_img = image_np[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        img_gray = cv2.cvtColor(bb_img, cv2.COLOR_BGR2GRAY)
        # apply binary thresholding
        _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        # draw contours on the original image
        
        # cv2.drawContours(image=image_np_copy[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], contours=contours[1], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
        # image_np_copy = cv2.rectangle(image_np_copy, top_left, bottom_right, (0,0,255), 1)


        #### Feature detectors using CV2 #### 
        # "","Grid","Pyramid" + 
        # "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"
        # method = "GridFAST"
        # feat_det = cv2.SIFT_create()
        # time1 = time.time()

        # # convert np image to grayscale
        # featPoints = feat_det.detect(
        #     cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY))
        # time2 = time.time()
        # if VERBOSE :
        #     print ('%s detector found: %s points in: %s sec.'%(method,
        #         len(featPoints),time2-time1))

        # for featpoint in featPoints:
        #     x,y = featpoint.pt
        #     cv2.circle(image_np,(int(x),int(y)), 3, (0,0,255), -1)
        
        cv2.imshow('cv_img', image_np_copy)
        cv2.waitKey(2)

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np_copy)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)
        
        #self.subscriber.unregister()

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)