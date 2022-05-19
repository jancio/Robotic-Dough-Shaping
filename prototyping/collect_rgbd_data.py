import numpy as np
from scipy.ndimage import filters

import sys
import roslib
import rospy

import cv2
from PIL import Image

from sensor_msgs.msg import CompressedImage

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

class image_feature:
    def __init__(self):
        self.subscriber1 = rospy.Subscriber("/camera/color/image_raw/compressed",
            CompressedImage, self.rgb_callback,  queue_size = 1)
        self.subscriber2 = rospy.Subscriber("/pc_filter/pointcloud/filtered",
            PointCloud2, self.pc_callback,  queue_size = 1)
        self.once = True
        self.save_path = './data/janko/dough-colors/'
        self.num = 1

    def rgb_callback(self, ros_data):
        image_np = cv2.imdecode(np.fromstring(ros_data.data, np.uint8), cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        np.save(self.save_path + f'purple{self.num}.npy', image_np)
        image_copy = Image.fromarray(image_np)
        image_copy.save(self.save_path + f'purple{self.num}.png')
        print('saved!')

    def pc_callback(self, ros_data):
        # if self.once:
        data = point_cloud2.read_points_list(ros_data, skip_nans=True, field_names = ("x", "y", "z"))
        # x, y, z, ?
        p = np.array(data)
        # print(p.shape)
        # print(len(data))
        np.save(self.save_path + f'pc-purple{self.num}.npy', p)
        print("save succeed")
        # self.once = False


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
