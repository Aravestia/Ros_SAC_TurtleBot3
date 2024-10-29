#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import SetModelState, GetWorldProperties
from gazebo_msgs.msg import ModelState

import tf
import math

class SpinObstacle1():
    def __init__(self, name, angular_velocity=1.0):
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_world_properties')

        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        model_names = self.get_world_properties().model_names

        self.euler = tf.transformations.euler_from_quaternion([0.0, 0.0, 0.0, 1.0])
        self.yaw = self.euler[2]

        while not rospy.is_shutdown():
            quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, self.yaw)

            msg = ModelState()
            msg.model_name = name
            msg.pose.orientation.z = quaternion[2]
            msg.pose.orientation.w = quaternion[3]
            msg.reference_frame = 'world'
            
            if name in model_names:
                self.set_model_state(msg)

            speed = 0.05

            self.yaw += ((math.pi / 360) * speed)

            if self.yaw >= (2 * math.pi):
                self.yaw = 0.0

            print(f"yaw: {self.yaw}")

if __name__ == '__main__':
    rospy.init_node('spin_obstacle_1', anonymous=True)
    s = SpinObstacle1(name="obstacle")
    #rospy.spin()