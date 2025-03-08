#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import SetModelState, GetWorldProperties
from gazebo_msgs.msg import ModelState

import tf
import math

class Obstacle():
    def __init__(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_world_properties')

        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        #model_names = self.get_world_properties().model_names

        self.base_x = 0.7
        self.base_y = 1.0

        self.x1 = self.base_x
        self.y1 = self.base_y

        self.x2 = -self.base_x
        self.y2 = -self.base_y

        while not rospy.is_shutdown():
            msg = ModelState()
            msg.model_name = "obstacle_1"
            msg.pose.position.x = self.x1
            msg.pose.position.y = self.y1
            msg.reference_frame = 'world'

            msg_2 = ModelState()
            msg_2.model_name = "obstacle_2"
            msg_2.pose.position.x = self.x2
            msg_2.pose.position.y = self.y2
            msg_2.reference_frame = 'world'
            
            self.set_model_state(msg)
            self.set_model_state(msg_2)    

            speed = 0.0017

            self.x1, self.y1 = self.set_pos(self.x1, self.y1, speed)
            self.x2, self.y2 = self.set_pos(self.x2, self.y2, speed)

    def set_pos(self, x, y, speed):
        if x >= self.base_x:
            x = self.base_x

            if y > -self.base_y:
                y -= speed
            else:
                y = -self.base_y

        if y <= -self.base_y:
            y = -self.base_y

            if x > -self.base_x:
                x -= speed
            else:
                x = -self.base_x

        if x <= -self.base_x:
            x = -self.base_x

            if y < self.base_y:
                y += speed
            else:
                y = self.base_y

        if y >= self.base_y:
            y = self.base_y

            if x < self.base_x:
                x += speed
            else:
                x = self.base_x

        return x, y

if __name__ == '__main__':
    rospy.init_node('obstacle_2', anonymous=True)
    s = Obstacle()
    #rospy.spin()