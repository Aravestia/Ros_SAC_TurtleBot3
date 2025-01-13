#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import SetModelState, GetWorldProperties, SpawnModel
from gazebo_msgs.msg import ModelState

class SpinObstacle4():
    def __init__(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')

        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        #model_names = self.get_world_properties().model_names

        self.radius = 0.12

        self.base_x = 0.5
        self.base_y = 0.5

        self.x1 = self.base_x
        self.y1 = self.base_y

        self.x2 = -self.base_x
        self.y2 = -self.base_y

        self.spawn_obstacle(self.x1, self.y1, "obstacle_1", self.radius)
        self.spawn_obstacle(self.x2, self.y2, "obstacle_2", self.radius)

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

            speed = 0.001

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
    
    def spawn_obstacle(self, x, y, name, radius):
        model_state_msg = ModelState()
        model_state_msg.model_name = name
        model_state_msg.pose.position.x = x
        model_state_msg.pose.position.y = y
        model_state_msg.pose.position.z = 0.0
        model_state_msg.pose.orientation.x = 0.0
        model_state_msg.pose.orientation.y = 0.0
        model_state_msg.pose.orientation.z = 0.0
        model_state_msg.pose.orientation.w = 1.0
        model_state_msg.reference_frame = 'world'

        self.spawn_model(model_state_msg.model_name, self.moving_obstacle_sdf(radius), "", model_state_msg.pose, "world")

    def moving_obstacle_sdf(self, radius):
        return f'''
            <?xml version="1.0" ?>
            <sdf version="1.5">
            <model name=goal_state_msg.model_name>
                <static>true</static>  <!-- Static means the object will not move -->

                <link name="link">
                    <pose>0 0 0 0 0 0</pose>  <!-- Position and orientation -->

                    <collision name="collision">

                        <geometry>
                            <cylinder>
                                <radius>{radius}</radius>
                                <length>0.5</length>
                            </cylinder>
                        </geometry>

                    </collision>

                    <visual name="visual">

                        <geometry>
                            <cylinder>
                                <radius>{radius}</radius>
                                <length>0.5</length>
                            </cylinder>
                        </geometry>

                        <material>
                            <ambient>1 0 0 1</ambient>
                            <diffuse>1 0 0 1</diffuse>
                        </material>

                    </visual>
                </link>
            </model>
            </sdf>
        '''

if __name__ == '__main__':
    rospy.init_node('spin_obstacle_4', anonymous=True)
    s = SpinObstacle4()
    #rospy.spin()