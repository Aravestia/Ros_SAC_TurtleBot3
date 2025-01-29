#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import SetModelState, GetWorldProperties, SpawnModel
from gazebo_msgs.msg import ModelState

class SpinObstacle3():
    def __init__(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')

        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        self.radius = 0.10

        self.base_x = 0.5
        self.base_y = 0.5

        #self.spawn_obstacle(self.base_x, self.base_y, "obstacle_1", self.radius)
        #self.spawn_obstacle(self.base_x, -self.base_y, "obstacle_2", self.radius)
        self.spawn_obstacle(-self.base_x, self.base_y, "obstacle_3", self.radius)
        #self.spawn_obstacle(-self.base_x, -self.base_y, "obstacle_4", self.radius)
    
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
    rospy.init_node('spin_obstacle_3', anonymous=True)
    s = SpinObstacle3()
    #rospy.spin()