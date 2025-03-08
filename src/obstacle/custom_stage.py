#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import SetModelState, GetWorldProperties, SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState

class CustomStage():
    def __init__(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')

        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        self.delete_model("ros_symbol")

        self.size = 3
        self.thickness = 0.1
        self.height = 1

        self.spawn_obstacle(self.size, 0, "wall_1", self.thickness, self.size * 2, self.height)
        self.spawn_obstacle(-self.size, 0, "wall_2", self.thickness, self.size * 2, self.height)
        self.spawn_obstacle(0, self.size, "wall_3", self.size * 2, self.thickness, self.height)
        self.spawn_obstacle(0, -self.size, "wall_4", self.size * 2, self.thickness, self.height)

        self.spawn_obstacle(1, 0.5, "wall_5", self.thickness, 3, self.height)
        self.spawn_obstacle(-0.5, -1, "wall_6", 3, self.thickness, self.height)
    
    def spawn_obstacle(self, x, y, name, length, breadth, height):
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

        self.spawn_model(model_state_msg.model_name, self.wall_sdf(length, breadth, height), "", model_state_msg.pose, "world")

    def wall_sdf(self, length, breadth, height):
        return f'''
            <?xml version="1.0" ?>
            <sdf version="1.5">
            <model name="wall">
                <static>true</static>  <!-- Static means the object will not move -->

                <link name="link">
                <pose>0 0 0 0 0 0</pose>  <!-- Position and orientation -->

                <collision name="collision">
                    <geometry>
                    <box>
                        <size>{length} {breadth} {height}</size>
                    </box>
                    </geometry>
                </collision>

                <visual name="visual">
                    <geometry>
                    <box>
                        <size>{length} {breadth} {height}</size>
                    </box>
                    </geometry>
                    <material>
                    <ambient>1 1 1 1</ambient>
                    <diffuse>1 1 1 1</diffuse>
                    </material>
                </visual>
                </link>
            </model>
            </sdf>
        '''

if __name__ == '__main__':
    rospy.init_node('custom_stage', anonymous=True)
    s = CustomStage()
    #rospy.spin()