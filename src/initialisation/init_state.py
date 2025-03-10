import numpy as np
from gazebo_msgs.srv import SetModelState, SpawnModel, DeleteModel, GetWorldProperties
from gazebo_msgs.msg import ModelState
import rospy
import tf

set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

def reset_goal(goal_position, goal_sdf):
    model_name = "goal_marker"

    model_state_msg = ModelState()
    model_state_msg.model_name = model_name
    model_state_msg.pose.position.x = goal_position[0]
    model_state_msg.pose.position.y = goal_position[1]
    model_state_msg.pose.position.z = 0.0
    model_state_msg.pose.orientation.x = 0.0
    model_state_msg.pose.orientation.y = 0.0
    model_state_msg.pose.orientation.z = 0.0
    model_state_msg.pose.orientation.w = 1.0
    model_state_msg.reference_frame = 'world'

    model_names = get_world_properties().model_names
    delete = False

    for model in model_names:
        if model == model_name:
            delete = True

    if delete:
        delete_model(model_name)

    rospy.sleep(3)

    spawn_model(model_state_msg.model_name, goal_sdf, "", model_state_msg.pose, "world")
    print(f"Goal set. {goal_position}")

def reset_turtlebot3_gazebo(spawn_position, amr_model, yaw=0.0):
    quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)

    model_state_msg = ModelState()
    model_state_msg.model_name = amr_model
    model_state_msg.pose.position.x = spawn_position[0]
    model_state_msg.pose.position.y = spawn_position[1]
    model_state_msg.pose.position.z = 0.0
    model_state_msg.pose.orientation.x = 0.0
    model_state_msg.pose.orientation.y = 0.0
    model_state_msg.pose.orientation.z = quaternion[2]
    model_state_msg.pose.orientation.w = quaternion[3]
    model_state_msg.twist.linear.x = 0.0
    model_state_msg.twist.linear.y = 0.0
    model_state_msg.twist.linear.z = 0.0
    model_state_msg.twist.angular.x = 0.0
    model_state_msg.twist.angular.y = 0.0
    model_state_msg.twist.angular.z = 0.0
    model_state_msg.reference_frame = 'world'

    set_model_state(model_state_msg)
    print(f"Turtlebot set. {spawn_position}")