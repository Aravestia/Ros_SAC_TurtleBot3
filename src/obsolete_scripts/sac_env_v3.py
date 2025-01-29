#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState, SpawnModel, DeleteModel, GetWorldProperties
from gazebo_msgs.msg import ModelState

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

import math
import tf
from datetime import datetime
import os
import numpy as np
import pandas as pd
import random
import time
import copy
import csv

from sdf_files import goal_sdf, waypoint_sdf
from a_star import a_star_search, img_to_grid, _get_closest_waypoint, point_to_grid

class SacEnvV3(gym.Env):
    def __init__(self, amr_model='turtlebot3_burger', epoch=0, init_positions=[], stage_map="", yaw=0.0, max_timesteps=10000, test_mode=False):
        super(SacEnvV3, self).__init__()

        self.velocity_multiplier = 0.15
        self.angular_velocity_multiplier = 2.84
        self.velocity = -1.0
        self.velocity_previous = -1.0
        self.angular_velocity = 0.0
        self.angular_velocity_previous = 0.0

        self.laserscan_maxcap = 3.5
        self.laserscan_mincap = 0.12
        self.laserscan_warning_threshold = 0.25
        self.laserscan_warning_threshold_normalised = self.normalise_value(self.laserscan_warning_threshold, self.laserscan_maxcap, self.laserscan_mincap)
        self.laserscan_closest = 1.0
        self.laserscan_closest_angle = 0.0
        self.laserscan = np.full(8, 1.0)

        self.angle_cap = 2 * math.pi

        self.init_positions = init_positions
        self.spawn_position = self.init_positions[0]
        self.position = self.spawn_position
        self.goal_position = self.init_positions[1]
            
        self.goal_distance_from_spawn_vector = self.goal_position - self.spawn_position
        self.goal_distance_from_spawn = np.linalg.norm(self.goal_distance_from_spawn_vector)
        self.goal_distance_previous = self.goal_distance_from_spawn
        self.goal_distance_record = 1.0
        self.goal_angle_from_spawn = math.atan2(
            self.goal_position[1] - self.spawn_position[1], 
            self.goal_position[0] - self.spawn_position[0]
        )

        self.spawn_orientation = np.array([0.0, 1.0]) # [z, w]
        self.yaw = yaw

        self.done = False
        self.truncated = False
        self.total_timesteps = 0
        self.max_timesteps = max_timesteps
        self.step_count = 0
        self.max_step_count = 2000
        self.stagnant_count = 0
        self.max_stagnant_count = 10
        self.reset_count = 0
        self.observation_state = []

        self.amr_model = amr_model
        self.epoch = epoch

        self.goal_file_path = r"/home/aravestia/isim/noetic/src/robot_planner/src/score.csv"
        self.goal_df = pd.read_csv(self.goal_file_path, header=0, index_col=0)
        self.goal_count = len(self.goal_df)
        self.goal_radius = 0.2
        self.completion_count = 0

        self.grid_row = 384
        self.grid_col = 384
        self.grid_margin = 5
        self.grid = img_to_grid(stage_map, self.grid_margin)
        self.grid_in = copy.deepcopy(self.grid)
        self.grid_x_offset = 200
        self.grid_y_offset = 184
        self.grid_resolution = 0.05
        self.grid_spawn = point_to_grid(self.grid_x_offset, self.grid_y_offset, self.spawn_position[0], self.spawn_position[1])
        self.grid_goal = point_to_grid(self.grid_x_offset, self.grid_y_offset, self.goal_position[0], self.goal_position[1])

        self.waypoint_occurrence = 1
        self.waypoints = a_star_search(
            self.grid_in,
            self.grid_spawn, 
            self.grid_goal, 
            self.grid_row, 
            self.grid_col, 
            self.grid_resolution, 
            self.grid_x_offset, 
            self.grid_y_offset,
            self.waypoint_occurrence
        )
        self.waypoint_closest = 0
        self.waypoint_closest_best = 0
        self.waypoint_min_distance = 0.5
        self.waypoint_lookahead = 15
        self.waypoint_closest_angle = 3

        self.follower_mode = False
        self.test_mode = test_mode

        self.moving_obstacle_radius = 0.15
        self.explored_grid_size = 5
        self.explored_areas = np.array([np.floor(self.explored_grid_size * self.spawn_position)])
        self.exploration_max = 40
        self.explored_grid_cap = self.exploration_max
        self.unexplored = 1.0

        print(f"{self.goal_df}. {self.goal_count}")

        self.goal_sdf = goal_sdf(self.goal_radius)

        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/gazebo/delete_model')

        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        ### Set Specific properties for world ###
        self.model_names = self.get_world_properties().model_names
        for model in self.model_names:
            if model.startswith("goal_marker_"):
                self.delete_model(model)
        #########################################

        self.laserscan_subscriber = rospy.Subscriber('/scan', LaserScan, self.laserscan_callback)
        self.odometry_subscriber = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
        self.imu_subscriber = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.twist_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Action space: [velocity, angular velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.observation_state_titles = [
            'unexplored_area',
            'velocity_normalised',
            'angular_velocity_normalised', 
            'laserscan_closest',
            'laserscan_closest_angle', 
            'laserscan_NW', 
            'laserscan_W',
            'laserscan_SW', 
            'laserscan_S', 
            'laserscan_SE', 
            'laserscan_E', 
            'laserscan_NE', 
            'laserscan_N',
        ]
        self.observation_state_titles_len = len(self.observation_state_titles)
        self.observation_space = spaces.Box(
            low=np.full(self.observation_state_titles_len, -1.0),
            high=np.full(self.observation_state_titles_len, 1.0),
            shape=(self.observation_state_titles_len,), 
            dtype=np.float32
        )

    def laserscan_callback(self, scan):
        laserscan_360 = self.normalise_value(
            np.clip(np.array(scan.ranges), self.laserscan_mincap, self.laserscan_maxcap), 
            self.laserscan_maxcap,
            self.laserscan_mincap
        )
        laserscan = np.array([])

        for i in range(8):
            laserscan = np.append(laserscan, np.min(laserscan_360[i * 45:(i + 1) * 45]))

        self.laserscan = laserscan
        self.laserscan_closest = np.min(laserscan)
        self.laserscan_closest_angle = self.degree_to_radians(np.argmin(laserscan_360))

    def odometry_callback(self, odom):
        position = odom.pose.pose.position
        self.position = np.array([position.x, position.y])

    def imu_callback(self, imu):
        orientation = imu.orientation
        euler = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        self.yaw = euler[2]

    def publish_velocity(self, velocity, angular_velocity):
        twist = Twist()
        twist.linear.x = velocity
        twist.angular.z = angular_velocity
        self.twist_publisher.publish(twist)

    def _get_observation_state(self):
        current_position = self.position

        goal_position = self.goal_position
        goal_distance_vector = goal_position - current_position
        goal_distance = self.normalise_value(np.linalg.norm(goal_distance_vector), self.goal_distance_from_spawn)
        goal_distance_normalised = (2 / (1 + np.exp(-3 * goal_distance))) - 1

        self.waypoint_closest, self.waypoint_min_distance = _get_closest_waypoint(
            current_position, 
            self.waypoints, 
            self.waypoint_closest_best, 
            self.waypoint_lookahead, 
            len(self.waypoints)
        )
        angular_velocity = self.angular_velocity
        velocity = self.velocity
        laserscan_closest = self.laserscan_closest
        laserscan_closest_angle = self.laserscan_closest_angle
        laserscan = self.laserscan

        self.waypoint_closest_angle = self.normalise_radians_angle(
            math.atan2(
                self.waypoints[self.waypoint_closest][1] - current_position[1], 
                self.waypoints[self.waypoint_closest][0] - current_position[0]
            ) - self.yaw
        )

        discrete_position = np.floor(self.explored_grid_size * current_position)

        self.unexplored = 1.0
        for x in self.explored_areas[-min(self.explored_grid_cap, len(self.explored_areas)):]:
            if np.array_equal(x, discrete_position):
                self.unexplored = -1.0
                break

        if laserscan_closest <= self.laserscan_warning_threshold_normalised:
            self.unexplored = -1.0

        if self.unexplored == 1.0:
            self.explored_areas = np.append(self.explored_areas, [discrete_position], axis=0)

        return np.nan_to_num(
            np.append(
                np.array([
                    self.unexplored,
                    velocity,
                    angular_velocity,
                    laserscan_closest,
                    laserscan_closest_angle,
                ]),
                laserscan
            )
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.reset_count += 1

        self.velocity = -1.0
        self.velocity_previous = -1.0
        self.angular_velocity = 0.0
        self.angular_velocity_previous = 0.0

        self.publish_velocity(0.0, 0.0)

        self.spawn_position = self.init_positions[0]
        self.position = self.spawn_position
        self.goal_position = self.init_positions[1]
            
        self.goal_distance_from_spawn_vector = self.goal_position - self.spawn_position
        self.goal_distance_from_spawn = np.linalg.norm(self.goal_distance_from_spawn_vector)
        self.goal_distance_previous = self.goal_distance_from_spawn
        self.goal_distance_record = 1.0
        self.goal_angle_from_spawn = math.atan2(
            self.goal_position[1] - self.spawn_position[1], 
            self.goal_position[0] - self.spawn_position[0]
        )

        self.grid_in = copy.deepcopy(self.grid)
        self.grid_spawn = point_to_grid(self.grid_x_offset, self.grid_y_offset, self.spawn_position[0], self.spawn_position[1])
        self.grid_goal = point_to_grid(self.grid_x_offset, self.grid_y_offset, self.goal_position[0], self.goal_position[1])

        self.waypoints = a_star_search(
            self.grid_in,
            self.grid_spawn, 
            self.grid_goal, 
            self.grid_row, 
            self.grid_col, 
            self.grid_resolution, 
            self.grid_x_offset, 
            self.grid_y_offset,
            self.waypoint_occurrence
        )
        self.waypoint_closest = 0
        self.waypoint_closest_best = 0

        self.explored_areas = np.array([np.floor(self.explored_grid_size * self.spawn_position)])
        self.unexplored = 1.0

        self.reset_turtlebot3_gazebo()
        self.reset_goal()

        self.done = False
        self.truncated = False

        self.step_count = 0
        self.stagnant_count = 0

        self.laserscan_closest = 1.0

        self.observation_state = self._get_observation_state()
        rospy.sleep(0.2)

        print(self.observation_state)

        return self.observation_state, {}

    def step(self, action):
        self.total_timesteps += 1
        self.step_count += 1

        self.done = False
        self.truncated = False
        
        self.observation_state = self._get_observation_state()

        self.follower_mode = (
            self.observation_state[len(self.observation_state) - 1] > self.laserscan_warning_threshold_normalised
        ) and (
            self.observation_state[len(self.observation_state) - 8] > self.laserscan_warning_threshold_normalised
        ) and (self.test_mode)

        if not self.follower_mode:
            self.velocity = float(action[0])
            self.angular_velocity = float(action[1])
        else:
            min_velocity = 0.0
            turning_rate = 0.5 / self.angular_velocity_multiplier
            angle_threshold = 0.15
            distance_threshold = 0.15

            if self.waypoint_min_distance < distance_threshold:
                self.waypoint_closest_best += 1

            if self.waypoint_closest_angle < -angle_threshold:
                self.angular_velocity = -turning_rate
                self.velocity = min_velocity
            elif self.waypoint_closest_angle > angle_threshold:
                self.angular_velocity = turning_rate
                self.velocity = min_velocity
            else:
                self.angular_velocity = 0.0
                self.velocity = 1.0

        self.publish_velocity(
            velocity=(self.velocity + 1.0) * 0.5 * self.velocity_multiplier,
            angular_velocity=self.angular_velocity * self.angular_velocity_multiplier,
        )

        rospy.sleep(0.02)
        self.observation_state = self._get_observation_state()
        reward = self._compute_reward()

        print("------------------------------------------")
        print("OBSERVATION SPACE")
        for i in range(self.observation_state_titles_len):
            print(f"{self.observation_state_titles[i]}: {self.observation_state[i]}")
        print(f"waypoint_closest_angle: {self.waypoint_closest_angle}")
        print(" ")
        print("ACTION SPACE")
        print(f"velocity : {self.velocity}")
        print(f"angular_velocity : {self.angular_velocity}")
        print(" ")
        print(f"epoch: {self.epoch}")
        print(f"total_timesteps: {self.total_timesteps}")
        print(f"step_count: {self.step_count}/{self.max_step_count}")
        print(f"reward: {reward}")
        print(f"current position: {np.round(self.position, 2)}")
        print(f"closest waypoint: {self.waypoint_closest}")
        print(f"total waypoints: {len(self.waypoints)}")
        print(f"completion count: {self.completion_count}")
        print(f"areas explored: {len(self.explored_areas)}/{self.exploration_max}")
        print(f"follower mode: {self.follower_mode}")
        print("------------------------------------------")
        print(" ")

        return self.observation_state, reward, self.done, self.truncated, {}

    def _compute_reward(self):
        unexplored = self.observation_state[0]
        velocity = self.observation_state[1]
        angular_velocity = self.observation_state[2]
        laserscan_closest = self.observation_state[3]
        laserscan_closest_angle = self.observation_state[4]

        laserscan_quadrant_index = 5
        laserscan_quadrants = [self.observation_state[i + laserscan_quadrant_index] for i in range(8)]

        step_count = self.step_count
        collision_threshold = self.normalise_value(self.laserscan_mincap + 0.01, self.laserscan_maxcap, self.laserscan_mincap)
        warning_threshold = self.laserscan_warning_threshold_normalised

        #penalty_goal_distance = -(goal_distance + 1) * 0.5
        penalty_obstacle_proximity = 0 if laserscan_closest > warning_threshold else -2.0
        penalty_facing_obstacle = 0

        if laserscan_quadrants[0] == laserscan_closest or laserscan_quadrants[7] == laserscan_closest:
            penalty_facing_obstacle += -5.0

        if laserscan_quadrants[1] == laserscan_closest or laserscan_quadrants[6] == laserscan_closest:
            penalty_facing_obstacle += -0.5

        penalty_collision = -50
        penalty_step_count = -(step_count / self.max_step_count)
        penalty_step_count_maxed = -50
        penalty_velocity = 0

        if laserscan_closest > warning_threshold:
            penalty_velocity = (velocity - 1) * 0.5
        else:
            velocity_threshold = -0.7

            if velocity > velocity_threshold:
                penalty_velocity = -5 * (velocity - velocity_threshold) / (1 - velocity_threshold)
            else:
                penalty_velocity = 2.5 * (velocity - velocity_threshold) / (1 - velocity_threshold)

        penalty_rapid_acceleration = -abs(self.velocity_previous - velocity) * 0.25
        penalty_rapid_turning = -abs(self.angular_velocity_previous - angular_velocity) * 0.25
        penalty_high_turning = -abs(angular_velocity) * 0.5

        self.velocity_previous = velocity
        self.angular_velocity_previous = angular_velocity

        reward_goal = 50
        reward_exploration_max = 50
        reward_exploration = (unexplored + 1) * 2.0
        reward = 0.0

        if self.waypoint_closest > self.waypoint_closest_best:
            self.waypoint_closest_best = self.waypoint_closest

        if self.step_count >= self.max_step_count: # Maxed Step Count
           reward += penalty_step_count_maxed
           self.end_episode()
        else:
            if laserscan_closest <= collision_threshold: # Collision
                reward += penalty_collision
                self.end_episode()
                print(f"!!!!!ROBOT COLLISION!!!!! scan: {laserscan_closest}")
            else:
                reward += reward_exploration
                reward += penalty_velocity 
                reward += penalty_rapid_acceleration 
                reward += penalty_rapid_turning
                #reward += penalty_high_turning
                reward += penalty_step_count

                if laserscan_closest < warning_threshold: # Too close to wall
                    reward += penalty_obstacle_proximity
                    reward += penalty_facing_obstacle
                else:   
                    if len(self.explored_areas) >= self.exploration_max: # Explored Enough
                        reward += reward_exploration_max
                        self.completion_count += 1

                        self.end_episode()
                        print(f"!!!!!ROBOT GOAL REACHED!!!!!")
        
        if self.total_timesteps > self.max_timesteps - 5:
            self.goal_df.to_csv(self.goal_file_path)

        return float(reward)
    
    def end_episode(self):
        self.goal_count += 1

        self.goal_df.loc[self.goal_count] = {
            'id': self.goal_count, 
            'score': len(self.explored_areas) / self.exploration_max, 
            'time': datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        }

        self.goal_distance_record = 1.0
        self.done = True
        
    def reset_goal(self):
        model_name = "goal_marker_"

        model_state_msg = ModelState()
        model_state_msg.model_name = model_name + str(self.reset_count)
        model_state_msg.pose.position.x = self.goal_position[0]
        model_state_msg.pose.position.y = self.goal_position[1]
        model_state_msg.pose.position.z = 0.0
        model_state_msg.pose.orientation.x = 0.0
        model_state_msg.pose.orientation.y = 0.0
        model_state_msg.pose.orientation.z = 0.0
        model_state_msg.pose.orientation.w = 1.0
        model_state_msg.reference_frame = 'world'

        model_names = self.get_world_properties().model_names

        for model in model_names:
            if model.startswith(model_name):
                self.delete_model(model)

        self.spawn_model(model_state_msg.model_name, self.goal_sdf, "", model_state_msg.pose, "world")
        print(f"Goal set. {self.goal_position}")

    def reset_turtlebot3_gazebo(self):
        yaw = 0.0
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)

        model_state_msg = ModelState()
        model_state_msg.model_name = self.amr_model
        model_state_msg.pose.position.x = self.spawn_position[0]
        model_state_msg.pose.position.y = self.spawn_position[1]
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

        self.set_model_state(model_state_msg)
        print(f"Turtlebot set. {self.spawn_position}")
    
    def degree_to_radians(self, angle):
        angle = angle * (math.pi / 180)
        return self.normalise_radians_angle(angle)
    
    def normalise_radians_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi

        return angle / math.pi
    
    def normalise_value(self, value, range_max, range_min=0):
        return (2 * ((value - range_min) / (range_max - range_min))) - 1

#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================

def init_stage_positions(stage):
    init_positions = [] # [spawn, goal]

    if stage == 1:
        init_positions = [[1, 1], [-1.25, -1.25]]

    if stage == 2:
        init_positions = [[1.25, 1.25], [0, 0]]   

    if stage == 3:
        init_positions = [[1, 1], [-1.25, -1.25]]

    if stage == 4: # Room
        init_positions = [[-1.5, 2], [2, -2]]
        
    if stage == 5: # Turtlebot_world
        init_positions = [[-2, -0.75], [2, 0.75]]

    return np.array(init_positions)

def init_map(stage):
    map = ""

    if stage == 2:
        map = r"/home/aravestia/isim/noetic/src/robot_planner/src/map/map_stage2.pgm"

    if stage == 5: # Turtlebot_world
        map = r"/home/aravestia/isim/noetic/src/robot_planner/src/map/map_turtlebot_world.pgm"

    return map

def main(args=None):
    epochs = 1000
    timesteps = 10000

    for i in range(epochs):
        rospy.init_node('sac_env_v3', anonymous=True)

        # Depends on map
        amr_model = 'turtlebot3_burger'
        model_pth = r"/home/aravestia/isim/noetic/src/robot_planner/src/models/sac_model_v3.0.pth"
        stage = 5
        stage_positions = init_stage_positions(stage)
        stage_map = init_map(stage)

        env = SacEnvV3(
            amr_model=amr_model,
            epoch=(i + 1),
            init_positions=stage_positions,
            stage_map=stage_map,
            max_timesteps=timesteps,
            test_mode=False
        )
        #env.reset()

        check_env(env)

        print(os.path.exists(model_pth))
        model = SAC.load(path=model_pth, env=env) if os.path.exists(model_pth) else SAC('MlpPolicy', env, ent_coef='auto', verbose=1)
        model.learn(total_timesteps=timesteps)
        model.save(model_pth)

        print(f"model saved! Epoch: {i + 1}")

        env.reset()
        time.sleep(5)

    #obs = env.reset()
    #done = False
    #while not done:
        #action, _states = model.predict(obs)
        #obs, rewards, done, info = env.step(action)

if __name__ == '__main__':
    main()

