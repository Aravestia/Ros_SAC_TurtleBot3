#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

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
import asyncio

import a_star
import create_sdf
import reset_state

class SacEnvV3_5(gym.Env):
    def __init__(self, amr_model='turtlebot3_burger', epoch=0, init_positions=[], stage_map="", yaw=0.0, max_timesteps=10000, test_mode=False):
        super(SacEnvV3_5, self).__init__()

        self.velocity_multiplier = 0.15
        self.angular_velocity_multiplier = 2.84
        self.velocity = -1.0
        self.velocity_previous = -1.0
        self.angular_velocity = 0.0
        self.angular_velocity_previous = 0.0

        self.laserscan_maxcap = 3.5
        self.laserscan_mincap = 0.12
        self.laserscan_warning_threshold = 0.3
        self.laserscan_warning_threshold_normalised = self.normalise_value(self.laserscan_warning_threshold, self.laserscan_maxcap, self.laserscan_mincap)
        self.laserscan_front_warning_threshold = 0.5
        self.laserscan_front_warning_threshold_normalised = self.normalise_value(self.laserscan_front_warning_threshold, self.laserscan_maxcap, self.laserscan_mincap)
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
        self.goal_angle_from_spawn = math.atan2(
            self.goal_position[1] - self.spawn_position[1], 
            self.goal_position[0] - self.spawn_position[0]
        )
        self.current_center = self.spawn_position
        self.current_distance_from_waypoint = 0.0

        self.spawn_orientation = np.array([0.0, 1.0]) # [z, w]
        self.yaw = yaw

        self.done = False
        self.truncated = False
        self.total_timesteps = 0
        self.max_timesteps = max_timesteps
        self.step_count = 0
        self.max_step_count = 1000 if not test_mode else 10000
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
        self.grid = a_star.img_to_grid(stage_map, self.grid_margin)
        self.grid_in = copy.deepcopy(self.grid)
        self.grid_x_offset = 200
        self.grid_y_offset = 184
        self.grid_resolution = 0.05
        self.grid_spawn = a_star.point_to_grid(self.grid_x_offset, self.grid_y_offset, self.spawn_position[0], self.spawn_position[1])
        self.grid_goal = a_star.point_to_grid(self.grid_x_offset, self.grid_y_offset, self.goal_position[0], self.goal_position[1])

        self.waypoint_occurrence = 2
        self.waypoints = a_star.a_star_search(
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
        self.waypoint_min_distance = 0.5
        self.waypoint_min_distance_threshold = 0.15
        self.waypoint_lookahead = 30
        self.waypoint_closest_angle = 3

        self.follower_mode = False
        self.follower_mode_previous = False
        self.test_mode = test_mode

        self.moving_obstacle_radius = 0.15
        self.goal_sdf = create_sdf.goal_sdf(self.goal_radius)

        self.cumulative_reward_file_path = r"/home/aravestia/isim/noetic/src/robot_planner/src/cumulative_reward.csv"
        self.cumulative_reward = 0
        self.cumulative_reward_df = pd.read_csv(self.cumulative_reward_file_path, index_col=0)

        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/gazebo/delete_model')

        reset_state.reset_turtlebot3_gazebo(self.spawn_position, self.amr_model)
        reset_state.reset_goal(self.goal_position, self.goal_sdf)

        self.laserscan_subscriber = rospy.Subscriber('/scan', LaserScan, self.laserscan_callback)
        self.odometry_subscriber = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
        self.imu_subscriber = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.twist_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.publish_velocity(0.0, 0.0)

        # Action space: [velocity, angular velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.observation_state_titles = [
            'distance_from_waypoint',
            'velocity_normalised',
            'angular_velocity_normalised', 
            'waypoint_closest_angle',
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

        self.waypoint_closest, self.waypoint_min_distance = a_star._get_closest_waypoint(
            current_position, 
            self.waypoints, 
            self.waypoint_closest, 
            self.waypoint_lookahead, 
            len(self.waypoints)
        )

        if self.waypoint_closest >= len(self.waypoints) and len(self.waypoints) > 1:
            self.waypoint_closest = len(self.waypoints) - 1

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
        waypoint_closest_angle = self.waypoint_closest_angle

        self.follower_mode = (
        laserscan_closest > self.laserscan_warning_threshold_normalised
        ) and (
            laserscan[0] > self.laserscan_front_warning_threshold_normalised
        ) and (
            laserscan[7] > self.laserscan_front_warning_threshold_normalised
        ) and self.test_mode

        if self.follower_mode_previous != self.follower_mode and self.test_mode:
            self.current_center = current_position
        
        self.follower_mode_previous = self.follower_mode

        waypoint_min_distance_normalised = (2 / (1 + np.exp(-2 * self.waypoint_min_distance))) - 1

        return np.nan_to_num(
            np.append(
                np.array([
                    waypoint_min_distance_normalised,
                    velocity,
                    angular_velocity,
                    waypoint_closest_angle,
                    laserscan_closest_angle,
                ]),
                laserscan
            )
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.reset_count += 1

        self.cumulative_reward = 0

        self.velocity = -1.0
        self.velocity_previous = -1.0
        self.angular_velocity = 0.0
        self.angular_velocity_previous = 0.0

        self.publish_velocity(0.0, 0.0)

        self.spawn_position = self.init_positions[0]
        self.position = self.spawn_position
        self.goal_position = self.init_positions[1]

        reset_state.reset_turtlebot3_gazebo(self.spawn_position, self.amr_model)
            
        self.goal_distance_from_spawn_vector = self.goal_position - self.spawn_position
        self.goal_distance_from_spawn = np.linalg.norm(self.goal_distance_from_spawn_vector)
        self.goal_distance_previous = self.goal_distance_from_spawn
        self.goal_angle_from_spawn = math.atan2(
            self.goal_position[1] - self.spawn_position[1], 
            self.goal_position[0] - self.spawn_position[0]
        )
        self.current_center = self.spawn_position
        self.current_distance_from_waypoint = 0.0

        self.grid_in = copy.deepcopy(self.grid)
        self.grid_spawn = a_star.point_to_grid(self.grid_x_offset, self.grid_y_offset, self.spawn_position[0], self.spawn_position[1])
        self.grid_goal = a_star.point_to_grid(self.grid_x_offset, self.grid_y_offset, self.goal_position[0], self.goal_position[1])

        self.waypoints = a_star.a_star_search(
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

        if self.waypoint_min_distance < self.waypoint_min_distance_threshold:
            self.waypoint_closest += 1

        if not self.follower_mode:
            self.velocity = float(action[0])
            self.angular_velocity = float(action[1])
        else:
            min_velocity = 0.0
            turning_rate = 1.0 / self.angular_velocity_multiplier
            angle_threshold = 0.15

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
        print(" ")
        print(f"epoch: {self.epoch}")
        print(f"total_timesteps: {self.total_timesteps}")
        print(f"step_count: {self.step_count}/{self.max_step_count}")
        print(f"reward: {reward}")
        print(f"current position: {np.round(self.position, 2)}")
        print(f"closest waypoint: {self.waypoint_closest}")
        print(f"total waypoints: {len(self.waypoints)}")
        print(f"completion count: {self.completion_count}")
        print(f"follower mode: {self.follower_mode}")
        print("------------------------------------------")
        print(" ")

        return self.observation_state, reward, self.done, self.truncated, {}

    def _compute_reward(self):
        distance_from_waypoint = self.observation_state[0]
        velocity = self.observation_state[1]
        angular_velocity = self.observation_state[2]
        waypoint_closest_angle = self.observation_state[3]
        laserscan_closest_angle = self.observation_state[4]

        laserscan_quadrant_index = 5
        laserscan_quadrants = np.array([self.observation_state[i + laserscan_quadrant_index] for i in range(8)])
        laserscan_closest = np.min(laserscan_quadrants)
        laserscan_closest_index = np.argmin(laserscan_quadrants)
        print(laserscan_closest_index)

        step_count = self.step_count
        collision_threshold = self.normalise_value(self.laserscan_mincap + 0.01, self.laserscan_maxcap, self.laserscan_mincap)
        warning_threshold = self.laserscan_warning_threshold_normalised
        velocity_threshold = -0.9
        waypoint_distance_threshold = 0.5
        laserscan_angle_threshold = 0.4

        penalty_collision = -1.0
        penalty_step_count_maxed = -1.0
        penalty_step_count = -1.0
        penalty_distance_from_waypoint = 0 if (
            distance_from_waypoint < waypoint_distance_threshold
        ) else (
            -(distance_from_waypoint - waypoint_distance_threshold) / (1 - waypoint_distance_threshold)
        )
        penalty_obstacle_proximity = 0 if (
            laserscan_closest > warning_threshold
        ) else -(warning_threshold - laserscan_closest) / (warning_threshold - collision_threshold)
        penalty_facing_obstacle = 0
        penalty_rapid_acceleration = -0.5 * abs(self.velocity_previous - velocity)
        penalty_rapid_turning = -0.5 * abs(self.angular_velocity_previous - angular_velocity)
        penalty_high_turning = -abs(angular_velocity)

        reward_goal = 1.0
        reward_velocity = 0.5 * (1 + velocity)
        reward_facing_waypoint = max(reward_velocity + penalty_high_turning, 0) if abs(waypoint_closest_angle) < 0.5 else 0
        reward_waypoint = max(reward_velocity + penalty_high_turning, 0) if (
            distance_from_waypoint < waypoint_distance_threshold
        ) else 0
        reward_turning_away = 0

        self.velocity_previous = velocity
        self.angular_velocity_previous = angular_velocity

        reward = 0.0

        if self.total_timesteps > self.max_timesteps - 5:
            self.goal_df.to_csv(self.goal_file_path)
        
        if self.waypoint_closest >= len(self.waypoints) - 1 and distance_from_waypoint < self.waypoint_min_distance_threshold: # Reached Goal
            reward += 10.0 * reward_goal
            self.completion_count += 1
            self.end_episode(float(reward))
            print(f"!!!!!ROBOT GOAL REACHED!!!!!")
            return float(reward)

        reward += 0.02 * reward_waypoint
        reward += 0.005 * reward_facing_waypoint
        #reward += 0.05 * reward_velocity
        #reward += 0.2 * penalty_distance_from_waypoint
        #reward += 0.025 * penalty_rapid_acceleration 
        #reward += 0.025 * penalty_rapid_turning
        #reward += 0.05 * penalty_high_turning
        #reward += 0.05 * penalty_facing_obstacle
        #reward += 0.01 * penalty_obstacle_proximity
        reward += 0.001 * penalty_step_count

        self.cumulative_reward += float(reward)

        if self.step_count >= self.max_step_count: # Maxed Step Count
           reward += 3.0 * penalty_step_count_maxed
           self.end_episode(float(reward))
           return float(reward)

        if laserscan_closest <= collision_threshold: # Collision
            reward += 10.0 * penalty_collision
            self.end_episode(float(reward))
            print(f"!!!!!ROBOT COLLISION!!!!! scan: {laserscan_closest}")
            return float(reward)

        return float(reward)
    
    def end_episode(self, reward):
        self.goal_count += 1

        self.cumulative_reward += reward
        self.cumulative_reward_df.loc[len(self.cumulative_reward_df)] = {
            "episode" : len(self.cumulative_reward_df),
            "cumulative reward" : self.cumulative_reward,
            }
        self.cumulative_reward_df.to_csv(self.cumulative_reward_file_path)

        self.done = True
    
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
