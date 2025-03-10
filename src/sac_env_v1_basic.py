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

from initialisation import sdf_templates, init_state
from data_collection import data_collector

class SacEnv(gym.Env):
    def __init__(self, amr_model='turtlebot3_burger', epoch=0, init_positions=[], stage_map="", yaw=0.0, max_timesteps=10000, test_mode=False):
        super(SacEnv, self).__init__()

        self.velocity_multiplier = 0.22
        self.angular_velocity_multiplier = 2.84
        self.velocity = self.velocity_multiplier
        self.angular_velocity = 0.0

        self.laserscan_maxcap = 3.5
        self.laserscan_mincap = 0.12
        self.laserscan_closest = self.laserscan_maxcap
        self.laserscan = np.full(12, self.laserscan_maxcap)

        self.angle_cap = 2 * math.pi

        self.init_positions = init_positions
        self.init_positions_previous = self.init_positions
        self.spawn_position = self.init_positions[0]
        self.position = self.spawn_position
        self.goal_position = self.init_positions[1]
            
        self.goal_distance_from_spawn_vector = None
        self.goal_distance_from_spawn = None
        self.goal_distance_previous = None
        self.goal_distance_record = None
        self.goal_angle_from_spawn = None

        self.spawn_orientation = np.array([0.0, 1.0]) # [z, w]
        self.yaw = 0.0

        self.done = False
        self.truncated = False
        self.total_timesteps = 0
        self.step_count = 0
        self.max_step_count = 500
        self.stagnant_count = 0
        self.max_stagnant_count = 10
        self.reset_count = 0
        self.observation_state = []

        self.amr_model = amr_model
        self.epoch = epoch

        self.goal_radius = 0.2

        self.goal_sdf = sdf_templates.goal_sdf(self.goal_radius)

        self.database = "data_v1.csv" if test_mode else "data_v1_train.csv"
        self.data = data_collector.find_csv(
            self.database, 
            pd.DataFrame({
                'episode': [],
                'final position x': [],
                'final position y': [],
                'success': [],
            })
        )

        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/gazebo/delete_model')

        init_state.reset_turtlebot3_gazebo(self.spawn_position, self.amr_model, randomise=True)
        init_state.reset_goal(self.goal_position, self.goal_sdf)

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

        '''
        Observation space: [
            normalised distance from robot to goal,
            angle from robot to goal (radians),
            laserscan distance to closest obstacle,
            laserscan spatial values (12 values 30 deg apart)
        ]
        '''
        self.observation_space = spaces.Box(
            low=np.append(
                np.array([
                    0.0,
                    -self.angle_cap,
                    self.laserscan_mincap,
                ]),
                np.full(12, self.laserscan_mincap)
            ),
            high=np.append(
                np.array([
                    100.0,
                    self.angle_cap,
                    self.laserscan_maxcap,
                ]),
                np.full(12, self.laserscan_maxcap)
            ),
            shape=(15,), 
            dtype=np.float32
        )

    def laserscan_callback(self, scan):
        laserscan_360 = np.clip(np.array(scan.ranges), self.laserscan_mincap, self.laserscan_maxcap)
        laserscan = np.array([])

        for i in range(12):
            laserscan = np.append(laserscan, laserscan_360[i * 30])

        self.laserscan = laserscan
        self.laserscan_closest = np.min(laserscan_360)

    def odometry_callback(self, odom):
        position = odom.pose.pose.position
        self.position = np.array([position.x, position.y])

    def imu_callback(self, imu):
        orientation = imu.orientation
        euler = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        self.yaw = euler[2]

    def publish_velocity(self, velocity, angular_velocity, laserscan_closest=1, goal_distance=1):
        twist = Twist()
        twist.linear.x = velocity
        twist.angular.z = angular_velocity
        self.twist_publisher.publish(twist)

    def _get_observation_state(self):
        current_position = self.position
        goal_position = self.goal_position

        goal_distance_vector = goal_position - current_position
        goal_distance = np.linalg.norm(goal_distance_vector)
        goal_distance_normalised = goal_distance / self.goal_distance_from_spawn # Normalised to 1

        yaw = self.yaw
        goal_angle = self.normalise_radians_angle(
            math.atan2(
                goal_position[1] - current_position[1], 
                goal_position[0] - current_position[0]
            ) - yaw
        )

        laserscan_closest = self.laserscan_closest
        laserscan = self.laserscan

        return np.nan_to_num(
            np.append(
                np.array([
                    goal_distance_normalised,
                    goal_angle,
                    laserscan_closest,
                ]),
                laserscan
            )
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.reset_count += 1
        self.publish_velocity(0, 0)

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

        init_state.reset_turtlebot3_gazebo(self.spawn_position, self.amr_model, randomise=True)
        #init_state.reset_goal(self.goal_position, self.goal_sdf)

        self.done = False
        self.truncated = False

        self.step_count = 0
        self.stagnant_count = 0

        self.laserscan_closest = self.laserscan_maxcap

        self.observation_state = self._get_observation_state()

        print(self.observation_state)

        return self.observation_state, {}

    def step(self, action):
        self.total_timesteps += 1
        self.step_count += 1

        self.done = False
        self.truncated = False
        
        self.observation_state = self._get_observation_state()

        self.velocity = float((action[0] + 1) * 0.5 * self.velocity_multiplier)
        self.angular_velocity = float(action[1] * self.angular_velocity_multiplier)
        self.publish_velocity(
            velocity = self.velocity,
            angular_velocity = self.angular_velocity,
            laserscan_closest = self.observation_state[3], 
            goal_distance = self.observation_state[0] * self.goal_distance_from_spawn
        )

        rospy.sleep(0.05)

        reward = self._compute_reward()

        print("------------------------------------------")
        print("OBSERVATION SPACE")
        print(f"goal_distance_normalised: {self.observation_state[0]}")
        print(f"goal_angle: {self.observation_state[1]}")
        print(f"laserscan_closest: {self.observation_state[2]}")
        print(f"laserscan_front: {self.observation_state[3]}")
        print(f"laserscan_left: {self.observation_state[6]}")
        print(f"laserscan_back: {self.observation_state[9]}")
        print(f"laserscan_right: {self.observation_state[12]}")
        print(" ")
        print("ACTION SPACE")
        print(f"velocity : {self.velocity}")
        print(f"angular_velocity : {self.angular_velocity}")
        print(" ")
        print(f"epoch: {self.epoch}")
        print(f"total_timesteps: {self.total_timesteps}")
        print(f"step_count: {self.step_count}/{self.max_step_count}")
        print(f"reward: {reward}")
        print(f"record: {self.goal_distance_record}")
        print("------------------------------------------")
        print(" ")

        return self.observation_state, reward, self.done, self.truncated, {}

    def _compute_reward(self):
        goal_distance_normalised = self.observation_state[0]
        goal_distance = goal_distance_normalised * self.goal_distance_from_spawn
        goal_distance_previous = self.goal_distance_previous

        position = self.position

        goal_angle = self.observation_state[1]

        laserscan_closest = self.observation_state[2]

        step_count = self.step_count

        collision_threshold = self.laserscan_mincap + 0.02

        reward_goal = 100 + (20 / self.max_step_count) * (self.max_step_count - step_count)

        self.goal_distance_previous = goal_distance

        reward_distance_from_goal = 1.0 * (1 - goal_distance_normalised)
        reward_facing_goal = 0.2 if (
            (abs(goal_angle) < math.pi/4) and (laserscan_closest > 0.3)
        ) else 0

        penalty_obstacle_proximity = 0 if (
            laserscan_closest > 0.3
        ) else (
            -4.0 * (0.3 - laserscan_closest) / (0.3 - collision_threshold)
        )
        penalty_collision = -100
        penalty_step_count = -0.5 * step_count / self.max_step_count
        penalty_step_count_maxed = -25

        if self.step_count >= self.max_step_count:
           reward = reward_distance_from_goal + penalty_step_count_maxed
           self.end_episode(position[0], position[1], 0)
        else:
            if self.goal_distance_record > goal_distance_normalised:
                self.goal_distance_record = goal_distance_normalised

            if laserscan_closest < collision_threshold:
                reward = reward_distance_from_goal + penalty_collision + penalty_obstacle_proximity + penalty_step_count + reward_facing_goal
                self.end_episode(position[0], position[1], 0)
                print(f"!!!!!ROBOT COLLISION!!!!! scan: {laserscan_closest}")
            else:
                reward = reward_distance_from_goal + penalty_obstacle_proximity + penalty_step_count + reward_facing_goal

                if goal_distance < self.goal_radius:
                    reward += reward_goal

                    self.end_episode(position[0], position[1], 1)
                    print(f"!!!!!ROBOT GOAL REACHED!!!!!")

        return float(reward)
    
    def end_episode(self, pos_x, pos_y, success):
        self.goal_distance_record = 1.0
        data_collector.collect_data(self.data, pos_x, pos_y, success, self.database)
        self.done = True
    
    def degree_to_radians(self, angle):
        angle = angle * (math.pi / 180)
        return self.normalise_radians_angle(angle)
    
    def normalise_radians_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi

        return angle

