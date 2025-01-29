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
#import time

from sdf_files import goal_sdf

class SacEnv(gym.Env):
    def __init__(
            self,
            amr_model='turtlebot3_burger'
        ):
        super(SacEnv, self).__init__()

        self.velocity_multiplier = 0.22
        self.angular_velocity_multiplier = 2.84
        self.velocity = 0.0
        self.angular_velocity = 0.0

        self.laserscan_maxcap = 3.5
        self.laserscan_mincap = 0.12
        self.laserscan_min = self.laserscan_maxcap
        self.laserscan_min_angle = 0.0
        self.laserscan = np.arange(360).fill(self.laserscan_maxcap)

        self.angle_cap = 2 * math.pi

        self.init_position = None
        self.position = None
        self.goal_position = None
            
        self.goal_distance_from_spawn_vector = None
        self.goal_distance_from_spawn = None
        self.goal_distance_previous = None
        self.goal_distance_record = None
        self.init_angle_from_goal = None

        self.init_orientation = np.array([0.0, 1.0]) # [z, w]
        self.yaw = 0.0
        self.acceleration = np.array([0.0, 0.0]) # [x, y]

        self.done = False
        self.truncated = False
        self.total_timesteps = 0
        self.step_count = 0
        self.max_step_count = 1000
        self.stagnant_count = 0
        self.max_stagnant_count = 10
        self.reset_count = 0
        self.observation_state = []

        self.amr_model = amr_model

        self.goal_database = r"/home/aravestia/isim/noetic/src/robot_planner/src/goal.csv"
        self.goal_df = pd.read_csv(self.goal_database, header=0, index_col=0)
        self.goal_count = len(self.goal_df)

        print(f"{self.goal_df}. {self.goal_count}")

        self.goal_sdf = goal_sdf

        self.laserscan_subscriber = rospy.Subscriber('/scan', LaserScan, self.laserscan_callback)
        self.odometry_subscriber = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
        self.imu_subscriber = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.twist_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        rospy.wait_for_service('/gazebo/set_model_state')
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/gazebo/delete_model')

        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)

        # Action space: [velocity, angular velocity]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]), 
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space: [
        #   normalised distance from robot to goal,
        #   normalised distance from robot to goal x,
        #   normalised distance from robot to goal y,
        #   angle from robot to goal (radians),
        #   laserscan distance to closest obstacle, 
        #   angle of min laserscan distance (radians),
        #   yaw,
        #   linear acceleration x,
        #   linear acceleration y,
        # ]
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,
                0.0,
                0.0,
                -self.angle_cap,
                self.laserscan_mincap,
                0.0, 
                -self.angle_cap,
                -100,
                -100,
            ]), 
            high=np.array([
                100.0,
                100.0,
                100.0,
                self.angle_cap,
                self.laserscan_maxcap, 
                self.angle_cap, 
                self.angle_cap,
                100,
                100
            ]),
            shape=(9,), 
            dtype=np.float32
        )

    def laserscan_callback(self, scan):
        self.laserscan = np.clip(np.array(scan.ranges), self.laserscan_mincap, self.laserscan_maxcap)
        self.laserscan_min = np.min(self.laserscan)
        self.laserscan_min_angle = np.where(self.laserscan == self.laserscan_min)[0][0] * math.pi/180

    def odometry_callback(self, odom):
        position = odom.pose.pose.position
        self.position = np.array([position.x, position.y])

    def imu_callback(self, imu):
        orientation = imu.orientation
        acceleration = imu.linear_acceleration

        euler = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        self.yaw = euler[2]
        self.acceleration = np.array([acceleration.x, acceleration.y])

    def publish_velocity(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.twist_publisher.publish(twist)

    def _get_observation_state(self):
        current_position = self.position
        goal_position = self.goal_position

        goal_distance_vector = goal_position - current_position
        goal_distance = np.linalg.norm(goal_distance_vector)
        goal_distance_normalised = goal_distance / self.goal_distance_from_spawn # Normalised to 1
        goal_distance_normalised_x = abs(goal_distance_vector[0] / self.goal_distance_from_spawn_vector[0])
        goal_distance_normalised_y = abs(goal_distance_vector[1] / self.goal_distance_from_spawn_vector[1])
        goal_angle = math.atan2(goal_position[1] - current_position[1], goal_position[0] - current_position[0]) # arctan(y/x) in radians

        laserscan_min = self.laserscan_min
        laserscan_min_angle = self.laserscan_min_angle

        yaw = self.yaw

        acceleration_x = self.acceleration[0]
        acceleration_y = self.acceleration[1]

        return np.nan_to_num(np.array([
            goal_distance_normalised,
            goal_distance_normalised_x,
            goal_distance_normalised_y,
            goal_angle,
            laserscan_min, 
            laserscan_min_angle,
            yaw,
            acceleration_x,
            acceleration_y
        ])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.publish_velocity(0, 0)

        self.init_position = np.array(random.choice([
            [-2, random.uniform(-1, 1)],
            [2, random.uniform(-1, 1)],
            [random.uniform(-1, 1), 2],
            [random.uniform(-1, 1), -2],
        ]))

        self.position = self.init_position
        self.goal_position = np.array([0.5 * random.choice([-1, 1]), 0.5 * random.choice([-1, 1])])
            
        self.goal_distance_from_spawn_vector = self.goal_position - self.init_position
        self.goal_distance_from_spawn = np.linalg.norm(self.goal_distance_from_spawn_vector)
        self.goal_distance_previous = self.goal_distance_from_spawn
        self.goal_distance_record = self.goal_distance_from_spawn
        self.init_angle_from_goal = math.atan2(self.goal_position[1] - self.init_position[1], self.goal_position[0] - self.init_position[0])

        self.yaw = math.pi * random.uniform(-1, 1)

        self.reset_goal()
        self.reset_turtlebot3_gazebo()

        self.done = False
        self.truncated = False

        self.step_count = 0
        self.stagnant_count = 0

        self.laserscan_min = self.laserscan_maxcap

        self.observation_state = self._get_observation_state()

        print(self.observation_state)

        return self.observation_state, {}
    
    def reset_goal(self):
        model_name = "goal_marker_"

        goal_state_msg = ModelState()
        goal_state_msg.model_name = model_name + str(self.reset_count)
        goal_state_msg.pose.position.x = self.goal_position[0]
        goal_state_msg.pose.position.y = self.goal_position[1]
        goal_state_msg.pose.position.z = 0.0
        goal_state_msg.pose.orientation.x = 0.0
        goal_state_msg.pose.orientation.y = 0.0
        goal_state_msg.pose.orientation.z = 0.0
        goal_state_msg.pose.orientation.w = 1.0
        goal_state_msg.reference_frame = 'world'

        world_properties = self.get_world_properties()

        if (model_name + str(self.reset_count - 1)) in world_properties.model_names:
            self.delete_model(model_name + str(self.reset_count - 1))

        self.spawn_model(goal_state_msg.model_name, self.goal_sdf, "", goal_state_msg.pose, "world")
        print(f"Goal set. {self.goal_position}")

    def reset_turtlebot3_gazebo(self):
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, self.yaw)

        turtlebot3_state_msg = ModelState()
        turtlebot3_state_msg.model_name = self.amr_model
        turtlebot3_state_msg.pose.position.x = self.init_position[0]
        turtlebot3_state_msg.pose.position.y = self.init_position[1]
        turtlebot3_state_msg.pose.position.z = 0.0
        turtlebot3_state_msg.pose.orientation.x = 0.0
        turtlebot3_state_msg.pose.orientation.y = 0.0
        turtlebot3_state_msg.pose.orientation.z = quaternion[2]
        turtlebot3_state_msg.pose.orientation.w = quaternion[3]
        turtlebot3_state_msg.twist.linear.x = 0.0
        turtlebot3_state_msg.twist.linear.y = 0.0
        turtlebot3_state_msg.twist.linear.z = 0.0
        turtlebot3_state_msg.twist.angular.x = 0.0
        turtlebot3_state_msg.twist.angular.y = 0.0
        turtlebot3_state_msg.twist.angular.z = 0.0
        turtlebot3_state_msg.reference_frame = 'world'

        self.set_model_state(turtlebot3_state_msg)
        print(f"Turtlebot set. {self.init_position}")

    def step(self, action):
        self.total_timesteps += 1
        self.step_count += 1

        self.done = False
        self.truncated = False

        self.velocity = float(action[0] * self.velocity_multiplier)
        self.angular_velocity = float(action[1] * self.angular_velocity_multiplier)
        self.publish_velocity(self.velocity, self.angular_velocity)

        rospy.sleep(0.08)

        reward = self._compute_reward()
        print(f"[{self.total_timesteps}][{self.step_count}] State: {np.around(self.observation_state, 4)}")
        print(f"[{self.total_timesteps}][{self.step_count}] Twist: ({round(self.velocity, 4)}, {round(self.angular_velocity, 4)})")
        print(f"[{self.total_timesteps}][{self.step_count}] Reward: {round(reward, 3)}, Record: {round(self.goal_distance_record, 4)}, Goals: {self.goal_count}")
        
        if self.step_count >= self.max_step_count:
           self.done = True

        return self.observation_state, reward, self.done, self.truncated, {}

    def _compute_reward(self):
        self.observation_state = self._get_observation_state()

        goal_distance_normalised = self.observation_state[0]
        goal_distance = goal_distance_normalised * self.goal_distance_from_spawn
        goal_distance_previous = self.goal_distance_previous

        laserscan_min = self.observation_state[4]

        step_count = self.step_count

        collision_threshold = 0.13

        reward_goal = 15 + (20 / self.max_step_count) * (self.max_step_count - step_count)
        reward_distance_from_goal = (math.e ** (2.5 * (1.0 - goal_distance_normalised))) - goal_distance_normalised

        penalty_obstacle_proximity = -20 * (((self.laserscan_maxcap - laserscan_min) / (self.laserscan_maxcap - self.laserscan_mincap)) ** 60)
        penalty_collision = -20
        penalty_stagnant = -3
        penalty_backwards_from_goal = -2
        penalty_step_count = -(2 / self.max_step_count) * (step_count - 1)

        reward = reward_distance_from_goal + penalty_obstacle_proximity + penalty_step_count

        if goal_distance + 0.001 < self.goal_distance_record:
            self.goal_distance_record = goal_distance

        if goal_distance - 0.001 > self.goal_distance_record:
            reward += penalty_backwards_from_goal

        if abs(goal_distance - goal_distance_previous) < 0.002 and goal_distance >= 0.2:
            self.stagnant_count += 1
            reward += penalty_stagnant
            #print(f"!!!ROBOT STAGNANT!!! [{self.stagnant_count}] now: {round(goal_distance, 4)}, prev: {round(goal_distance_previous, 4)}")
        else:
            self.stagnant_count = 0

        self.goal_distance_previous = goal_distance

        if goal_distance < 0.1:
            # Large positive reward for reaching goal + efficiency
            reward += reward_goal
            self.goal_count += 1
            self.reset_count += 1
            self.goal_distance_record = self.goal_distance_from_spawn
            self.done = True
            print(f"!!!!!GOAL REACHED!!!!!")

            self.goal_df.loc[self.goal_count] = {'id': self.goal_count, 'steps': self.step_count, 'time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
            self.goal_df.to_csv(self.goal_database)
        else:
            if laserscan_min < collision_threshold:
                # Large penalty for being dangerously close to a wall
                reward += penalty_collision
                self.reset_count += 1
                self.goal_distance_record = self.goal_distance_from_spawn
                self.done = True
                print(f"!!!!!COLLISION DETECTED!!!!! scan: {round(self.laserscan_min, 4)}")

        return float(reward)

def main(args=None):
    timesteps = 2 * 10000

    rospy.init_node('sac_env', anonymous=True)

    # Depends on map
    amr_model = 'turtlebot3_burger'
    model_pth = r"/home/aravestia/isim/noetic/src/robot_planner/src/models/sac_model_2.pth"

    env = SacEnv(amr_model=amr_model)
    #env.reset()

    check_env(env)

    model = SAC.load(path=model_pth, env=env) if os.path.exists(model_pth) else SAC('MlpPolicy', env, ent_coef='auto', verbose=1, learning_rate=0.001)
    model.learn(total_timesteps=timesteps)
    model.save(model_pth)

    print("model saved!")

    env.reset()

    #obs = env.reset()
    #done = False
    #while not done:
        #action, _states = model.predict(obs)
        #obs, rewards, done, info = env.step(action)

if __name__ == '__main__':
    main()

