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
    def __init__(self, amr_model='turtlebot3_burger'):
        super(SacEnv, self).__init__()

        self.velocity_multiplier = 0.15
        self.angular_velocity_multiplier = 2.84
        self.velocity = self.velocity_multiplier
        self.angular_velocity = 0.0

        self.laserscan_maxcap = 3.5
        self.laserscan_mincap = 0.12
        self.laserscan_closest = self.laserscan_maxcap
        self.laserscan = np.full(24, self.laserscan_maxcap)

        self.angle_cap = 2 * math.pi

        self.stage = 4
        self.init_positions = np.array([[0.0, 0.0], [1.0, 1.0]])
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

        self.goal_database = r"/home/aravestia/isim/noetic/src/robot_planner/src/goal.csv"
        self.goal_df = pd.read_csv(self.goal_database, header=0, index_col=0)
        self.goal_count = len(self.goal_df)
        self.goal_radius = 0.2

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
        for i in range(1,10001):
            if (f'goal_marker_{str(i)}') in self.model_names:
                self.delete_model(f'goal_marker_{str(i)}')
        #########################################

        self.laserscan_subscriber = rospy.Subscriber('/scan', LaserScan, self.laserscan_callback)
        self.odometry_subscriber = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
        self.imu_subscriber = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.twist_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Action space: [velocity, angular velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0]), 
            high=np.array([1.0]),
            dtype=np.float32
        )

        '''
        Observation space: [
            normalised distance from robot to goal,
            angle from robot to goal (radians),
            yaw
            laserscan distance to closest obstacle,
            laserscan spatial values (24 values 15 deg apart)
        ]
        '''
        self.observation_space = spaces.Box(
            low=np.append(
                np.array([
                    0.0,
                    -self.angle_cap,
                    -self.angle_cap,
                    self.laserscan_mincap,
                ]),
                np.full(24, self.laserscan_mincap)
            ),
            high=np.append(
                np.array([
                    100.0,
                    self.angle_cap,
                    self.angle_cap,
                    self.laserscan_maxcap,
                ]),
                np.full(24, self.laserscan_maxcap)
            ),
            shape=(28,), 
            dtype=np.float32
        )

    def laserscan_callback(self, scan):
        laserscan_360 = np.clip(np.array(scan.ranges), self.laserscan_mincap, self.laserscan_maxcap)
        laserscan = np.array([])

        for i in range(24):
            laserscan = np.append(laserscan, laserscan_360[i * 15])

        self.laserscan = laserscan
        self.laserscan_closest = np.min(laserscan_360)

    def odometry_callback(self, odom):
        position = odom.pose.pose.position
        self.position = np.array([position.x, position.y])

    def imu_callback(self, imu):
        orientation = imu.orientation
        euler = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        self.yaw = euler[2]

    def publish_velocity(self, angular_velocity, laserscan_closest=1, goal_distance=1):
        self.velocity = self.velocity_multiplier if (
            goal_distance > self.goal_radius + 0.15
        ) else (0.25 * self.velocity_multiplier)

        twist = Twist()
        twist.linear.x = self.velocity
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
                    yaw,
                    laserscan_closest,
                ]),
                laserscan
            )
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.reset_count += 1
        self.publish_velocity(0)

        while (self.init_positions == self.init_positions_previous).all():
            self.init_positions = self.init_stage_positions(self.stage)

        self.spawn_position = self.init_positions[0]
        self.init_positions_previous = self.init_positions

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

        self.reset_goal()
        self.reset_turtlebot3_gazebo()

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

        #self.velocity = float((action[0] + 1) * 0.5 * self.velocity_multiplier)
        self.angular_velocity = float(action[0] * self.angular_velocity_multiplier)
        self.publish_velocity(
            angular_velocity = self.angular_velocity,
            laserscan_closest = self.observation_state[3], 
            goal_distance = self.observation_state[0] * self.goal_distance_from_spawn
        )

        rospy.sleep(0.08)

        reward = self._compute_reward()

        print("------------------------------------------")
        print("OBSERVATION SPACE")
        print(f"goal_distance_normalised: {self.observation_state[0]}")
        print(f"goal_angle: {self.observation_state[1]}")
        print(f"yaw: {self.observation_state[2]}")
        print(f"laserscan_closest: {self.observation_state[3]}")
        print(f"laserscan_front: {self.observation_state[4]}")
        print(f"laserscan_left: {self.observation_state[10]}")
        print(f"laserscan_back: {self.observation_state[16]}")
        print(f"laserscan_right: {self.observation_state[22]}")
        print(" ")
        print("ACTION SPACE")
        print(f"velocity : {self.velocity}")
        print(f"angular_velocity : {self.angular_velocity}")
        print(" ")
        print(f"total_timesteps: {self.total_timesteps}")
        print(f"step_count: {self.step_count}/{self.max_step_count}")
        print(f"reward: {reward}")
        print(f"record: {self.goal_distance_record}")
        print(f"goal count: {self.goal_count}")
        print("------------------------------------------")
        print(" ")

        return self.observation_state, reward, self.done, self.truncated, {}

    def _compute_reward(self):
        goal_distance_normalised = self.observation_state[0]
        goal_distance = goal_distance_normalised * self.goal_distance_from_spawn
        goal_distance_previous = self.goal_distance_previous

        goal_angle = self.observation_state[1]

        laserscan_closest = self.observation_state[3]

        step_count = self.step_count
        stagnant_count = self.stagnant_count

        collision_threshold = self.laserscan_mincap + 0.01

        reward_goal = 15 + (20 / self.max_step_count) * (self.max_step_count - step_count)
        reward_distance_from_goal = -0.5 if (
            goal_distance_normalised > 1.0
        ) else (
            2 * (1 - goal_distance_normalised) + ((1 - goal_distance_normalised) ** 2)
        )
        reward_facing_goal = 0.5 if (
            (abs(goal_angle) < math.pi/4) and (laserscan_closest > collision_threshold + 0.02)
        ) else 0

        if abs(goal_distance - goal_distance_previous) < 0.002 and goal_distance >= self.goal_radius:
            stagnant_count += 1
            print("!!!ROBOT STAGNANT!!!")
        else:
            stagnant_count = 0

        self.stagnant_count = stagnant_count
        self.goal_distance_previous = goal_distance

        penalty_obstacle_proximity = 0 if (
            laserscan_closest > 0.3
        ) else (
            -10 * (((self.laserscan_maxcap - laserscan_closest) / (self.laserscan_maxcap - self.laserscan_mincap)) ** 50)
        )
        penalty_collision = -10
        penalty_step_count = -(1 / self.max_step_count) * (step_count - 1)
        penalty_step_count_maxed = -15

        if self.step_count >= self.max_step_count:
           reward = penalty_step_count_maxed
           self.end_episode()
        else:
            if self.goal_distance_record > goal_distance_normalised:
                self.goal_distance_record = goal_distance_normalised

            if laserscan_closest < collision_threshold:
                reward = penalty_collision + penalty_obstacle_proximity + penalty_step_count
                self.end_episode()
                print(f"!!!!!ROBOT COLLISION!!!!! scan: {laserscan_closest}")
            else:
                reward = reward_distance_from_goal
                reward += penalty_obstacle_proximity + penalty_step_count

                reward += reward_facing_goal

                if goal_distance < self.goal_radius:
                    reward += reward_goal

                    self.goal_count += 1
                    self.end_episode()
                    print(f"!!!!!ROBOT GOAL REACHED!!!!!")

                    self.goal_df.loc[self.goal_count] = {
                        'id': self.goal_count, 
                        'steps': self.step_count, 
                        'stage': self.stage,
                        'time': datetime.now().strftime("%d/%m/%Y"),
                    }
                    self.goal_df.to_csv(self.goal_database)

        return float(reward)
    
    def end_episode(self):
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

        world_properties = self.get_world_properties()

        if world_properties and (model_name + str(self.reset_count - 1)) in world_properties.model_names:
            self.delete_model(model_name + str(self.reset_count - 1))

        self.spawn_model(model_state_msg.model_name, self.goal_sdf, "", model_state_msg.pose, "world")
        print(f"Goal set. {self.goal_position}")

    def reset_turtlebot3_gazebo(self):
        yaw = math.pi * random.uniform(-1, 1)
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
        
    def init_stage_positions(self, stage):
        init_positions = np.array([[], []])

        if stage == 1:
            init_positions = np.array(random.choice([
                [[1, 1], [-1, -1]],
                [[-1, -1], [1, 1]],
                [[1, -1], [-1, 1]],
                [[-1, 1], [1, -1]]
            ]))

        if stage == 2:
            init_positions = np.array(random.choice([
                [[1, 1], [-1.25, -1.25]],
                [[-1, -1], [1.25, 1.25]],
                [[1, -1], [-1.25, 1.25]],
                [[-1, 1], [1.25, -1.25]]
            ]))

        if stage == 3:
            init_positions = np.array(random.choice([
                [[0, 0], [-1.25, -1.25]],
                [[0, 0], [1.25, 1.25]],
                [[0, 0], [-1.25, 1.25]],
                [[0, 0], [1.25, -1.25]]
            ]))

        if stage == 4:
            init_positions = np.array(random.choice([
                [[2, -1.5], [-2, 2]],
                [[1.5, -2], [-2, 2]],
                [[-2, 1.5], [2, -2]],
                [[-1.5, 2], [2, -2]]
            ]))

        return init_positions
    
    def degree_to_radians(self, angle):
        angle = angle * (math.pi / 180)
        return self.normalise_radians_angle(angle)
    
    def normalise_radians_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi

        return angle

def main(args=None):
    timesteps = 12 * 10000

    rospy.init_node('sac_env', anonymous=True)

    # Depends on map
    amr_model = 'turtlebot3_burger'
    model_pth = r"/home/aravestia/isim/noetic/src/robot_planner/src/models/sac_model.pth"

    env = SacEnv(amr_model=amr_model)
    #env.reset()

    check_env(env)

    print(os.path.exists(model_pth))
    model = SAC.load(path=model_pth, env=env) if os.path.exists(model_pth) else SAC('MlpPolicy', env, ent_coef='auto', verbose=1)
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

