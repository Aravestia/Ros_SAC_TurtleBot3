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
from sac_env_v3 import SacEnvV3
from sac_env_v2 import SacEnvV2

class AStarController():
    def __init__(self, amr_model='turtlebot3_burger', init_positions=[[0.0, 0.0], [1.0, 1.0]], map=""):
        self.velocity_multiplier = 0.15
        self.angular_velocity_multiplier = 2.84
        self.velocity = self.velocity_multiplier
        self.angular_velocity = 0.0

        self.laserscan_maxcap = 3.5
        self.laserscan_mincap = 0.12
        self.laserscan_warning_threshold = 0.3
        self.laserscan_closest = self.laserscan_maxcap
        self.laserscan = np.full(12, self.laserscan_maxcap)

        self.yaw = 0.0

        self.spawn_position = init_positions[0]
        self.goal_position = init_positions[1]
        self.position = init_positions[0]

        self.laserscan_subscriber = rospy.Subscriber('/scan', LaserScan, self.laserscan_callback)
        self.odometry_subscriber = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
        self.imu_subscriber = rospy.Subscriber('/imu', Imu, self.imu_callback)
        self.twist_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.near_obstacle = False
        self.far_obstacle = True

    def laserscan_callback(self, scan):
        laserscan_360 = np.clip(np.array(scan.ranges), self.laserscan_mincap, self.laserscan_maxcap)

        self.laserscan = laserscan_360
        self.laserscan_closest = np.min(laserscan_360)

        self.near_obstacle = self.laserscan_closest < 0.35
        self.far_obstacle = not self.near_obstacle

    def odometry_callback(self, odom):
        position = odom.pose.pose.position
        self.position = np.array([position.x, position.y])

    def imu_callback(self, imu):
        orientation = imu.orientation
        euler = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        self.yaw = euler[2]

    def publish_velocity(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.twist_publisher.publish(twist)

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
    rospy.init_node('a_star_controller', anonymous=True)
    controller = AStarController()

    # Depends on map
    amr_model = 'turtlebot3_burger'
    model_pth_obstacle_avoidance = r"/home/aravestia/isim/noetic/src/robot_planner/src/models/sac_model_v3.0.pth"
    model_pth_a_star = r"/home/aravestia/isim/noetic/src/robot_planner/src/models/sac_model_v2.1-demo.pth"
    stage = 5
    stage_positions = init_stage_positions(stage)
    stage_map = init_map(stage)

    env = SacEnvV2(
    amr_model=amr_model,
    epoch=1,
    init_positions=stage_positions,
    stage_map=stage_map,
    yaw=controller.yaw
    )
    env2 = SacEnvV3(
        amr_model=amr_model,
        epoch=1,
        init_positions=stage_positions,
        stage_map=stage_map,
        yaw=controller.yaw
    )

    model = SAC.load(path=model_pth_a_star, env=env)
    model2 = SAC.load(path=model_pth_obstacle_avoidance, env=env2)
    
    while not rospy.is_shutdown():
        env = SacEnvV2(
            amr_model=amr_model,
            epoch=1,
            init_positions=stage_positions,
            stage_map=stage_map,
            yaw=controller.yaw
        )
        env2 = SacEnvV3(
            amr_model=amr_model,
            epoch=1,
            init_positions=stage_positions,
            stage_map=stage_map,
            yaw=controller.yaw
        )

        model = SAC.load(path=model_pth_a_star, env=env)
        model2 = SAC.load(path=model_pth_obstacle_avoidance, env=env2)

        if controller.near_obstacle:
            obs, info = env2.reset()
            done = False

            while not done:
                done = controller.far_obstacle
                if done:
                    print("Stopping...")
                    controller.publish_velocity()
                    obs, info = env2.reset()
                    break

                action, _states = model2.predict(obs)
                obs, rewards, done, _, info = env2.step(action)
        else:
            obs, info = env.reset()
            done = False

            while not done:
                done = controller.near_obstacle
                if done:
                    print("Stopping...")
                    controller.publish_velocity()
                    obs, info = env.reset()
                    break

                action, _states = model.predict(obs)
                obs, rewards, done, _, info = env.step(action)

if __name__ == '__main__':
    main()