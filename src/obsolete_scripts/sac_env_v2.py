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

import heapq
import cv2
from scipy.ndimage import binary_dilation

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

from create_sdf import goal_sdf

class SacEnvV2(gym.Env):
    def __init__(self, amr_model='turtlebot3_burger', epoch=0, init_positions=[], stage_map="", yaw=0.0):
        super(SacEnvV2, self).__init__()

        self.velocity_multiplier = 0.15
        self.angular_velocity_multiplier = 2.84
        self.velocity = self.velocity_multiplier
        self.angular_velocity = 0.0

        self.laserscan_maxcap = 3.5
        self.laserscan_mincap = 0.12
        self.laserscan_warning_threshold = 0.3
        self.laserscan_closest = self.laserscan_maxcap
        self.laserscan = np.full(12, self.laserscan_maxcap)

        self.angle_cap = 2 * math.pi

        self.init_positions = init_positions
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
        self.max_step_count = 4000
        self.stagnant_count = 0
        self.max_stagnant_count = 10
        self.reset_count = 0
        self.observation_state = []

        self.amr_model = amr_model
        self.epoch = epoch

        self.goal_radius = 0.2

        self.grid_row = 384
        self.grid_col = 384
        self.grid = self.img_to_grid(stage_map, 5)
        self.grid_in = copy.deepcopy(self.grid)
        self.grid_x_offset = 200
        self.grid_y_offset = 184
        self.grid_resolution = 0.05
        self.grid_spawn = []
        self.grid_goal = []
        self.waypoints = []
        self.waypoint_radius = 0.12
        self.waypoint_occurrence = 2
        self.current_waypoint = 0
        self.current_waypoint_position = []

        self.moving_obstacle_radius = 0.15

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
            low=np.array([-1.0]), 
            high=np.array([1.0]),
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

    def publish_velocity(self, angular_velocity, laserscan_closest=3.5, goal_distance=1):
        self.velocity = self.velocity_multiplier if (
            laserscan_closest > self.laserscan_warning_threshold
        ) else (0.5 * self.velocity_multiplier)

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

        waypoint_position = self.current_waypoint_position
        waypoint_distance = np.linalg.norm(waypoint_position - current_position)

        yaw = self.yaw
        goal_angle = self.normalise_radians_angle(
            math.atan2(
                goal_position[1] - current_position[1], 
                goal_position[0] - current_position[0]
            ) - yaw
        )
        waypoint_angle = self.normalise_radians_angle(
            math.atan2(
                waypoint_position[1] - current_position[1], 
                waypoint_position[0] - current_position[0]
            ) - yaw
        )

        laserscan_closest = self.laserscan_closest
        laserscan = self.laserscan

        return np.nan_to_num(
            np.append(
                np.array([
                    waypoint_distance,
                    waypoint_angle,
                    laserscan_closest,
                ]),
                laserscan
            )
        ).astype(np.float32)
    
    class Cell:
        def __init__(self):
        # Parent cell's row index
            self.parent_i = 0
        # Parent cell's column index
            self.parent_j = 0
        # Total cost of the cell (g + h)
            self.f = float('inf')
        # Cost from start to this cell
            self.g = float('inf')
        # Heuristic cost from this cell to destination
            self.h = 0

    def img_to_grid(self, filename, margin):
        grid = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        grid = np.where(grid < 250, 0, 1)

        grid = 1 - binary_dilation(
            grid == 0,
            structure=np.ones((2 * margin + 1, 2 * margin + 1), dtype=bool)
        ).astype(int)

        return grid

    def create_waypoints(self, waypoints, occurrence):
        w = {}
        c = 0
        d = 0
        for waypoint in waypoints:
            if c != 0 and (c == (len(waypoints) - 1) or c % occurrence == 0):
                w[d] = np.round(waypoint, decimals=2)
                d += 1
            c += 1
        
        return w

    def a_star_search(self, grid, src, dest):
        def is_valid(row, col):
            return (row >= 0) and (row < self.grid_row) and (col >= 0) and (col < self.grid_col)
        
        def is_unblocked(grid, row, col):
            return grid[row][col] == 1
        
        def is_destination(row, col, dest):
            return row == dest[0] and col == dest[1]

        def calculate_h_value(row, col, dest):
            return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5
        
        def trace_path(cell_details, dest):
            path = []
            row = dest[0]
            col = dest[1]

            # Trace the path from destination to source using parent cells
            while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
                path.append((row, col))
                temp_row = cell_details[row][col].parent_i
                temp_col = cell_details[row][col].parent_j
                row = temp_row
                col = temp_col

            # Add the source cell to the path
            path.append((row, col))
            # Reverse the path to get the path from source to destination
            path.reverse()

            # Print the path
            for i in path:
                self.waypoints.append([
                    (i[1] - self.grid_x_offset) * self.grid_resolution,
                    (self.grid_y_offset - i[0]) * self.grid_resolution
                ])

        # Check if the source and destination are valid
        if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
            print("Source or destination is invalid")
            return

        # Check if the source and destination are unblocked
        if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
            print("Source or the destination is blocked")
            return

        # Check if we are already at the destination
        if is_destination(src[0], src[1], dest):
            print("We are already at the destination")
            return

        # Initialize the closed list (visited cells)
        closed_list = [[False for _ in range(self.grid_col)] for _ in range(self.grid_row)]
        # Initialize the details of each cell
        cell_details = [[self.Cell() for _ in range(self.grid_col)] for _ in range(self.grid_col)]

        # Initialize the start cell details
        i = src[0]
        j = src[1]
        cell_details[i][j].f = 0
        cell_details[i][j].g = 0
        cell_details[i][j].h = 0
        cell_details[i][j].parent_i = i
        cell_details[i][j].parent_j = j

        # Initialize the open list (cells to be visited) with the start cell
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))

        # Initialize the flag for whether destination is found
        found_dest = False

        # Main loop of A* search algorithm
        while len(open_list) > 0:
            # Pop the cell with the smallest f value from the open list
            p = heapq.heappop(open_list)

            # Mark the cell as visited
            i = p[1]
            j = p[2]
            closed_list[i][j] = True

            # For each direction, check the successors
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dir in directions:
                new_i = i + dir[0]
                new_j = j + dir[1]

                # If the successor is valid, unblocked, and not visited
                if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                    # If the successor is the destination
                    if is_destination(new_i, new_j, dest):
                        # Set the parent of the destination cell
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j
                        print("The destination cell is found")
                        # Trace and print the path from source to destination
                        trace_path(cell_details, dest)
                        found_dest = True
                        return
                    else:
                        # Calculate the new f, g, and h values
                        g_new = cell_details[i][j].g + 1.0
                        h_new = calculate_h_value(new_i, new_j, dest)
                        f_new = g_new + h_new

                        # If the cell is not in the open list or the new f value is smaller
                        if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                            # Add the cell to the open list
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            # Update the cell details
                            cell_details[new_i][new_j].f = f_new
                            cell_details[new_i][new_j].g = g_new
                            cell_details[new_i][new_j].h = h_new
                            cell_details[new_i][new_j].parent_i = i
                            cell_details[new_i][new_j].parent_j = j

        # If the destination is not found after visiting all cells
        if not found_dest:
            print("Failed to find the destination cell")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.reset_count += 1
        self.publish_velocity(0)

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

        self.waypoints = []
        self.grid_in = copy.deepcopy(self.grid)
        self.grid_spawn = np.array([
            self.grid_y_offset - (self.spawn_position[1] / 0.05),
            (self.spawn_position[0] / 0.05) + self.grid_x_offset
        ]).astype(int)
        self.grid_goal = np.array([
            self.grid_y_offset - (self.goal_position[1] / 0.05),
            (self.goal_position[0] / 0.05) + self.grid_x_offset
        ]).astype(int)
        self.a_star_search(self.grid_in, self.grid_spawn, self.grid_goal)
        self.waypoints = self.create_waypoints(self.waypoints, self.waypoint_occurrence)
        self.current_waypoint = 0
        self.current_waypoint_position = self.waypoints[self.current_waypoint]

        self.reset_turtlebot3_gazebo()
        self.reset_goal()

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
            laserscan_closest = self.observation_state[2],
            goal_distance = self.observation_state[0] * self.goal_distance_from_spawn
        )

        rospy.sleep(0.02)

        reward = self._compute_reward()

        print("------------------------------------------")
        print("OBSERVATION SPACE")
        print(f"waypoint_distance: {self.observation_state[0]}")
        print(f"waypoint_angle: {self.observation_state[1]}")
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
        print(f"current position: {np.round(self.position, 2)}")
        print(f"current waypoint position: {self.current_waypoint_position}")
        print(f"current waypoint: {self.current_waypoint}")
        print(f"total waypoints: {len(self.waypoints)}")
        print("------------------------------------------")
        print(" ")

        return self.observation_state, reward, self.done, self.truncated, {}

    def _compute_reward(self):
        waypoints = self.waypoints
        closest_waypoint = self.current_waypoint
        waypoint_distance = self.observation_state[0]

        goal_angle = self.observation_state[1]
        laserscan_closest = self.observation_state[2]
        step_count = self.step_count
        collision_threshold = self.laserscan_mincap + 0.01
        warning_threshold = self.laserscan_warning_threshold

        waypoint_radius = self.waypoint_radius

        reward_goal = 50 + (10 * (self.max_step_count - step_count) / self.max_step_count)
        reward_waypoint = 4

        penalty_distance_from_waypoint = -min(waypoint_distance - self.waypoint_radius, 2)
        penalty_not_facing_waypoint = 0 if (abs(goal_angle) < math.pi/4) else -0.2
        penalty_obstacle_proximity = 0 if (
            laserscan_closest >= warning_threshold
        ) else (
            -22 * ((warning_threshold - laserscan_closest) / (warning_threshold - collision_threshold)) - 3
        )
        penalty_collision = -50
        penalty_step_count = -(1 / self.max_step_count) * (step_count - 1)
        penalty_step_count_maxed = -50

        if self.step_count >= self.max_step_count:
           reward = penalty_distance_from_waypoint + penalty_step_count_maxed
           self.end_episode()
        else:
            if laserscan_closest < collision_threshold:
                reward = penalty_distance_from_waypoint + penalty_collision + penalty_obstacle_proximity + penalty_step_count + penalty_not_facing_waypoint
                self.end_episode()
                print(f"!!!!!ROBOT COLLISION!!!!! scan: {laserscan_closest}")
            else:
                waypoint_min_distance = 1000
                for i in range(self.current_waypoint, len(waypoints)):
                    distance = np.linalg.norm(self.position - waypoints[i])
                    if distance < waypoint_min_distance:
                        waypoint_min_distance = distance
                        closest_waypoint = i

                self.current_waypoint = closest_waypoint
                self.current_waypoint_position = waypoints[closest_waypoint]

                if laserscan_closest < warning_threshold:
                    reward = penalty_obstacle_proximity + penalty_step_count
                else:
                    reward = penalty_distance_from_waypoint + penalty_obstacle_proximity + penalty_step_count + penalty_not_facing_waypoint

                    if waypoint_distance < waypoint_radius:
                        if closest_waypoint == (len(waypoints) - 1):
                            reward += reward_goal

                            self.end_episode()
                            print(f"!!!!!ROBOT GOAL REACHED!!!!!")
                        else:
                            reward += reward_waypoint
                            self.current_waypoint = closest_waypoint + 1
                            self.current_waypoint_position = waypoints[self.current_waypoint]

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

        model_names = self.get_world_properties().model_names

        for model in model_names:
            if model.startswith(model_name):
                self.delete_model(model)

        self.spawn_model(model_state_msg.model_name, self.goal_sdf, "", model_state_msg.pose, "world")
        print(f"Goal set. {self.goal_position}")

    def reset_turtlebot3_gazebo(self):
        yaw = self.yaw
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

        return angle

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

    if stage == 6: # Turtlebot_world
        init_positions = [[-0.5, 0.5], [1.75, -1.75]]

    return np.array(init_positions)

def init_map(stage):
    map = ""

    if stage == 2:
        map = r"/home/aravestia/isim/noetic/src/robot_planner/src/map/map_stage2.pgm"

    if stage == 5: # Turtlebot_world
        map = r"/home/aravestia/isim/noetic/src/robot_planner/src/map/map_turtlebot_world.pgm"

    if stage == 6: # Turtlebot_world
        map = r"/home/aravestia/isim/noetic/src/robot_planner/src/map/map_local_minimum.pgm"

    return map

def main(args=None):
    epochs = 1000
    timesteps = 100000

    for i in range(epochs):
        rospy.init_node('sac_env_v2', anonymous=True)

        # Depends on map
        amr_model = 'turtlebot3_burger'
        model_pth = r"/home/aravestia/isim/noetic/src/robot_planner/src/models/sac_model_v2.1-demo.pth"
        stage = 6
        stage_positions = init_stage_positions(stage)
        stage_map = init_map(stage)

        env = SacEnvV2(
            amr_model=amr_model,
            epoch=(i + 1),
            init_positions=stage_positions,
            stage_map=stage_map
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

