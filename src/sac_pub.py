#!/usr/bin/env python3

import rospy

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

import numpy as np
import pandas as pd

from sac_env_v3 import SacEnvV3

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
        init_positions = [[-0.5, 0.75], [-0.5, -1]]

    return np.array(init_positions)

def init_map(stage):
    map = ""

    if stage == 2:
        map = r"/home/aravestia/isim/noetic/src/robot_planner/src/map/map_stage2.pgm"

    if stage == 5: # Turtlebot_world
        map = r"/home/aravestia/isim/noetic/src/robot_planner/src/map/map_turtlebot_world.pgm"

    return map

def main(args=None):
    rospy.init_node('sac_pub', anonymous=True)

    # Depends on map
    amr_model = 'turtlebot3_burger'
    model_pth = r"/home/aravestia/isim/noetic/src/robot_planner/src/models/sac_model_v3.0.pth"
    stage = 5
    stage_positions = init_stage_positions(stage)
    stage_map = init_map(stage)

    env = SacEnvV3(
    amr_model=amr_model,
    epoch=1,
    init_positions=stage_positions,
    stage_map=stage_map,
    test_mode=True
    )

    model = SAC.load(path=model_pth, env=env)
    
    while not rospy.is_shutdown():
        obs, info = env.reset()
        done = False

        while not done:
            if done:
                print("Stopping...")
                obs, info = env.reset()
                break

            action, _states = model.predict(obs)
            obs, rewards, done, _, info = env.step(action)

if __name__ == '__main__':
    main()