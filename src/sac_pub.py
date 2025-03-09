#!/usr/bin/env python3

import obstacle.custom_stage
import obstacle.obstacle_4
import obstacle.obstacle_5
import rospy

from stable_baselines3 import SAC
#from stable_baselines3.common.env_checker import check_env

import time
import os

from sac_env_v3 import SacEnvV3
from sac_env_v3_5 import SacEnvV3_5
from sac_env_v4 import SacEnvV4
import init_stage

def main(args=None):
    epochs = 10000
    timesteps = 4000

    rospy.init_node('sac_pub', anonymous=True)

    test_mode = True

    amr_model = 'turtlebot3_burger'
    model_pth = r"/home/aravestia/isim/noetic/src/robot_planner/src/models/sac_model_v3.0.pth"

    stage = 'local_minimum' if test_mode else 'local_minimum_train'
    stage_positions = init_stage.init_stage_positions(stage)
    stage_map = init_stage.init_map(stage)
    stage_map_custom = obstacle.custom_stage.CustomStage()
    stage_obstacle = obstacle.obstacle_5.Obstacle()

    if test_mode:
        env = SacEnvV3(
            amr_model=amr_model,
            epoch=1,
            init_positions=stage_positions,
            stage_map=stage_map,
            test_mode=test_mode
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
    else:
        for i in range(epochs):
            stage_positions = init_stage.init_stage_positions(stage, i)

            env = SacEnvV3(
                amr_model=amr_model,
                epoch=(i + 1),
                init_positions=stage_positions,
                stage_map=stage_map,
                max_timesteps=timesteps,
                test_mode=test_mode
            )

            #check_env(env)

            print(f"model path exists: {os.path.exists(model_pth)}")
            model = SAC.load(path=model_pth, env=env) if os.path.exists(model_pth) else SAC('MlpPolicy', env, ent_coef='auto', verbose=1)
            model.learn(total_timesteps=timesteps)
            model.save(model_pth)

            print(f"model saved! Epoch: {i + 1}")

            env.reset()
            time.sleep(3)

if __name__ == '__main__':
    main()