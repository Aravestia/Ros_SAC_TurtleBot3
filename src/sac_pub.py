#!/usr/bin/env python3

import rospy

from stable_baselines3 import SAC
#from stable_baselines3.common.env_checker import check_env

import time
import os

from sac_env_v1_basic import SacEnv
#from sac_env_v2_a_star import SacEnv
#from sac_env_v3_SAS import SacEnv

from initialisation.obstacle.obstacle_5 import Obstacle
from initialisation.custom_stage.custom_stage_1 import CustomStage
from initialisation import init_stage

def main(args=None):
    epochs = 10000
    timesteps = 3000

    rospy.init_node('sac_pub', anonymous=True)

    test_mode = False
    stage_custom = False
    stage_obstacle = False
    amr_model = 'turtlebot3_burger'
    env_version = 1
    model_version = "1.3"

    model_pth = os.path.dirname(os.path.abspath(__file__))
    model_pth = os.path.join(model_pth, "models", f"v{env_version}", f"sac_model_v{model_version}.pth")

    if env_version == 1:
        stage = 'local_minimum' if test_mode else 1
    elif env_version == 2:
        stage = 'turtlebot_world' if test_mode else 'turtlebot_world_train'
    elif env_version == 3:
        stage = 'local_minimum' if test_mode else 'local_minimum_train'
    else:
        stage = None

    if stage is not None:
        stage_positions = init_stage.init_stage_positions(stage)
        stage_map = init_stage.init_map(stage)

        if stage_custom:
            CustomStage()
        if stage_obstacle:
            Obstacle()

        if test_mode:
            env = SacEnv(
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

                env = SacEnv(
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
    else:
        print("version does not exist.")

if __name__ == '__main__':
    main()