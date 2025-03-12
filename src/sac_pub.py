#!/usr/bin/env python3

import rospy

from stable_baselines3 import SAC
#from stable_baselines3.common.env_checker import check_env

import time
import os
import json
import importlib

from initialisation.init_obstacle import Obstacle
from initialisation.init_custom_stage import CustomStage
from initialisation import init_stage

def main(args=None):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sac_pub_options"), 'r') as file:
        options = json.load(file)
    
    epochs = 10000
    timesteps = 3000

    rospy.init_node('sac_pub', anonymous=True)

    test_mode = options["test_mode"]
    stage_custom = options["stage_custom"]
    stage_custom_id = options["stage_custom_id"]
    stage_obstacle = options["stage_obstacle"]
    stage_obstacle_id = options["stage_obstacle_id"]
    amr_model = options["amr_model"]
    env_version = options["env_version"]
    model_version = options["model_version"]
    stage_test_mode = options["stage_test_mode"]
    stage_train_mode = options["stage_train_mode"]
    
    if env_version == 1:
        from sac_env_v1_basic import SacEnv
    elif env_version == 2:
        from sac_env_v2_a_star import SacEnv
    elif env_version == 3:
        from sac_env_v3_SAS import SacEnv
    else:
        print("environment is invalid.")
        return

    model_pth = os.path.dirname(os.path.abspath(__file__))
    model_pth = os.path.join(model_pth, "models", f"v{env_version}", f"sac_model_v{model_version}.pth")

    stage = stage_test_mode if test_mode else stage_train_mode
    stage_positions = init_stage.init_stage_positions(stage)
    stage_map = init_stage.init_map(stage)

    if stage_custom:
        CustomStage(stage_custom_id)

    if stage_obstacle:
        Obstacle(stage_obstacle_id)

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

            model_pth = os.path.dirname(os.path.abspath(__file__))
            model_pth = os.path.join(model_pth, "models", f"v{env_version}", f"sac_model_v{model_version}.pth")

            print(f"model path exists: {os.path.exists(model_pth)}")
            model = SAC.load(path=model_pth, env=env) if os.path.exists(model_pth) else SAC('MlpPolicy', env, ent_coef='auto', verbose=1)
            model.learn(total_timesteps=timesteps)
            model.save(model_pth)

            print(f"model saved! Epoch: {i + 1}")

            env.reset()
            time.sleep(3)

if __name__ == '__main__':
    main()