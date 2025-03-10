import numpy as np
import os

def init_stage_positions(stage, epoch=0):
    init_positions = [] # [spawn, goal]
    init_choices = []

    if stage == 1:
        init_positions = [[1, 1], [-1, -1]]
    elif stage == 2:
        init_positions = [[1.25, 0.5], [0, 0]]   
    elif stage == 3:
        init_positions = [[1, 1], [-1.25, -1.25]]
    elif stage == 'house':
        init_positions = [[-1.5, 2], [2, -2]]    
    elif stage == 'turtlebot_world_train':
        init_choices = np.array([
            [[-0.5, 0.75], [-0.5, -1]], 
            [[-0.5, -0.75], [-0.5, 0.75]],
            [[-0.6, 0.75], [0.6, 0.75]],
            [[0.6, 0.75], [-0.6, 0.75]]
        ])

        init_positions = init_choices[epoch % len(init_choices)]
    elif stage == 'turtlebot_world':
        init_positions = [[-0.5, -1], [-0.5, 0.75]]
    elif stage == 'local_minimum':
        init_positions = [[-0.5, 0.5], [1.75, 0]]
    elif stage == 'local_minimum_train':
        init_positions = [[1.75, 1.6], [1.75, 0]]
    else:
        return None

    return np.array(init_positions)

def init_map(stage):
    maps = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps")

    if stage == 2:
        map = "map_stage2.pgm"
    elif stage == 'turtlebot_world_train':
        map = "map_turtlebot_world.pgm"
    elif stage == 'turtlebot_world':
        map = "map_turtlebot_world.pgm"
    elif stage == 'local_minimum':
        map = "map_local_minimum.pgm"
    elif stage == 'local_minimum_train':
        map = "map_local_minimum.pgm"
    else:
        return None

    return os.path.join(maps, map)
