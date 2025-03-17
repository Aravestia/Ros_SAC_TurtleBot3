import numpy as np
import os

def init_stage_positions(stage, epoch=0):
    init_positions = [] # [spawn, goal]
    init_choices = []

    if stage == "stage_1":
        init_positions = [[-1, 0], [1, 0]]
    elif stage == "stage_2":
        init_positions = [[1.25, 0.5], [0, 0]]   
    elif stage == "stage_3":
        init_positions = [[1, 1], [-1.25, -1.25]]
    elif stage == 'house':
        init_positions = [[-1.5, 2], [2, -2]]    
    elif stage == 'world_train':
        init_choices = np.array([
            [[-0.5, 0.75], [-0.5, -1]], 
            [[-0.5, -0.75], [-0.5, 0.75]],
            [[-0.6, 0.75], [0.6, 0.75]],
            [[0.6, 0.75], [-0.6, 0.75]]
        ])

        init_positions = init_choices[epoch % len(init_choices)]
    elif stage == 'world':
        init_positions = [[-0.5, -1], [-0.5, 0.75]]
    elif stage == 'local_minimum':
        init_positions = [[-0.5, 0.5], [1.75, 0]]
    elif stage == 'local_minimum_train':
        init_positions = [[1.75, 1.6], [1.75, 0]]
    elif stage == 'local_minimum_ideal_path':
        init_positions = [[-0.5, 0.5], [1.75, 0]]
    elif stage == "stage_1_ideal_path":
        init_positions = [[-1, 0], [1, 0]]
    else:
        return None

    return np.array(init_positions)

def init_map(stage):
    maps = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps")

    if stage == "stage_1":
        map = "map_stage_1.pgm"
    elif stage == "stage_2":
        map = "map_stage_2.pgm"
    elif stage == 'world_train':
        map = "map_world.pgm"
    elif stage == 'world':
        map = "map_world.pgm"
    elif stage == 'local_minimum':
        map = "map_local_minimum.pgm"
    elif stage == 'local_minimum_train':
        map = "map_local_minimum.pgm"
    elif stage == 'local_minimum_ideal_path':
        map = "map_local_minimum_ideal_path.pgm"
    elif stage == 'stage_1_ideal_path':
        map = "map_stage_1_ideal_path.pgm"
    else:
        return None

    return os.path.join(maps, map)
