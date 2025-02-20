import numpy as np

def init_stage_positions(stage, epoch=0):
    init_positions = [] # [spawn, goal]
    init_choices = []

    if stage == 1:
        init_positions = [[1, 1], [-1.25, -1.25]]
    elif stage == 2:
        init_positions = [[1.25, 1.25], [0, 0]]   
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
    elif stage == 'turtlebot_world_test':
        init_positions = [[-0.5, -1], [-0.5, 0.75]]
    else:
        return None

    return np.array(init_positions)

def init_map(stage):
    map = ""

    if stage == 2:
        map = r"/home/aravestia/isim/noetic/src/robot_planner/src/map/map_stage2.pgm"
    elif stage == 'turtlebot_world_train':
        map = r"/home/aravestia/isim/noetic/src/robot_planner/src/map/map_turtlebot_world.pgm"
    elif stage == 'turtlebot_world_test':
        map = r"/home/aravestia/isim/noetic/src/robot_planner/src/map/map_turtlebot_world.pgm"
    else:
        return None

    return map