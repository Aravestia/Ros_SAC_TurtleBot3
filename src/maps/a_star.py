import math
import heapq
import cv2
import numpy as np
from scipy.ndimage import binary_dilation

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

def img_to_grid(filename, margin):
    grid = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    grid = np.where(grid < 250, 0, 1)

    grid = 1 - binary_dilation(
        grid == 0,
        structure=np.ones((2 * margin + 1, 2 * margin + 1), dtype=bool)
    ).astype(int)

    return grid

def a_star_search(grid, src, dest, grid_row, grid_col, grid_resolution, grid_x_offset, grid_y_offset, occurrence):
    waypoints = []

    def is_valid(row, col):
        return (row >= 0) and (row < grid_row) and (col >= 0) and (col < grid_col)
    
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
            waypoints.append([
                (i[1] - grid_x_offset) * grid_resolution,
                (grid_y_offset - i[0]) * grid_resolution
            ])

    def create_waypoints(waypoints):
        w = {}
        c = 0
        d = 0
        for waypoint in waypoints:
            if c != 0 and (c == (len(waypoints) - 1) or c % occurrence == 0):
                w[d] = waypoint
                d += 1
            c += 1

        return w

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
    closed_list = [[False for _ in range(grid_col)] for _ in range(grid_row)]
    # Initialize the details of each cell
    cell_details = [[Cell() for _ in range(grid_col)] for _ in range(grid_col)]

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
                    return create_waypoints(waypoints)
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

    return None

def _get_closest_waypoint(current_position, waypoints, current_closest_waypoint, lookahead, lookahead_max):
    min_distance = 10000
    closest_waypoint = current_closest_waypoint
    range_max = min(current_closest_waypoint + lookahead, lookahead_max)
    range_min = current_closest_waypoint

    for i in range(range_min, range_max):
        m = np.linalg.norm(current_position - waypoints[i])
        if m < min_distance:
            min_distance = m
            closest_waypoint = i

    return closest_waypoint, min_distance

def point_to_grid(grid_x_offset, grid_y_offset, point_x, point_y):
    return np.array([
            grid_y_offset - (point_y / 0.05),
            (point_x / 0.05) + grid_x_offset
        ]).astype(int)