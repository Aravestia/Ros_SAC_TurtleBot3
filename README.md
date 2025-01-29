Path planning algorithm for Turtlebot3 using OpenAI Gymnasium & ROS Noetic
---
# How It Works
It makes use of A* Search + SAC Reinforcement Learning Algorithm
- Defaults to path planned out by A* Search
- Activates RL Algorithm when close to obstacle not on global map
- Transitions back to A* Search when obstacle is successfully bypassed
