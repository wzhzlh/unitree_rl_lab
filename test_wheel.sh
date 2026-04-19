#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate isaaclab
cd /home/zhangjiayi/unitree_rl_lab
source ~/.bashrc
./unitree_rl_lab.sh -t --task Unitree-WheelDog-Velocity --headless --num_envs 2 --max_iterations 1
