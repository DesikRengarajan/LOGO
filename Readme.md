# **Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration**
Code for _Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration, ICLR 2022 (Spotlight)_

[Video of TurtleBot Demonstration](https://www.youtube.com/watch?v=6WKfggS5gSM)

This codebase is based on a publicly available github repository [Khrylx/PyTorch-RL](https://github.com/Khrylx/PyTorch-RL)

To run experiments, you will need to install the following packages preferably in a conda virtual environment 
- gym 0.18.0
- pytorch 1.8.1
- mujoco-py 2.0.2.13
- tesnsorboard 2.5.0

The python file to run LOGO is present in [logo/run_logo.py](https://github.com/DesikRengarajan/LOGO/blob/main/logo/run_logo.py)

To run the code with the default parameters, simply execute the following command
```
python run_logo.py --env-num i
```
Where _i_ is an integer between 1-8 corresponding to the following experiments
1. Hopper-v2
2. Censored Hopper-v2
3. HalfCheetah-v2
4. Censored HalfCheetah-v2
5. Walker2d-v2
6. Censored Walker2d-v2
7. InvertedDoublePendulum-v2
8. Censored InvertedDoublePendulum-v2

The tensorboard logs will be saved in a folder titled 'Results'

For the full observation setting, we can initialize the policy network using behavior cloning, this enables faster learning, to do so simply execute the following command 
```
python run_logo.py --env-num i --init-BC
```
