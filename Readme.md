Code for Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration

The code is based on a publicly available github repository https://github.com/Khrylx/PyTorch-RL which uses the MIT Licence

To reproduce the results, you will need to install the following packages preferably in a conda virtual environment 
	- gym 0.18.0
	- pytorch 1.8.1
	- mujoco-py 2.0.2.13
	- tesnsorboard 2.5.0

The python file to run the code is present in the folder titled 'logo'

To run the code with the default parameters, simply execute the following command
	python run_logo.py --env-num i

Where 'i' is an integer between 1-8 corresponding to the following experiments
1: Hopper-v2
2: Censored Hopper-v2
3: HalfCheetah-v2
4: Censored HalfCheetah-v2
5: Walker2d-v2
6: Censored Walker2d-v2
7: InvertedDoublePendulum-v2
8: Censored InvertedDoublePendulum-v2

The tensorboard logs will be saved in a folder titled 'Results'
