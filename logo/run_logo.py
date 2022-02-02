import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.hopper_sparse import SparseHopperEnv
from envs.half_cheetah_sparse import SparseHalfCheetahEnv
from envs.walker_2d_sparse import SparseWalker2dEnv
from utils.delay_env import DelayRewardWrapper
from utils import *
from utils.delay_env import DelayRewardWrapper 
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from core.ppo import ppo_step
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent
from torch.utils.tensorboard import SummaryWriter
import datetime
from collections import deque

parser = argparse.ArgumentParser(description='LOGO')
parser.add_argument('--train-env-name', default=" ", metavar='G',
                    help='Training Env')
parser.add_argument('--env-name', default=" ", metavar='G',
                    help='Evaluation Env')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--demo-traj-path', metavar='G',
                    help='Demonstration trajectories')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
					help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
					help='damping (default: 1e-2)')
parser.add_argument('--render', action='store_true', default=False,
					help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
					help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
					help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
					help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
					help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
					help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
					help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
					help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
					help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
					help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=1500, metavar='N',
					help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')


parser.add_argument('--init-BC', action='store_true', default=False,
					help='Initialize with BC policy')
parser.add_argument('--env-num', type=int, default=-1, metavar='N',
					help='Env number')
parser.add_argument('--window', type=int, default=10, metavar='N',
					help='observation window')
parser.add_argument('--nn-param', nargs='+', type=int,default=[128,128])
parser.add_argument('--K-delta', type=int, default=-1, metavar='N',
					help='warmup samples befrore decay')
parser.add_argument('--delta-0', type=float, default=3e-2, metavar='G',
					help='max kl value (default: 1e-2)')
parser.add_argument('--delta', type=float, default=0.95, metavar='G',
					help='KL decay')
args = parser.parse_args()


nn_size = tuple(args.nn_param)
if (args.env_num == 1):
	args.env_name = 'Hopper-v2'
	args.demo_traj_path = 'exp_traj/Hopper-v2_rwd_1369_expert_traj_5.p'	
	args.K_delta = 50	
	args.sparse_val = 2.
	env_tag = 'Sparse2'
	args.delta_0 = 0.01
	args.low_kl = 5e-7
	args.delta = 0.95
	args.min_batch_size = 20000
	print("Sparse Hopper with sparsity: ",args.sparse_val)
	env = SparseHopperEnv(args.sparse_val)
	# args.seed = 11
	args.observe = 0
	if args.init_BC:
		args.model_path = 'Cloned_Policies/Hopper-v2_bc_policy_test_256_50.pt'
		args.delta_0 = 0.05
	eval_env = gym.make(args.env_name)


elif (args.env_num == 2):
	args.env_name = 'Hopper-v2'
	args.demo_traj_path = 'exp_traj/Hopper-v2_rwd_1369_expert_traj_5.p'
	args.K_delta = 50	
	args.sparse_val = 2.
	env_tag = 'Censored_Sparse2'
	args.delta_0 = 0.02
	args.low_kl = 5e-7
	args.delta = 0.95
	args.min_batch_size = 20000
	print("Sparse Hopper with sparsity: ",args.sparse_val)
	env = SparseHopperEnv(args.sparse_val)
	# args.seed = 1
	args.observe = 7	
	if args.init_BC:
		print('Cannot initialzie BC due to censored data, running with random initialization')
	eval_env = gym.make(args.env_name)

elif (args.env_num == 3):
	args.env_name = 'HalfCheetah-v2'
	args.demo_traj_path = 'exp_traj/HalfCheetah-v2_rwd_2658_traj_5.p'	
	args.K_delta = 50	
	args.sparse_val = 20.
	env_tag = 'Sparse20'
	args.delta_0 = 0.2
	args.low_kl = 5e-7
	args.delta = 0.95
	args.min_batch_size = 20000
	print("Sparse HalfCheetah with sparsity: ",args.sparse_val)
	env = SparseHalfCheetahEnv(args.sparse_val)
	args.max_iter_num = 2500
	# args.seed = 2
	args.observe = 0
	if args.init_BC:
		args.model_path = 'Cloned_Policies/HalfCheetah-v2_bc_policy_test_256_50.pt'
		args.delta_0 = 0.05
	eval_env = gym.make(args.env_name)



elif (args.env_num == 4):
	args.env_name = 'HalfCheetah-v2'
	args.demo_traj_path = 'exp_traj/HalfCheetah-v2_rwd_2658_traj_5.p'
	args.K_delta = 50	
	args.sparse_val = 20.
	env_tag = 'Censored_Sparse20'
	args.delta_0 = 0.1
	args.low_kl = 5e-7
	args.delta = 0.95
	args.min_batch_size = 20000
	args.max_iter_num = 2500
	print("Sparse HalfCheetah with sparsity: ",args.sparse_val)
	env = SparseHalfCheetahEnv(args.sparse_val)
	# args.seed = 1
	args.observe = 14	
	if args.init_BC:
		print('Cannot initialzie BC due to censored data, running with random initialization')
	eval_env = gym.make(args.env_name)




elif (args.env_num == 5):
	args.env_name = 'Walker2d-v2'
	args.demo_traj_path = 'exp_traj/Walker2d-v2_rwd_2448_expert_traj_5.p'		
	args.K_delta = 50	
	args.sparse_val = 2.
	env_tag = 'Sparse2'
	args.delta_0 = 0.03
	args.low_kl = 5e-7
	args.delta = 0.95
	args.min_batch_size = 20000
	print("Sparse Walker with sparsity: ",args.sparse_val)
	env = SparseWalker2dEnv(args.sparse_val)
	# args.seed = 1
	args.observe = 0
	if args.init_BC:
		args.model_path = 'Cloned_Policies/Walker2d-v2_bc_policy_test_256_50.pt'
		args.delta_0 = 0.05
	eval_env = gym.make(args.env_name)



elif (args.env_num == 6):
	args.env_name = 'Walker2d-v2'
	args.demo_traj_path = 'exp_traj/Walker2d-v2_rwd_2448_expert_traj_5.p'
	args.K_delta = 50	
	args.sparse_val = 2.
	env_tag = 'Censored_Sparse2'
	args.delta_0 = 0.05
	args.low_kl = 5e-7
	args.delta = 0.95
	args.min_batch_size = 20000
	print("Sparse Walker with sparsity: ",args.sparse_val)
	env = SparseWalker2dEnv(args.sparse_val)
	# args.seed = 1
	args.observe = 10	
	if args.init_BC:
		print('Cannot initialzie BC due to censored data, running with random initialization')
	eval_env = gym.make(args.env_name)

elif (args.env_num == 7):
	args.env_name = 'InvertedDoublePendulum-v2'
	args.demo_traj_path = 'exp_traj/InvertedDoublePendulum-v2_rwd_340_expert_traj_5.p'	
	args.K_delta = 5
	args.delay_val = 1000
	env_tag = 'Dealy1000'
	args.delta_0 = 0.2
	args.low_kl = 5e-7
	args.delta = 0.9
	args.min_batch_size = 5000
	args.max_iter_num = 40
	print("Delayed IDP with sparsity: ",args.delay_val)
	env = gym.make(args.env_name)
	env = DelayRewardWrapper(env, args.delay_val, 1000)
	# args.seed = 1
	args.observe = 0
	if args.init_BC:
		args.model_path = 'Cloned_Policies/InvertedDoublePendulum-v2_bc_policy_test_32_40.pt'
		args.delta_0 = 0.05
	eval_env = gym.make(args.env_name)

elif (args.env_num == 8):
	args.env_name = 'InvertedDoublePendulum-v2'
	args.demo_traj_path = 'exp_traj/InvertedDoublePendulum-v2_rwd_340_expert_traj_5.p'
	args.K_delta = 5
	args.delay_val = 1000
	env_tag = 'Censored_Dealy1000'
	args.delta_0 = 0.2
	args.low_kl = 5e-7
	args.delta = 0.9
	args.min_batch_size = 5000
	args.max_iter_num = 40
	print("Delayed IDP with sparsity: ",args.delay_val)
	env = gym.make(args.env_name)
	env = DelayRewardWrapper(env, args.delay_val, 1000)
	# args.seed = 1
	args.observe = 7	
	if args.init_BC:
		print('Cannot initialzie BC due to censored data, running with random initialization')
	eval_env = gym.make(args.env_name)	

else:
	print('Running on custom Env')
	env = gym.make(args.train_env_name)
	eval_env = gym.make(args.env_name)
	env_tag = args.train_env_name






	
##########################################################################

writer = SummaryWriter('Results/Env_{}_{}/LOGO_{}'
		.format(env_tag,args.env_name,datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

if args.K_delta > -1:
	print('Adaptive Decay')
	print('delta_0:',args.delta_0)
	print('Warmup Iterations:',args.K_delta)
	print('KL geometric decay value:',args.delta)
else:
	print('Constant Decay')
	print('delta_0:',args.delta_0)

##########################################################################

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(args.gpu_index)
	print('Using cuda device:',device)



if args.observe == 0:
	args.observe = env.observation_space.shape[0]


print('Observing the first ' + str(args.observe) + ' states')
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]

"""seeding"""

np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)
eval_env.seed(args.seed)


"""define actor and critic"""
if is_disc_action:
	policy_net = DiscretePolicy(state_dim, env.action_space.n)
else:
	policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std,hidden_size=nn_size)

value_net = Value(state_dim,hidden_size=nn_size)
value_net_exp = Value(state_dim,hidden_size=nn_size)
discrim_net = Discriminator(args.observe + action_dim)
discrim_criterion = nn.BCELoss()

if args.model_path is not None:
	print('Loading Model....',args.model_path)
	policy_net.load_state_dict(torch.load(args.model_path))
	

to_device(device, policy_net, value_net, value_net_exp,discrim_net, discrim_criterion)

optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

# load trajectory
expert_traj = pickle.load(open(args.demo_traj_path, "rb"))
action_indices = [i for i in range(state_dim,expert_traj.shape[1])]
state_indices = [i for i in range(args.observe)]
state_action_indices = state_indices + action_indices
partial_expert_traj = expert_traj[:,state_action_indices]
print('Demo trajectory samples: ', partial_expert_traj.shape[0])


def demo_reward(state, action):	
	partial_state = state[:,:args.observe]
	partial_state_action = tensor(np.hstack([partial_state, action]), dtype=dtype).to(device)
	with torch.no_grad():		
		return -torch.log(discrim_net(partial_state_action)).squeeze()


"""create agent"""
agent = Agent(env, policy_net, device, eval_env = eval_env,
			  num_threads=args.num_threads)


def update_params(batch, i_iter,kl):
	states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
	actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
	rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
	masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
	rewards_exp = demo_reward(np.stack(batch.state),np.stack(batch.action))
	with torch.no_grad():
		values = value_net(states)
		values_exp = value_net_exp(states)
		fixed_log_probs = policy_net.get_log_prob(states, actions)

	"""get advantage estimation from the trajectories"""
	advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

	advantages_exp, returns_exp = estimate_advantages(rewards_exp, masks, values_exp, args.gamma, args.tau, device)

	"""update discriminator"""
	for _ in range(1):
		expert_state_actions = torch.from_numpy(partial_expert_traj).to(dtype).to(device)		
		partial_states = states[:,:args.observe]
		g_o = discrim_net(torch.cat([partial_states, actions], 1))
		e_o = discrim_net(expert_state_actions)
		optimizer_discrim.zero_grad()
		discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
			discrim_criterion(e_o, zeros((partial_expert_traj.shape[0], 1), device=device))
		discrim_loss.backward()
		optimizer_discrim.step()

	
	trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)
	if (kl > 6e-7):
		trpo_step(policy_net, value_net_exp, states, actions, returns_exp, advantages_exp, kl, args.damping, args.l2_reg,fixed_log_probs = fixed_log_probs)



writer.add_text('Evaluation env name',str(args.env_name))
writer.add_text('Train env name',str(env_tag))
writer.add_text('Demonstration trajectories path', str(args.demo_traj_path))
writer.add_text('K_delta', str(args.K_delta))
writer.add_text('delta_0', str(args.delta_0))
writer.add_text('delta', str(args.delta))
writer.add_text('Seed', str(args.seed))
writer.add_text('Observable state',str(args.observe))
writer.add_text('Expert trajectory samples', str(partial_expert_traj.shape))
if args.model_path is not None:
	writer.add_text('Model Path', str(args.model_path))

def main_loop():
	kl = args.delta_0
	prev_rwd = deque(maxlen = args.window)
	prev_rwd.append(0)


	for i_iter in range(args.max_iter_num):
		"""generate multiple trajectories that reach the minimum batch_size"""
		discrim_net.to(torch.device('cpu'))
		batch, log = agent.collect_samples(args.min_batch_size, render=args.render)
		discrim_net.to(device)

		if (args.K_delta > -1):
			if (i_iter > args.K_delta):
				avg_prev_rwd = np.mean(prev_rwd)						
				if (avg_prev_rwd < log['avg_reward']):					
					kl = max(args.low_kl,kl*args.delta)

		writer.add_scalar('KL',kl,i_iter+1)
		prev_rwd.append(log['avg_reward'])
		t0 = time.time()
		update_params(batch, i_iter,kl)
		t1 = time.time()
		"""evaluate with determinstic action (remove noise for exploration)"""
		discrim_net.to(torch.device('cpu'))
		_, log_eval = agent.collect_samples(args.eval_batch_size, eval_flag = True,mean_action=True)
		discrim_net.to(device)
		t2 = time.time()

		if i_iter % args.log_interval == 0:
			print('{}\tT_sample {:.4f}\tT_update {:.4f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}\t KL{:}'
				.format(i_iter, log['sample_time'], t1-t0,log['avg_reward'],log_eval['avg_reward'],kl))

		writer.add_scalar('rewards/train_R_avg',log['avg_reward'],i_iter+1)
		writer.add_scalar('rewards/eval_R_avg',log_eval['avg_reward'],i_iter+1)


		"""clean up gpu memory"""
		torch.cuda.empty_cache()


main_loop()
