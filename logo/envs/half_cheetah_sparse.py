import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SparseHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

	def __init__(self,sparse_val = 20.):
		self.sparse_val = sparse_val
		print('Sparse val', self.sparse_val)
		mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
		utils.EzPickle.__init__(self)

	def step(self, action):
		xposbefore = self.sim.data.qpos[0]
		self.do_simulation(action, self.frame_skip)
		xposafter = self.sim.data.qpos[0]
		ob = self._get_obs()

		# --------- Dense Reward ---------
		# reward_ctrl = - 0.1 * np.square(action).sum()
		# reward_run = (xposafter - xposbefore) / self.dt
		# reward = reward_ctrl + reward_run

		# --------- Sparse Reward ---------
		# a reward +1 is given for every time the agent moves forward over a specific number of units.
		if xposafter - self.init_qpos[0] > self.sparse_val:
			reward = 1.
		else:
			reward = 0.
		done = False

		return ob, reward, done, {}


	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat[1:],
			self.sim.data.qvel.flat,
		])

	def reset_model(self):
		qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
		qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
		self.set_state(qpos, qvel)
		return self._get_obs()

	def viewer_setup(self):
		self.viewer.cam.distance = self.model.stat.extent * 0.5