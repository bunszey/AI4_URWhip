###             FILE FROM                  ###
#https://github.com/wangjunyi9999/OIAC_whips/#
##############################################
import numpy as np
import torch
from typing import Optional


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		super(ReplayBuffer, self).__init__()
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim), dtype=np.float32)
		self.action = np.zeros((max_size, action_dim), dtype=np.float32)
		self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
		self.reward = np.zeros((max_size, 1), dtype=np.float32)
		self.not_done = np.zeros((max_size, 1), dtype=np.float32)

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		if batch_size > self.size:
			raise ValueError(f"The size of replay buffer must be larger than batch size!!")
		sampled_indices = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.as_tensor(self.state[sampled_indices], device=self.device, dtype=torch.float32),
			torch.as_tensor(self.action[sampled_indices], device=self.device, dtype=torch.float32),
			torch.as_tensor(self.next_state[sampled_indices], device=self.device, dtype=torch.float32),
			torch.as_tensor(self.reward[sampled_indices], device=self.device, dtype=torch.float32),
			torch.as_tensor(self.not_done[sampled_indices], device=self.device, dtype=torch.float32),
		)

	def save(self, path):
		np.savez(path, state=self.state, action=self.action, next_state=self.next_state, reward=self.reward, not_done=self.not_done)

	def load(self, path):
		data = np.load(path)
		self.state = data['state']
		self.action = data['action']
		self.next_state = data['next_state']
		self.reward = data['reward']
		self.not_done = data['not_done']
		self.size = self.state.shape[0]

class VPGBuffer(object):
	"""
	A buffer for storing trajectories experienced by a VPG or PPO agent interacting with the environment, 
	and using Generalized Advantage Estimation (GAE-Lambda)
	for calculating the advantages of state-action pairs.
	"""
	def __init__(self, state_dim, action_dim, max_size, gamma=0.99, lam=0.95, is_discrete=False):
		super(VPGBuffer, self).__init__()
		self.gamma = gamma
		self.lam = lam
		self.max_size = max_size
		self.ptr = 0
		self.path_start_idx = 0

		self.state_buf = np.zeros((max_size, state_dim), dtype=np.float32)
		self.action_buf = np.zeros((max_size, action_dim), dtype=np.float32) if not is_discrete else \
			np.zeros(max_size, dtype=np.float32)
		self.reward_buf = np.zeros(max_size, dtype=np.float32)
		self.v_buf = np.zeros(max_size, dtype=np.float32)
		self.logp_buf = np.zeros(max_size, dtype=np.float32)
		self.advantage_buf = np.zeros(max_size, dtype=np.float32)
		self.reward_to_go_buf = np.zeros(max_size, dtype=np.float32)
		self.full = False

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, reward, v, logp_pi):
		"""
		Add one timestep of agent-environment interaction to the buffer.
		"""
		assert self.ptr < self.max_size  # buffer has to have room so you can store
		self.state_buf[self.ptr] = state
		self.action_buf[self.ptr] = action
		self.reward_buf[self.ptr] = reward
		self.v_buf[self.ptr] = v
		self.logp_buf[self.ptr] = logp_pi
		self.ptr += 1
		if self.ptr == self.max_size: self.full = True

	@staticmethod
	def discount_cumsum(x, discount: float):
		"""
		magic from rllab for computing discounted cumulative sums of vectors.
		input: 
			vector x: [x0, x1, x2]

		output:
			[x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
		"""
		import scipy.signal
		return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

	def finish_path(self, last_val: float = 0.):
		"""
		Call this at the end of a trajectory, or when one gets cut off by an epoch ending. 
		This looks back in the buffer to where the trajectory started, 
		and uses rewards and value estimates from
		the whole trajectory to compute advantage estimates with GAE-Lambda,
		as well as compute the rewards-to-go for each state, to use as
		the targets for the value function.

		The "last_val" argument should be 0 if the trajectory ended
		because the agent reached a terminal state (died), 
		and otherwise should be V(s_T), the value function estimated for the last state.
		This allows us to bootstrap the reward-to-go calculation to account
		for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
		"""
		path_slice = slice(self.path_start_idx, self.ptr)
		reward_array = np.append(self.reward_buf[path_slice], last_val)
		v_array = np.append(self.v_buf[path_slice], last_val)
		
		deltas = reward_array[:-1] + self.gamma * v_array[1:] - v_array[:-1]
		# the next line implement GAE-Lambda advantage calculation
		gae_advantage = self.discount_cumsum(deltas, self.gamma * self.lam)
		self.advantage_buf[path_slice] = gae_advantage
		lambda_return = gae_advantage + v_array[:-1]  # G_\lambda = A_{GAE} + V
		self.reward_to_go_buf[path_slice] = lambda_return
		# self.advantage_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
		# # the next line computes `rewards-to-go`, to be targets for the value function
		# self.reward_to_go_buf[path_slice] = self.discount_cumsum(reward_array, self.gamma)[:-1]
		self.path_start_idx = self.ptr

	@staticmethod
	def statistics_scalar(x):
		x = np.array(x, dtype=np.float32)
		global_sum, global_n = np.sum(x), len(x)
		mean = float(global_sum) / float(global_n)
		global_sum_sq = np.sum((x - mean)**2)
		std = np.sqrt(global_sum_sq / global_n)  # compute global std
		return mean, std

	def sample(self, batch_size: Optional[int] = None):
		"""
		Call this at the end of an epoch to get all of the data from the buffer, 
		with advantages appropriately normalized (shifted to have mean zero and std one). 
		Also, resets some pointers in the buffer.
		"""
		assert self.full  # buffer has to be full before you can get
		indices = np.random.permutation(self.max_size)
		if batch_size is None:
			batch_size = self.max_size
		
		self.ptr, self.path_start_idx = 0, 0

		start_idx = 0
		while start_idx < self.max_size:
			yield self._get_samples(indices[start_idx : start_idx + batch_size])
			start_idx += batch_size

	def _get_samples(self, batch_inds: np.ndarray):
		advantage_buf = self.advantage_buf[batch_inds]
		# the next two lines implement the advantage normalization trick
		advantage_mean, advantage_std = self.statistics_scalar(advantage_buf)
		advantage_buf = (advantage_buf - advantage_mean) / (advantage_std + 1e-8)
		return (
			torch.as_tensor(self.state_buf[batch_inds],        device=self.device, dtype=torch.float32), 
			torch.as_tensor(self.action_buf[batch_inds],       device=self.device, dtype=torch.float32), 
			torch.as_tensor(self.reward_to_go_buf[batch_inds], device=self.device, dtype=torch.float32), 
			torch.as_tensor(advantage_buf,                     device=self.device, dtype=torch.float32), 
			torch.as_tensor(self.logp_buf[batch_inds],         device=self.device, dtype=torch.float32),
		)