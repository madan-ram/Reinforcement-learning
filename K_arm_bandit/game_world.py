import os, sys
import numpy as np

class KArmBandit():

	def __init__(self, k):

		self.bandit_distribution = np.random.random((k))

	def get_bandit_distribution(self):
		return self.bandit_distribution

	def get_reward(self, action):

		if self.bandit_distribution[action] <= np.random.random():
			return 1
		return 0
