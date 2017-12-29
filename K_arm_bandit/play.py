import numpy as np
import os, sys
from game_world import *
import math

class Policy():

	def __init__(self, k):
		self.value_function = [1.0/k]*k
		# N is a action counter vector
		self.N = [0]*k

	def get_value_function(self, action):
		return self.value_function[action]

	def set_value_function(self, action, value):
		self.value_function[action] = value

	def get_all_value_function(self):
		return self.value_function

	def softmax_with_temperature(self, X, temperature):
		X = np.asarray(X)/temperature
		return np.exp(X)/np.sum(np.exp(X))

	def get_random_action(self, temperature=0.01):
		p = self.softmax_with_temperature(self.get_all_value_function(), temperature=temperature)
		return np.random.choice(range(k), p=p)

	def UCB(self, iteration_num, exploration_prob=0.0):
		ucb_estimation = exploration_prob*np.sqrt(math.log(iteration_num)/np.asarray(self.N))
		action = np.argmax([x+y for x, y in zip(self.get_all_value_function(), ucb_estimation)])
		return action

	def get_max_action(self):
		action = np.argmax(self.value_function)
		return action

	def update_value_function_using_averaging(self, action, reward, iteration_num):
		self.set_value_function(action, float((self.get_value_function(action)*(iteration_num-1))+reward)/iteration_num)
		self.N[action] += 1

if __name__ == '__main__':
	k=10
	# epsilon is the exploration probablity
	policy_algo = 'UCB'
	# policy_algo can be greedy or UCB

	num_games = 50
	num_iteration = 1000
	for i in range(11):
		total_reward = 0
		epsilon = i/10
		for game_id in range(num_games):
			bandit_game = KArmBandit(k=k)
			player_policy = Policy(k=k)
			for _id in range(num_iteration):

				# Check all bandit atleast once
				if _id<k:
					action = _id
				else:
					if 	policy_algo == 'greedy':
						# Select action using greedy
						# For high temperatures (temperature -> infinity), all actions have nearly the same probability and the lower the temperature, the more expected rewards affect the probability. 
						# For a low temperature (temperature -> 0+), the probability of the action with the highest expected reward tends to 1
						if np.random.random() <= epsilon:
							action = player_policy.get_random_action(temperature=0.5)
						else:
							action = player_policy.get_max_action()

					# Selection action using UCB
					if 	policy_algo == 'UCB':
						action = player_policy.UCB((_id+1), exploration_prob=epsilon)

				reward = bandit_game.get_reward(action)
				player_policy.update_value_function_using_averaging(action, reward, _id+1)			

				total_reward += reward

		print('Epsilon: ', epsilon, ' average reward: ', total_reward/float(num_iteration*num_games))
