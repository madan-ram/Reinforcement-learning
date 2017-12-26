__author__ = 'Madan Ram'
import gym
import numpy as np
import gym_tic_tac_toe
import random
from datetime import datetime
random.seed(datetime.now())

class unique_element:
	def __init__(self, value, occurrences):
		self.value = value
		self.occurrences = occurrences

def perm_unique(elements):
	eset=set(elements)
	listunique = [unique_element(i, elements.count(i)) for i in eset]
	u=len(elements)
	return perm_unique_helper(listunique, [0]*u, u-1)

def perm_unique_helper(listunique, result_list, d):
	if d < 0:
		yield tuple(result_list)
	else:
		for i in listunique:
			if i.occurrences > 0:
				result_list[d] = i.value
				i.occurrences -= 1
				for g in  perm_unique_helper(listunique, result_list, d-1):
					yield g
				i.occurrences += 1


def random_plus_middle_move(moves, p):
	if ([p, 4] in moves):
		m = [p, 4]
	else:
		m = random_move(moves, p)
	return m

def random_move(moves):
	m = random.choice(moves)
	return m

def get_move_from_state(state_before, state_after):
	for i, j in zip(state_before, state_after):
		if i != j:
			return j
	return None


def check_game(board, p):
	# check game over
	for i in range(3):
		# horizontals and verticals
		if ((board[i * 3] == p and board[i * 3 + 1] == p and board[i * 3 + 2] == p)
			or (board[i + 0] == p and board[i + 3] == p and board[i + 6] == p)):
			reward = p
			return True

	# diagonals
	if((board[0] == p and board[4] == p and board[8] == p)
		or (board[2] == p and board[4] == p and board[6] == p)):
			reward = p
			return True



class Policy:

	def __init__(self, player_id, init_value=0.5, alpha=0.1):
		self.player_id = player_id
		self.init_value = init_value
		self.value_function = {}

		self.set_policy_value([0]*9, 0.5)
		states = []
		px = 0
		py = 0
		for i in range(9):
			state = [0]*9
			if i%2 == 0:
				px += 1
			else:
				py += 1

			for x in range(px):
				state[x] = 1

			for y in range(px, px+py):
				state[y] = -1
			# print(states)
			states += [list(d) for d in perm_unique(state)]

		for state in states:
			self.value_function[self.__convert_state_to_state_str(state)] = random.random()


	def __convert_state_to_state_str(self, state):
		return ':'.join(map(str, state))

	def __convert_state_str_to_state(self, state_str):
		return list(map(int, state.split(':')))

	def get_policy_value(self, state):
		return self.value_function[self.__convert_state_to_state_str(state)]

	def set_policy_value(self, state, value):
		self.value_function[self.__convert_state_to_state_str(state)] = value

	def move_to_state(self, state, move):
		new_state = state.copy()
		new_state[move] = self.player_id
		return state

	def get_max_move_from_moves(self, state, moves):
		max_move = None
		max_value = None
		for move in moves:
			
			new_state = self.move_to_state(state, move)

			value = self.get_policy_value(new_state)
			if max_value is None:
				max_value = value
				max_move = move
			elif max_value < value:
				max_value = value
				max_move = move

		return max_move, max_value
		
	def update_value_function(self, previous_state, after_state, reward):
		prev_value = self.get_policy_value(previous_state)
		update_value = prev_value+alpha*(reward-prev_value)
		# print(update_value, previous_state, after_state)
		self.set_policy_value(previous_state, update_value)

if __name__ == '__main__':
	num_episodes = 2000
	num_steps_per_episode = 10
	init_value = 0.5
	alpha = 0.1
	explore_prob = 0.20
	
	explore_prob_A = 0.0
	explore_prob_B = 1.0

	collected_rewards = []
	# Create policy
	player_A_policy = Policy(player_id=1, init_value=init_value)
	player_B_policy = Policy(player_id=-1, init_value=init_value)
	
	# Start the game
	env = gym.make('tic_tac_toe-v0')
	
	number_of_draws = 0
	number_of_player_A_win = 0
	number_of_player_B_win = 0
	move_heat_map = None

	for i in range(num_episodes):
		s = env.reset()
		done = False
		
		total_reward = 0
		prev_player_state_A = None
		prev_player_state_B = None

		# print ("starting new episode")
		# env.render(mode='human', close=False)
		# print ("started")
		
		for j in range(num_steps_per_episode):

			moves = env.move_generator()
			# print ("moves: ", moves)
			if (not moves):
				print ("out of moves")
				break

			player_id = env.state['on_move']
			before_state = env.state['board']

			policy = None
			if player_id == player_A_policy.player_id:
				policy = player_A_policy
			else:
				policy = player_B_policy

			if player_id == player_A_policy.player_id:
				explore_prob = explore_prob_A
			else:
				explore_prob = explore_prob_B


			# Select action by max or random exploration
			_, move = random_move(moves)
			value = policy.get_policy_value(policy.move_to_state(before_state, move))

			if random.random() > explore_prob:
				# convert move to state
				move, value = policy.get_max_move_from_moves(before_state, [m for p, m in moves])

			# env.render()
			s1, reward, done, _ = env.step([player_id, move])
			after_state = env.state['board']
			
			total_reward += reward
		
			# Update learning by playing game
			prev_player_state = None
			if player_id == player_A_policy.player_id:
				prev_player_state = prev_player_state_A
			else:
				prev_player_state = prev_player_state_B

			if prev_player_state is not None and not done:
				policy.update_value_function(prev_player_state, after_state, reward=value)
			elif done:

				policy.update_value_function(prev_player_state, after_state, reward=1)
				print ("game over: ", reward)

				print('After state: ')
				board = np.asarray(after_state).reshape((3, -1))
				print(board)
				if move_heat_map is None:
					move_heat_map = board
				else:
					move_heat_map += board

				break

			# Update previous status
			if player_id == player_A_policy.player_id:
				prev_player_state_A = after_state
			else:
				prev_player_state_B = after_state

		if total_reward == 0:
			number_of_draws += 1
		elif total_reward == 1:
			number_of_player_A_win += 1
		elif total_reward == -1:
			number_of_player_B_win += 1

		collected_rewards.append(total_reward)
		print ("total reward ", total_reward, " after episode: ", i, ". steps: ", j+1)

	print("==============================================")
	print('Average moves heat map')
	print(move_heat_map/float(num_episodes))
	print("==============================================")

	print('Average reward of draws: ', float(number_of_draws)/num_episodes)
	print('Average reward of player A wins: ', float(number_of_player_A_win)/num_episodes)
	print('Average reward of player B wins: ', float(number_of_player_B_win)/num_episodes)

	print ("average score: ", sum(collected_rewards) / num_episodes)
	print("#########")

