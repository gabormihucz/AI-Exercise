import gym
import time
from IPython.display import clear_output
import random
import plotting
import numpy as np
import time
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import os, sys
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import lines

# problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent 
def main(problem_id):
    # get problem_id
    try:
        problem_id = int(problem_id)
    except:
        print("Please run the program giving one integer argument between 0 and 7")

    # Reset the random generator to a known state (for reproducability)
    np.random.seed(12)

    # initialise environment as stochastic
    # reward for the whole i -0.06, this was the best reward by experimentation
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=True, reward_hole=-0.06)

    # initalise Q-table as a 64x4 array holding all 0s
    states = env.observation_space.n
    actions = env.action_space.n
    Q = np.zeros((states, actions))

    max_episodes = 10000  
    max_iter_per_episode = 1000

    # RL parameters
    alpha = 0.1  # learning rate
    gamma = 0.99 # discount rate
    epsilon = 1  # threshold that decides whether the agent explores or exploits its knowledge about the world
                 # first episode will be completely random
    # initialise summary stats
    stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(max_episodes),
            episode_rewards=np.zeros(max_episodes))    

    for episode in range(max_episodes):
        state = env.reset()
        done = False
        # every 100th episode set epsilon large (but gradually lower as the number of iterations increase), to introduce occasional exploration
        if episode % 100 == 0:
            epsilon = (1-(episode/max_episodes))
        
        for step in range(max_iter_per_episode):
          
            # take best action according to Q-table if random value is greater than epsilon, otherwise take a random action
            random_value = random.uniform(0,1)
            if random_value > epsilon:
                action = np.argmax(Q[state,:])
            else:
                action = env.action_space.sample()

            # get the outcomes of the action  
            new_state, reward, done, info = env.step(action)

            # update Q-table for Q(state,action)
            Q[state, action] = Q[state,action] + alpha * (reward + gamma*np.max(Q[new_state,:]) - Q[state,action])
           
            # Update statistics
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = step

            # update current state and total reward
            state = new_state

            if done:
                break
        # set epsilon to a low value to ensure exploitation, this value worked well based on empirical analysis
        epsilon = 0.015

    # write Q-table to a file and average reward to a file
    with open("rl_info.txt", "w+") as f:
        f.write("\nProblem ID: {}\n".format(problem_id))
        f.write("Average reward: {} \n".format(np.mean(stats.episode_rewards)))
        f.write("Q-table:\n")
        f.write(str(Q))

    # return values for the evaluation script
    return (stats,env, Q)
    
if __name__ == "__main__":
   main(sys.argv[1])
