"""
  University of Glasgow 
  Artificial Intelligence 2018-2019
  Assessed Exercise

  Basic demo code for the CUSTOM Open AI Gym problem used in AI (H) '18-'19
"""
import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import os, sys
from helpers import *
import plotting
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import lines

print("Working dir:"+os.getcwd())
print("Python version:"+sys.version)

# problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent 
def main(problem_id):
    try:
        problem_id = int(problem_id)
    except:
        print("Please run the program giving one integer argument between 0 and 7")

    # Setup the parameters for the specific problem (you can change all of these if you want to)         
    reward_hole = -0.06    
    is_stochastic = True  

    max_episodes = 10000   
    max_iter_per_episode = 1000 

    # Generate the specific problem 
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole)

    # Reset the random generator to a known state (for reproducability)
    np.random.seed(12)

    # gather sample statistics
    stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(max_episodes),
            episode_rewards=np.zeros(max_episodes)) 

    for e in range(max_episodes): # iterate over episodes
        observation = env.reset() # reset the state of the env to the starting state     
        
        for i in range(max_iter_per_episode):
            action = env.action_space.sample() #take random actions
            observation, reward, done, info = env.step(action) # observe what happends when you take the action

            # update stats
            stats.episode_rewards[e] += reward
            stats.episode_lengths[e] = i
            
            if done:
                break

    # write average reward to a file
    with open("random_info.txt", "w") as f:
        f.write("\nProblem ID: {}\n".format(problem_id))
        f.write("Average reward: {} \n".format(np.mean(stats.episode_rewards)))
            
    # return stats for evaluation
    return (stats, env)
if __name__ == "__main__":
   main(sys.argv[1])
