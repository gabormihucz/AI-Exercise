import os, sys
print("Working dir:"+os.getcwd())
print("Python version:"+sys.version)

from copy import deepcopy 
import numpy as np

from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import os, sys
from helpers import *
import plotting

import matplotlib.pyplot as plt
from matplotlib import lines

import networkx as nx
print("networkx version:"+nx.__version__)

from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display
print("ipywidgets version:" + widgets.__version__)

# Download/pull the AIMA toolbox from https://github.com/aimacode/aima-python
# And add path the to the AIMA Python Toolbox folder on your system
AIMA_TOOLBOX_ROOT="aima-python"
sys.path.append(AIMA_TOOLBOX_ROOT)
from search import *

# best first graph search from AIMA toolbox
def my_best_first_graph_search(problem_id, problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    flag=False
    action_list=[] # to store the optimal path

    # heuristic function
    f = memoize(f, 'f')
    node = Node(problem.initial) # assign current node

    #test if the current node is the goal node
    if problem.goal_test(node.state):
        print("There is no action to take, you are at the goal already.")
        run( action_list, problem_id )

    # create a priority queue with the heuristic function, shortest distance to goal comes first
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    
    explored = set() # keep track of explored nodes
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            # if we reached the goal, create an action list based on optimal path found by A* search
            for node in node.path():  
                if flag:
                    previous_x = int(previous_state[2])     
                    previous_y = int(previous_state[4])
                    current_x = int(node.state[2])
                    current_y = int(node.state[4])
                    #create action list based on coordinates of currently explored node compared to previous node
                    if current_x > previous_x: 
                        action_list.append(1)
                    if current_x < previous_x:
                        action_list.append(3)
                    if current_y > previous_y:
                        action_list.append(2)
                    if current_y < previous_y:
                        action_list.append(0)
                    previous_state = node.state
                else:
                    previous_state = node.state
                    flag = True
            run(action_list, problem_id ) # perform the learned optimal path
                    
        
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)

            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)

    return run(action_list, problem_id )

# A* search is an extension of the best first graph search function with a suitable heuristic
def my_astar_search(problem_id, problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h') # define the heuristic function
    return my_best_first_graph_search(problem_id, problem, lambda n: n.path_cost + h(n))

# parses the LochLomond environment to create a graph out of the states and actions
# returns a GraphProblem based on locations and actions that the parser returned
def setup(problem_id):
    np.random.seed(12)

    # Setup the parameters for the specific problem (you can change all of these if you want to) 
    reward_hole = -1.0     
    is_stochastic = False  # deterministic

    max_episodes = 10000   
    max_iter_per_episode = 1000 

    # Generate the specific problem 
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole)

    # parse with provided parser and create undirected graph
    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)
    state_space_actions = UndirectedGraph(state_space_actions)
    state_space_actions.locations = state_space_locations
    # initialise a graph
    G = nx.Graph()

    # positions for node labels
    node_label_pos = {k:[v[0],v[1]-0.25] for k,v in state_space_locations.items()} # spec the position of the labels relative to the nodes

    # use this while labeling edges
    edge_labels = dict()

    # add edges between nodes in the map - UndirectedGraph defined in search.py
    for node in state_space_actions.nodes():
        connections = state_space_actions.get(node)
        for connection in connections.keys():
            distance = connections[connection]        
            G.add_edge(node, connection) # add edges to the graph        
            edge_labels[(node, connection)] = distance # add distances to edge_labels

    return GraphProblem(state_initial_id, state_goal_id, state_space_actions)

# action_list contains the optimal actions to reach the goal in a problem environment
def run(action_list, problem_id):
    # Setup the parameters for the specific problem (you can change all of these if you want to) 
    reward_hole = -0.06    

    max_episodes = 10000   
    max_iter_per_episode = 1000 

    # Generate the specific problem 
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole)
    np.random.seed(12)

    # initialise stats
    stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(max_episodes),
            episode_rewards=np.zeros(max_episodes)) 

    for e in range(max_episodes): # iterate over episodes

        observation = env.reset() # reset the state of the env to the starting state     

        for i in range(max_iter_per_episode):
            action = action_list[i] # your agent goes here (the current agent takes random actions)
            
            observation, reward, done, info = env.step(action) # observe what happends when you take the action
            # update stats
            stats.episode_rewards[e] += reward
            stats.episode_lengths[e] = i
            
            if done:
                break
    return (stats, env, action_list)

def main(problem_id):
    # get problem id
    try:
        problem_id = int(problem_id)
    except:
        print("Please run the program giving one integer argument between 0 and 7")  

    # create GraphProblem to run A* search on
    frozen_problem = setup(problem_id)

    # get stats and action_list to use for the evaluation script / run A* search witht the standard GraphProblem heuristic: a straight-line distance from a node's state to goal
    stats,env, action_list = my_astar_search(problem_id, problem=frozen_problem, h=None)
    # write average reward to a file
    with open("simple_info.txt", "w") as f:
        f.write("\nProblem ID: {}\n".format(problem_id))
        f.write("Average reward: {} \n".format(np.mean(stats.episode_rewards)))
                
    return (stats, env, action_list)


if __name__ == "__main__":
   main(sys.argv[1])
