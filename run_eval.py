import run_rl
import run_simple
import run_random
import matplotlib.pyplot as plt
import statistics
import plotting
import numpy as np
import sys

def run_plots(stats,env,Q=None, action_list=None, random=False, simple=False):
    

    total_wins=[]
    trials = 30
    max_episodes = 1000  
    max_iter_per_episode = 1000

    for trial in range(trials):
        total_wins.append(0)
        for episode in range(max_episodes):
            state = env.reset()
            done=False

            for step in range(max_iter_per_episode):
                if random:
                    action = env.action_space.sample()
                elif simple:
                    action = action_list[step]
                else:
                    action = np.argmax(Q[state,:])
                new_state, reward, done, info = env.step(action)

                if done:
                    if reward ==1:        
                        total_wins[trial]+=1
                    break
                state = new_state

    return  total_wins 

def main(problem_id):
    # get problem id
    try:
        problem_id = int(problem_id)
    except:
        print("Please run the program giving one integer argument between 0 and 7")  

    # run random, simple and A* agent, and produce plots based on their stats
    random_stats, random_env = run_random.main(problem_id)
    simple_stats, simple_env, action_list = run_simple.main(problem_id)
    rl_stats, rl_env, Q = run_rl.main(problem_id)
    stats=(random_stats,simple_stats,rl_stats, problem_id)
    plotting.plot_episode_stats(stats, smoothing_window=100)

    random_wins = run_plots(random_stats, random_env, random=True)
    simple_wins = run_plots(simple_stats, simple_env, action_list=action_list, simple=True)
    rl_wins = run_plots(rl_stats, rl_env, Q)

    print("Random Mean: ", statistics.mean(random_wins))
    print("Random Standard deviation: ", statistics.stdev(random_wins))
    print("Simple Mean: ", statistics.mean(simple_wins))
    print("Simple Standard deviation: ", statistics.stdev(simple_wins))
    print("Q Mean: ", statistics.mean(rl_wins))
    print("Q Standard deviation: ", statistics.stdev(rl_wins))
    
    fig = plt.figure(figsize=(10,5))
    fig.suptitle('Problem ID {}'.format(problem_id), fontsize=15)
    
    plt.plot(random_wins, label = "Random")
    plt.plot(simple_wins, label = "Simple")
    plt.plot(rl_wins, label = "Q-Learning")
    fig.legend()
    plt.ylabel('Number of times reaching the goal')
    plt.xlabel('Episodes')
    plt.show()
    
if __name__ == "__main__":
   main(sys.argv[1])
