run_random.py (a random, senseless agent), run_simple.py (a deterministic agent with full knowledge of the environment using A\* search) and run_rl.py (a Q-learning agent in a stochastic environment) are three agents attempting to succeed in [OpenAI's FrozenLake environment](https://gym.openai.com/envs/FrozenLake-v0/)

**Setup:**  
In order to run the scripts, please create a virtual environment and install the dependencies using the following command:    
`pip install -r dependencies.txt`  
**Run:**  
Each run_xxx.py file expects a problem_id ( [0-7] ) as input, including run_eval.py.  
run_eval.py will run all three agents on the specified problem id and plot three graphs for each (episode length over time, episode reward over time and number of episodes per time step).

The file report.pdf contains a short summary of the implementation and evaluation of the agents.  
