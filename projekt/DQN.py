import numpy as np
from DQNAgent import DQNAgent
from utils import plot_learning_curve, make_env


# env = make_env("ALE/Pong-v5")
# env_name = "Pong-v5"
# best_score = -np.inf
# n_games = 10000
# max_n_steps = 1000000
# algo = "DQN"
# eps_min = 0.1
# eps_better = 0.5
# avg_human = 9.3
# avg_random = -20.7
# avg_best_linear = -19

env = make_env("ALE/CrazyClimber-v5")
env_name = "CrazyClimber-v5"
best_score = -np.inf
n_games = 10000
max_n_steps = 1000000
algo = "DQN"
eps_min = 0.1
eps_better = 50
avg_human = 35411
avg_random = 10781
avg_best_linear = 23411

agent = DQNAgent(gamma=0.99, eps=1, lr=0.0001,
                 input_dims=env.observation_space.shape,
                 n_actions=env.action_space.n, buffer_size=50000, eps_min=eps_min,
                 batch_size=32, replace=1000, eps_dec=1e-5,
                 chkpt_dir="models/", algo=algo,
                 env_name=env_name)

plot_name = agent.algo + '_' + agent.env_name + '_' + str(max_n_steps) + 'steps'
figure_file = 'plots/' + plot_name + '.png'
n_steps = 0
scores, eps_history, steps_array = [], [], []

for i in range(n_games):
    print(i)
    done = False
    observation, info = env.reset()
    score = 0
    while not done:
        action = agent.epsilon_greedy(observation)
        observation_step, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
        agent.add_transition(observation, action, reward, observation_step, done)
        agent.learn()
        observation = observation_step
        n_steps += 1
    scores.append(score)
    steps_array.append(n_steps)

    avg_score = np.mean(scores[-100:])
    print('episode: ', i,'score: ', score,
         ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
        'epsilon %.2f' % agent.eps, 'steps', n_steps)

    if avg_score > best_score + eps_better and n_steps > 300000:
        agent.save_models()
        best_score = avg_score
    eps_history.append(agent.eps)
    if n_steps > max_n_steps:
        break

plot_learning_curve(steps_array, scores, eps_history, figure_file, avg_human, avg_random, avg_best_linear)
