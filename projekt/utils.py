import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.core import ObsType, WrapperObsType


def plot_learning_curve(x, scores, epsilons, filename, avg_human, avg_random, avg_best_linear):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    scores_len = len(scores)
    running_avg = np.empty(scores_len)
    for i in range(scores_len):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    ax2.plot(x, running_avg, color="C1")
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.axhline(y=avg_human, color='C2', linestyle='-', label="avg_human")
    plt.axhline(y=avg_random, color='C3', linestyle='-', label="avg_random")
    plt.axhline(y=avg_best_linear, color='C4', linestyle='-', label="avg_best_linear")
    plt.legend(bbox_to_anchor=(1.0075, 1.15), ncol=3)

    plt.savefig(filename)


def make_env(env_name, human=False):
    # we create game environment
    if human:
        env = gym.make(env_name, render_mode="human", full_action_space=False)
    else:
        env = gym.make(env_name, full_action_space=False)
    # chose "better" frame
    env = TakeMaxFrame(env)
    # resize our observation
    env = gym.wrappers.ResizeObservation(env, 84)
    # convert our observation to grayscale
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    #custom wrapper to adjust observastion
    env = ReshapeObservationSpaceWrapper(env, shape=(84, 84))
    # we stack last 4 frames
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env


class ReshapeObservationSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(ReshapeObservationSpaceWrapper, self).__init__(env)
        self.shape = shape
        #reshape observation space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation: ObsType) -> WrapperObsType:
        #reshape observation
        observation = observation.reshape(*self.shape)
        #scale values
        observation = observation / 255.0
        return observation


class TakeMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super(TakeMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))

    def step(self, action):
        combined_reward = 0.0
        # we repeat each action for 4 steps
        terminated = False
        truncated = False
        for i in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            combined_reward += reward
            id = i % 2
            # we keep last 2 frames
            self.frame_buffer[id] = observation
            if terminated or truncated:
                break
        # we return max frame
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, combined_reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation = self.env.reset()
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = observation
        return observation


class ReplayBuffer(object):
    def __init__(self, buffer_size, input_shape, eps=1e-2, alpha=0.1, beta=0.1):
        self.buffer_size = buffer_size
        self.buffer_cntr = 0
        # we keep data in numpy arrays
        self.state_buffer = np.zeros((self.buffer_size, *input_shape), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_size, *input_shape), dtype=np.float32)
        self.action_buffer = np.zeros(self.buffer_size, dtype=np.int64)
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminal_buffer = np.zeros(self.buffer_size, dtype=np.bool)

    # adding transition to buffer
    def add_transition(self, state, action, reward, next_state, done):
        id = self.buffer_cntr % self.buffer_size
        self.state_buffer[id] = state
        self.action_buffer[id] = action
        self.reward_buffer[id] = reward
        self.next_state_buffer[id] = next_state
        self.terminal_buffer[id] = done
        self.buffer_cntr += 1

    # sampling from buffer
    def sample_buffer(self, batch_size):
        max_buffer = min(self.buffer_cntr, self.buffer_size)
        batch = np.random.choice(max_buffer, batch_size, replace=False)
        states = self.state_buffer[batch]
        actions = self.action_buffer[batch]
        rewards = self.reward_buffer[batch]
        next_states = self.next_state_buffer[batch]
        terminal = self.terminal_buffer[batch]
        return states, actions, rewards, next_states, terminal
