# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```
```
Python 3.x
Gymnasium
NumPy
Matplotlib
Google Colab / Jupyter Notebook
```
---

## Environment Description

```
env_desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]
```
## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

1.Start the FrozenLake-v1 environment using Gymnasium.

2.Initialize the Q-table with zeros for all states and actions.

3.Set the learning parameters: number of episodes, learning rate alpha, discount factor gamma, and exploration rate epsilon.

4.Generate a complete episode using an epsilon-greedy policy.

5.Store the state, action, and reward for every step of the episode.

6.Calculate the return Gt by processing the episode.

7.Update the Q-value using:
```
       Q(s,a)←Q(s,a)+α[Gt−Q(s,a)]
```
8.Decrease epsilon gradually to reduce exploration and increase exploitation.

9.Repeat the process for all episodes.

10.Extract the learned policy by selecting the action with the maximum Q-value for each state.

11.Calculate the state-value function using:

```
V(s)=maxaQ(s,a)
```
12.Display the Q-table, state-value function, learned policy, average reward, and learning curve.


## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python


import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create Environment
# -------------------------------------------------

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Environment details
n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------


def epsilon_greedy_action(state, epsilon):
    """
    Selects an action using epsilon-greedy policy.
    """
    if np.random.random() < epsilon:
        return env.action_space.sample()      # Explore
    else:
        return np.argmax(Q[state])            # Exploit


# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current epsilon-greedy policy.
    Returns a list of (state, action, reward).
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):
        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode



# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

epsilon = epsilon_start

for episode_num in range(num_episodes):

    # Generate one complete episode
    episode = generate_episode(epsilon)

    G = 0

    # Process episode backwards
    for state, action, reward in reversed(episode):

        G = gamma * G + reward

        # Incremental Monte Carlo update
        Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])

    # Store episode reward
    episode_rewards.append(sum([step[2] for step in episode]))

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Extract Greedy Policy
# -------------------------------------------------

optimal_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("Name: Ramesh Renuka")
print("Register Number: 212223240136")
def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)
    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(optimal_policy)

success_rate = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", success_rate)



# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500
moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")
plt.grid(True)
plt.show()

env.close()

```

## Output

### Final Q-table:




### Manto Carlo Learning curve :



## Result

The Monte Carlo Control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned the action-value function Q(s,a) through complete episodes using an epsilon-greedy policy.

## Inference

The performance of the Monte Carlo Control algorithm improves when the number of training episodes is increased. With 4,800 episodes, the agent achieved an average reward of 0.865 over the last 1,000 episodes, and the learning curve was still increasing, indicating that the agent was still learning. With 20,000 episodes, the average reward increased to 0.961, and the learning curve became more stable around a higher reward level. The Q-table and state-value estimates also became more reliable with additional training. Therefore, increasing the number of episodes from 4,800 to 20,000 provided more experience, improved the learned policy, and resulted in better and more stable performance. Hence, 20,000 episodes gives better learning performance than 4,800 episodes.

