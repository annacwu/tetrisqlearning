# Playing Tetris with Reinforcement Learning

**Authors**: Anna Wu and Ryan Koeninger  
**Date**: May 2025

---

## Introduction

Tetris was one of the earliest examples of video games ever created and remains one of the most popular games of all time. The premise of Tetris is simple enough: the user moves and rotates falling blocks on a 10×20 cell field, with blocks stacking on top of each other. If the stack of blocks becomes too high, the game ends and the player must restart. Survival is the main goal of the game, and this is done by clearing lines—if a row is fully filled in by blocks, with no gaps, it is removed from the board, and all blocks above it are moved down to fill in the space. The game is scored by the number of blocks placed and the number of lines cleared, with increased rewards given for higher numbers of rows cleared with a single piece placement.

Tetris provides an excellent area of study for machine learning techniques, particularly Reinforcement Learning (RL). RL involves a cycle wherein an Agent (the neural network) interacts with some Environment, with the goal of maximizing some reward from the environment and refining its action choices to this end. Tetris provides a clean translation from game to RL design, with:

- A small set of actions (rotating and moving blocks)
- A discrete environment space (the 10×20 cell game board)
- A simple reward system (based on lines cleared and points earned)

This State, Action, Reward sequence forms a Markov Decision Process, by which the Agent determines the best action to take at any given state to maximize long-term rewards.

Our goal was to train an RL model to play Tetris at a level comparable to or surpassing the average human, placing blocks in such a way as to survive for extended periods of time and clear lines consistently.

---

## Background

We based our project on Deep Q-learning (DQN), a method suited to discrete, finite action spaces as part of a Markov Decision Process (MDP). Our implementation relies on standard mathematical principles in DQN such as the **Bellman equation** for the optimal action-value function:

Q*(s, a) = r(s, a) + γ * E[ Q*(s′, a′) ]

With this, our policy chooses the action that maximizes the expected Q-value:

π*(s) = argmax_a Q*(s, a)

To balance exploitation versus exploration, we used an ε-greedy policy with ε = 0.1. While we considered a decaying-ε strategy, we concluded that continual exploration made more sense given the randomness of the board at all stages.

We decided to use the board itself as input to the model—representing empty and filled cells directly—based on previous work in pixel-based learning for Atari games. We also adopted **macro actions** (i.e., sequences of moves per piece) to reduce reward delay, which is especially useful in sparse reward environments like Tetris.

---

## Implementation

Our implementation includes three main files:

- `tetris.py`
- `environment.py`
- `network.py`

### Base Game (`tetris.py`)

This is a terminal-playable version of Tetris, featuring:

- 7 standard tetrominoes
- A 10×20 board (0 = empty, 1–7 = filled cells)
- An active board and a locked board
- Standard movement: left, right, down, rotate (×3), hard drop
- “Bag” randomizer (standard Tetris feature)
- Rendering via `curses` or `pygame`

This game logic was designed to be simple yet comprehensive enough for neural network training.

### Environment (`environment.py`)

We define a `TetrisEnv` class to interface with the DQN, including input space, action space, and reward function.

#### Input Space

The input is a flattened 1D array (length 240) representing the active board. Each element is an integer from 0–7.

#### Action Space

Initially, we used low-level actions (`left`, `right`, `rotate`, etc.), but rewards were too sparse and the network learned to build tall towers.

We moved to **macro actions**—tuples of the form `(block_type, rotation, col)`. This expanded action space still remains discrete and finite but allowed the model to make higher-level choices and receive immediate feedback.

#### Reward Function

The reward function was the most carefully tuned component. We gave large positive rewards for clearing rows, and penalized:

- Tall towers
- Gaps in rows
- Irregular column heights
- Blocking in empty spaces

We also gave smaller positive rewards for:

- Broad block coverage
- Number of blocks placed
- Flush block placement
- Near-complete rows
- Traditional score increases

This helped balance early training where negative rewards were overwhelming.

### Network (`network.py`)

We implemented a standard Deep Q-Network (DQN) similar to models used in Atari reinforcement learning tasks. (See original paper: Mnih et al., 2013.) The model:

- Takes the 240-length input vector
- Passes it through several dense layers
- Outputs Q-values for all possible actions
- Uses experience replay and target networks for stability

---

## References

1. Sutton, R. S., & Barto, A. G. (2020). *Reinforcement Learning: An Introduction*.
2. Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning*. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
3. Shao, Y., et al. (2019). *A Survey on Deep Reinforcement Learning*. [arXiv:1812.05551](https://arxiv.org/abs/1812.05551)
