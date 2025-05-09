from environment import TetrisEnv
import numpy as np
import torch
from torch import nn, optim
from tqdm.notebook import tqdm
import random
import curses

""" Uncomment this if you want to just run it without rendering """
# ENV = TetrisEnv()

class QNetwork(nn.Module):

    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, num_actions)
        )
    
    def forward(self, x):
        return self.model(x)
    
class ReplayMemory: 
    def __init__(self, cap, evict_oldest=False):
        self.capacity = cap
        self.data = []
        self.evict_oldest = evict_oldest
        if self.evict_oldest: 
            self.oldest = 0
    
    def push(self, state, action, reward, nstate, term):
        if len(self.data) < self.capacity: 
            self.data.append((state, action, reward, nstate, term))
        else: 
            if self.evict_oldest:
                idx = self.oldest
                self.oldest = (self.oldest + 1) % self.capacity
            else: 
                idx = random.randint(0, self.capacity - 1)
            self.data[idx] = (state, action, reward, nstate, term)

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)
    
    def __len__(self):
        return len(self.data)
    
    def empty(self):
        self.data = []
        if self.evict_oldest:
            self.oldest = 0


def train(env, gamma=0.99, lr=1e-3, tau=0.5, batch_size=128, num_interactions= 10000, eps=0.1):
    policy = QNetwork(env.state_dim, env.num_actions)
    target = QNetwork(env.state_dim, env.num_actions)
    target.load_state_dict(policy.state_dict())

    replay_buffer = ReplayMemory(10000)

    opt = optim.Adam(policy.parameters(), lr=lr)
    loss = nn.SmoothL1Loss() # look into if this is right?

    rng = np.random.default_rng()

    eps_start = 1.0
    eps_min = 0.05

    state = env.reset()
    ep_r = 0
    ep_rewards = []
    for i in tqdm(range(num_interactions)):
        # USING GREEDY EPSILON
        # eps = max(eps_min, eps_start - (eps_start - eps_min) * (i / (num_interactions - 1)))
        if rng.random() < eps: 
            action_idx = rng.integers(0, env.num_actions)
            action = env.actions[action_idx]
        else: 
            action_idx = policy(torch.tensor(state, dtype=torch.float)).argmax()
            action = env.actions[action_idx]

        nstate, reward, term = env.step(action)
        replay_buffer.push(state, action_idx, reward, nstate, term)
        state = nstate
        ep_r += reward

        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            st_batch, act_batch, r_batch, nst_batch, t_batch = zip(*batch)
            st_batch = torch.tensor(np.array(st_batch)).float()
            act_batch = torch.tensor(np.array(act_batch)).unsqueeze(dim=1)
            r_batch = torch.tensor(np.array(r_batch)).float()
            nst_batch = torch.tensor(np.array(nst_batch)).float()
            t_batch = torch.tensor(np.array(t_batch))

            pred_vals = policy(st_batch).gather(1, act_batch).squeeze()

            pred_next_vals = target(nst_batch).max(dim=1).values

            pred_next_vals[t_batch] = 0.0

            expected_q = r_batch + gamma * pred_next_vals

            loss_val = loss(pred_vals, expected_q)

            opt.zero_grad()
            loss_val.backward()
            opt.step()

        p_state_dict = policy.state_dict()
        t_state_dict = target.state_dict()
        for key in p_state_dict:
            t_state_dict[key] = p_state_dict[key] * tau + t_state_dict[key] * (1 - tau)
        target.load_state_dict(t_state_dict)

        if term: 
            ep_rewards.append(ep_r)
            state = env.reset()
            ep_r = 0

    return policy, ep_rewards

# q_policy, q_returns = train(ENV, lr=2e-4, num_interactions=10000)
# print(q_returns)

"""
run this one if you want it to render in terminal
using python3 -i agent.py
make sure it is in a terminal window sized adequately large or you will get errors
"""
def main(stdscr):
    env = TetrisEnv(stdscr)
    policy, rewards = train(env, lr=2e-4, num_interactions=20000)
    print(rewards)

if __name__ == "__main__":
    curses.wrapper(main) 
