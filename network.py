from environment import TetrisEnv
import numpy as np
import torch
from torch import nn, optim
from tqdm.notebook import tqdm
import random
import curses
import time
import pygame

class QNetwork(nn.Module):

    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),            
            nn.Linear(128, num_actions)
        )
    
    def forward(self, x):
        return self.model(x)
    
class ReplayMemory: 
    def __init__(self, cap):
        self.capacity = cap
        self.data = []
    
    def push(self, state, action, reward, nstate, term):
        if len(self.data) < self.capacity: 
            self.data.append((state, action, reward, nstate, term))
        else: 
            idx = random.randint(0, self.capacity - 1)
            self.data[idx] = (state, action, reward, nstate, term)

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)
    
    def __len__(self):
        return len(self.data)
    
# using macro actions for delayed reward training. 
# change anything with 'macro' to just 'actions' for simple training
def train(env, 
          gamma=0.99, 
          lr=1e-3, 
          tau=0.5, 
          batch_size=128, 
          num_interactions= 10000, 
          eps=0.1, 
          verbose=True,
          render=False):

    policy = QNetwork(env.state_dim, env.num_macro_actions)
    target = QNetwork(env.state_dim, env.num_macro_actions)
    target.load_state_dict(policy.state_dict())

    replay_buffer = ReplayMemory(10000)

    opt = optim.Adam(policy.parameters(), lr=lr)
    loss = nn.SmoothL1Loss() # for DQN

    rng = np.random.default_rng()

    state = env.reset()
    ep_r = 0
    ep_rewards = []
    
    for i in tqdm(range(num_interactions)):
        # renders to terminal
        if render:
            env.render()
        
        # using greedy epsilon: policy action with prob 1 - epsilon, exploration otherwise
        if rng.random() < eps: 
            action_idx = rng.integers(0, env.num_macro_actions)
        else: 
            action_idx = policy(torch.tensor(state, dtype=torch.float)).argmax()
        action = env.macro_actions[action_idx]

        nstate, reward, term = env.macro_step(action)
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
            # print episode rewards while training
            if verbose:
                epi = len(ep_rewards)
                print(f"Episode {epi:3d} - Reward: {ep_r:.2f}")
            state = env.reset()
            ep_r = 0

    return policy, ep_rewards

# to run the network, call python3 -i network.py
def main(stdscr, policy: QNetwork):
   pass 
    
# set up to run and show graphics while training
if __name__ == "__main__":
    env, policy = TetrisEnv(graphical=True), None
    policy, rewards = train(
        env,
        lr=2e-4,
        num_interactions=40000,
        verbose=True,   
        render=False     
    )
    pygame.quit()
    
    # the following renders it in terminal while training
    # curses.wrapper(main, policy) 

