from environment import TetrisEnv
import numpy as np
import torch
from torch import nn, optim
from tqdm.notebook import tqdm
import random
import curses
import time
import pygame

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


def train(env, 
          gamma=0.99, 
          lr=1e-3, 
          tau=0.5, 
          batch_size=128, 
          num_interactions= 10000, 
          eps=0.1, 
          verbose=True,
          render=False,
          summary_window=50):
    

    policy = QNetwork(env.state_dim, env.num_macro_actions)
    target = QNetwork(env.state_dim, env.num_macro_actions)
    target.load_state_dict(policy.state_dict())

    replay_buffer = ReplayMemory(10000)

    opt           = optim.Adam(policy.parameters(), lr=lr)
    loss          = nn.SmoothL1Loss() # look into if this is right?

    rng           = np.random.default_rng()

    state         = env.reset()
    ep_r          = 0
    ep_rewards    = []
    
    iterator      = tqdm(range(num_interactions)) if verbose else range(num_interactions)
    
    for step in iterator:
        if render:
            env.render()

        
        # USING GREEDY EPSILON

        # NOTE for some reason doing it like it was below was making the score get worse?
        # eps = max(eps_min, eps_start - (eps_start - eps_min) * (i / (num_interactions - 1)))
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
            if verbose:
                epi = len(ep_rewards)
                print(f"Episode {epi:3d} - Reward: {ep_r:.2f}")
            state = env.reset()
            ep_r = 0
        
        if verbose:
            if ep_rewards:
                window = min(len(ep_rewards), summary_window)
                avg_last = sum(ep_rewards[-window:]) / window
                print(f"\nDone {len(ep_rewards)} episodes.  "
                    f"Avg reward over last {window}: {avg_last:.2f}")
            else:
                print("\nNo complete episodes were recorded.")

    return policy, ep_rewards

def evaluate(env: TetrisEnv, policy: QNetwork, episodes: 3, render_delay: 0.02):
    for epi in range(1, episodes + 1):
        state, done, total = env.reset(), False, 0.0
        while not done:
            with torch.no_grad():
                logits = policy(torch.tensor(state, dtype=torch.float32))
                act    = int(logits.argmax().item())
            state, r, done = env.step(env.actions[act])
            total += r
            env.render(env.stdscr)
            time.sleep(render_delay)
        print(f"[Eval] Episode {epi}: {total:.2f}")

def evaluate_pygame(policy: QNetwork, episodes=3, render_delay=0.02):
    env = TetrisEnv(graphical=True)
    for epi in range(1, episodes + 1):
        state, done, total = env.reset(), False, 0.0
        while not done:
            with torch.no_grad():
                logits = policy(torch.tensor(state, dtype=torch.float32))
                act = int(logits.argmax().item())
            state, r, done = env.macro_step(env.macro_actions[act])
            total += r
            env.game.render_pygame()
            pygame.time.wait(int(render_delay))
        print(f"[Eval] Episode {epi}: {total:.2f}")
    
    input("Press Enter to quit...")
    pygame.quit()
"""
run this one if you want it to render in terminal
using python3 -i agent.py
make sure it is in a terminal window sized adequately large or you will get errors
"""
def main(stdscr, policy: QNetwork):
    env = TetrisEnv(graphical=True)
    # evaluate(env, policy, episodes=5, render_delay=0.05)
    # stdscr.addstr(0, 0, "Press any key to exit...")
    # stdscr.getch()
    
    

if __name__ == "__main__":
    env, policy = TetrisEnv(graphical=True), None
    policy, rewards = train(
        env,
        lr=2e-4,
        num_interactions=20_000,
        verbose=True,   # suppress tqdm & prints
        render=False     # never call env.render()
    )
    print(rewards)
    
    # curses.wrapper(main, policy)
    # evaluate_pygame(policy, episodes=5, render_delay=0.05)

