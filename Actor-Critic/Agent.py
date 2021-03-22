import gym
import pybullet_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp
import time
import numpy as np

ENV = gym.make("InvertedPendulumSwingupBulletEnv-v0")
OBS_DIM = ENV.observation_space.shape[0]
ACT_DIM = ENV.action_space.shape[0]
ACT_LIMIT = ENV.action_space.high[0]
ENV.close()
GAMMA = 0.9
EPS = 0.0001
##############################################################
############ 1. Actor Network, Critic Network 구성 ############
##############################################################

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.act_fc = nn.Linear(OBS_DIM, 128)
        self.act_mu = nn.Linear(128, ACT_DIM)
        self.act_sigma = nn.Linear(128, ACT_DIM)

        self.cri_fc = nn.Linear(OBS_DIM, 128)
        self.cri_v = nn.Linear(128, 1)

        self.relu = nn.ReLU6()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        nn.init.xavier_normal(self.act_fc.weight)
        nn.init.xavier_normal(self.act_mu.weight)
        nn.init.xavier_normal(self.act_sigma.weight)
        nn.init.xavier_normal(self.cri_fc.weight)
        nn.init.xavier_normal(self.cri_v.weight)
        
    def act(self, x):
        x = self.relu(self.act_fc(x))
        mu = self.tanh(self.act_mu(x))
        sigma = self.softplus(self.act_sigma(x)) + EPS

        return mu, sigma

    def cri(self, x):
        x = self.relu(self.cri_fc(x))
        v = self.cri_v(x)

        return v

###########################################################################################
############  2. Local actor 학습(global actor, n_steps 받아와서 학습에 사용합니다.)  ############
###########################################################################################

def Worker(global_actor, n_steps, multi):
    if n_steps == 1 and multi == 1:
        mode = "SS"
    elif n_steps != 1 and multi == 1:
        mode = "MS"
    elif n_steps !=1 and multi != 1:
        mode = "MM"

    env = gym.make('InvertedPendulumSwingupBulletEnv-v0')

    local_actor = ActorCritic()

    if mode == "SS":
        lr = 0.001
    elif mode == "MS":
        lr = 0.0005
    elif mode == "MM":
        lr=0.0001
    
    optimizer = optim.Adam(global_actor.parameters(), lr=lr)

    t = 1
    score = 0.0
    beta = 0.05
    start_time = time.time()
    for train_episode in range(3000):
        local_actor.load_state_dict(global_actor.state_dict())

        t_start = t - 1

        state = env.reset()
        done = False

        rewards, log_probs, values, Rs = [], [], [], []
        policy_losses, value_losses = [], []
        entropies = []
        R = 0

        while True:
            # get action
            mu, sigma = local_actor.act(torch.from_numpy(state).float())
            norm_dist = Normal(mu, sigma)
            action = norm_dist.sample()
            action = torch.clamp(action, min=-ACT_LIMIT, max=ACT_LIMIT)

            # get next_state and reward according to action
            next_state, reward, done, _ = env.step(action)
            score += reward

            log_prob = norm_dist.log_prob(action)
            value = local_actor.cri(torch.from_numpy(state).float())
            entropy = norm_dist.entropy()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

            # gradient update
            if t - t_start == n_steps or done:
                if done:
                    R = 0
                else:
                    R = local_actor.cri(torch.from_numpy(next_state).float())

                for r in rewards[::-1]:
                    R = r + GAMMA * R
                    Rs.insert(0, R)

                for log_prob, value, entropy, R in zip(log_probs, values, entropies, Rs):
                    advantage = R - value.item()

                    policy_losses.append(-(log_prob * advantage + beta * entropy))
                    value_losses.append(F.mse_loss(value, torch.tensor([R])))

                loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
                
                optimizer.zero_grad()
                loss.backward()
                
                for local_param, global_param in zip(local_actor.parameters(), global_actor.parameters()):
                    global_param._grad = local_param.grad

                optimizer.step()

                local_actor.load_state_dict(global_actor.state_dict())

                rewards, log_probs, values, Rs = [], [], [], []
                policy_losses, value_losses = [], []
                entropies = []
                R = 0

                state = next_state
                t += 1
                t_start = t - 1

                if done:
                    if mode == "SS":
                        if score > 500:
                            optimizer.param_groups[0]['lr'] = 0.00005
                        elif score > 400:
                            optimizer.param_groups[0]['lr'] = 0.0001
                        elif score > 300:
                            optimizer.param_groups[0]['lr'] = 0.0002
                        elif score > 200:
                            optimizer.param_groups[0]['lr'] = 0.0003
                        elif score > 100:
                            optimizer.param_groups[0]['lr'] = 0.0005

                    beta = beta * 0.999 if beta > 0.025 else 0.025
                    break
            else:
                state = next_state
                t += 1

        #print("Train Episode: {}, Score: {:.1f}, Time: {:.2f}".format(train_episode, score, time.time() - start_time))
        score = 0.0

    env.close()
    print("Training process reached maximum episode.")
