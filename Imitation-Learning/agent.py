import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import time

GAMMA = 0.98
LEARNING_RATE = 0.0001
EPSILON = 0.001
TARGET_UPDATE_ITER = 400
BATCH_SIZE = 64
N_STEP = 10

class ReplayBuffer:
    def __init__(self):
        self.demo_replay = list()
        self.expr_replay = deque(maxlen=500)
        self.replay = list()

    def append_demo(self, transition):
        self.demo_replay.append(transition)

    def append_expr(self, transition):
        self.expr_replay.append(transition)
        self.update()

    def update(self):
        self.replay = self.demo_replay.copy() + list(self.expr_replay.copy())

    def sample(self, batch_size):
        return random.sample(self.replay, batch_size)

def get_demo_traj():
    demo = np.load("./demo_traj.npy", allow_pickle=True)
    replay_buffer = ReplayBuffer()

    for episode in demo:
        n_rewards = deque(maxlen=N_STEP)
        for s, a, r, ns, d in episode:
            n_rewards.append(r)
            replay_buffer.append_demo((s, a, np.array(n_rewards.copy()), ns, d, True))
    
    replay_buffer.update()

    return replay_buffer

##########################################################################
############                                                  ############
############                  DQfDNetwork 구현                 ############
############                                                  ############
##########################################################################
class DQfDNetwork(nn.Module):
    def __init__(self, in_size=4, hidden_size=32, out_size=2):
        super(DQfDNetwork, self).__init__()
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_size)

        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)

        return x

##########################################################################
############                                                  ############
############                  DQfDagent 구현                   ############
############                                                  ############
##########################################################################

class DQfDAgent(object):
    def __init__(self, env, use_per, n_episode):
        self.env = env
        self.use_per = use_per
        self.n_EPISODES = n_episode
        self.epsilon = EPSILON
        
        self.n_steps = 10

        self.replay_buffer = get_demo_traj()

        self.main_network = DQfDNetwork()
        self.target_network = DQfDNetwork()
        self.target_network.load_state_dict(self.main_network.state_dict())

    def get_action(self, state):
        state = torch.FloatTensor(state)
        if np.random.rand() > self.epsilon:
            action = torch.argmax(self.main_network(state).detach()).numpy()
        else:
            action = np.random.choice(2)
        return action
    
    def update(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def train(self):
        ###### 1. DO NOT MODIFY FOR TESTING ######
        test_mean_episode_reward = deque(maxlen=20)
        test_over_reward = False
        test_min_episode = np.inf
        ###### 1. DO NOT MODIFY FOR TESTING ######
        
        optimizer = optim.Adam(self.main_network.parameters(), lr=LEARNING_RATE)

        # Pretrain 5000 step
        self.pretrain()
        
        t = 1
        mean_episode_reward = []
        for e in range(self.n_EPISODES):
            ########### 2. DO NOT MODIFY FOR TESTING ###########
            test_episode_reward = 0
            ########### 2. DO NOT MODIFY FOR TESTING  ###########

            done = False
            n_rewards = deque(maxlen=10)
            state = self.env.reset()

            while not done:
                action = self.get_action(state)

                next_state, reward, done, _ = self.env.step(action)
                n_rewards.append(reward)
                self.replay_buffer.append_expr((state, action, np.array(n_rewards.copy()), next_state, done, False))

                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += reward      
                ########### 3. DO NOT MODIFY FOR TESTING  ###########
                minibatch = self.replay_buffer.sample(BATCH_SIZE)

                states, actions, rewards, next_states, dones, demos = map(np.array, zip(*minibatch))
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.tensor(dones).reshape(BATCH_SIZE, -1)
                demos = torch.tensor(demos).reshape(BATCH_SIZE, -1)

                next_action = torch.argmax(self.main_network(next_states).detach().squeeze(), dim=1).view(-1, 1)
                DQN_target = self.target_network(next_states).squeeze().gather(1, next_action)
                Q_hat = self.main_network(states).squeeze().gather(1, actions)

                # 1-step loss
                one_step_target = 1.0 + (~dones)*GAMMA*DQN_target.clone()
                J_DQ = F.mse_loss(one_step_target, Q_hat)

                # n-step loss
                n_step_target = (~dones)*DQN_target.clone()
                for i, rs in enumerate(rewards):
                    for r in rs[::-1]:
                        n_step_target[i] = r + GAMMA*n_step_target[i]
                J_n = F.mse_loss(n_step_target, Q_hat)            

                # Supervised loss
                Q_E = Q_hat.clone()
                a = torch.tensor([0, 1]*BATCH_SIZE).reshape(BATCH_SIZE, -1)
                l = torch.abs(a - actions) * 0.8
                J_E = torch.max(self.main_network(states).squeeze() + l, dim=1)[0].reshape(BATCH_SIZE, -1)
                J_E = (demos*(J_E - Q_E)).sum()

                # L2 loss
                J_L2 = torch.tensor(0.)
                for param in self.main_network.parameters():
                    J_L2 += torch.norm(param)
                J_L2 *= 1e-5

                loss = J_DQ + J_n + J_E + J_L2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % TARGET_UPDATE_ITER == 0 and t != 0:
                    self.update()

                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward)==20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########

                state = next_state
                t += 1

            ########### 5. DO NOT MODIFY FOR TESTING  ###########
            if test_over_reward:
                print("END train function")
                break
            ########### 5. DO NOT MODIFY FOR TESTING  ###########


        ########### 6. DO NOT MODIFY FOR TESTING  ###########
        return test_min_episode, np.mean(test_mean_episode_reward)
        ########### 6. DO NOT MODIFY FOR TESTING  ###########

    def pretrain(self):
        optimizer = optim.Adam(self.main_network.parameters(), lr=0.0005)
        for i in range(5000):
            minibatch = self.replay_buffer.sample(BATCH_SIZE)

            states, actions, rewards, next_states, dones, _ = map(np.array, zip(*minibatch))
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.tensor(dones).reshape(BATCH_SIZE, -1)

            next_action = torch.argmax(self.main_network(next_states).detach().squeeze(), dim=1).view(-1, 1)
            DQN_target = self.target_network(next_states).squeeze().gather(1, next_action)
            Q_hat = self.main_network(states).squeeze().gather(1, actions)

            # 1-step loss
            one_step_target = 1.0 + (~dones)*GAMMA*DQN_target.clone()
            J_DQ = F.mse_loss(one_step_target, Q_hat)

            # n-step loss
            n_step_target = (~dones)*DQN_target.clone()
            for t, rs in enumerate(rewards):
                for r in rs[::-1]:
                    n_step_target[t] = r + GAMMA*n_step_target[t]
            J_n = F.mse_loss(n_step_target, Q_hat)            

            # Supervised loss
            Q_E = Q_hat.clone()
            a = torch.tensor([0, 1]*BATCH_SIZE).reshape(BATCH_SIZE, -1)
            l = torch.abs(a - actions) * 0.8
            J_E = torch.max(self.main_network(states).squeeze() + l, dim=1)[0].reshape(BATCH_SIZE, -1)
            J_E = (J_E - Q_E).sum()

            # L2 loss
            J_L2 = torch.tensor(0.)
            for param in self.main_network.parameters():
                J_L2 += torch.norm(param)
            J_L2 *= 1e-5

            loss = J_DQ + J_n + J_E + J_L2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % TARGET_UPDATE_ITER == 0 and i != 0:
                self.update()
        self.update()
        print("END pretrain function")
