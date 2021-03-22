# import pacman game 
from pacman import Directions
from pacmanUtils import *
from game import Agent
import game

# import torch library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#import other libraries
import random
import numpy as np
import time
from collections import deque

# model parameters
DISCOUNT_RATE = 0.95        # discount factor
LEARNING_RATE = 0.0005      # learning rate parameter
REPLAY_MEMORY = 50000       # Replay buffer 의 최대 크기
LEARNING_STARTS = 300 	    # 300 스텝 이후 training 시작
TARGET_UPDATE_ITER = 400   # update target network
BATCH_SIZE = 64

EPSILON_START = 0.8

    
class PacmanDQN(PacmanUtils):
    def __init__(self, args):        
        print("Started Pacman DQN algorithm")
        #print(args)
        self.double = args['double']
        self.multistep = args['multistep']
        self.n_steps = args['n_steps']

        self.trained_model = args['trained_model']
        if self.trained_model:
            mode = "Test trained model"
        else:
            mode = "Training model"

        self.numTraining = args['numTraining']
        
        print("=" * 100)
        print("Double : {}    Multistep : {}/{}steps    Train : {}    Test : {}    Mode : {}".format(
                self.double, self.multistep, self.n_steps, args['numTraining'], args['numTesting'], mode))
        print("=" * 100)

        # initialize DQNs
        self.replay_buffer = deque(maxlen=REPLAY_MEMORY)
        self.transition = deque(maxlen=self.n_steps)

        self.mainDQN = DQN()
        self.targetDQN = DQN()
        self.targetDQN.eval()

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.mainDQN.parameters(), lr=LEARNING_RATE)

        # copy mainDQN to targetDQN
        self.targetDQN.load_state_dict(self.mainDQN.state_dict())

        if self.multistep:
            self.discounts = torch.FloatTensor([DISCOUNT_RATE**i for i in range(self.n_steps+1)])


        # Target 네트워크와 Local 네트워크, epsilon 값을 설정
        if self.trained_model:  # Test
            self.epsilon = 0
            # load saved model
            self.mainDQN.load_state_dict(torch.load("./mainDQN_state_dict.pt"))
            self.mainDQN.eval()
        else:                   # Train
            self.epsilon = EPSILON_START  # epsilon init value
            if args['numTraining'] < 7000:
                self.epsilons = np.linspace(EPSILON_START, 0.01, args['numTraining']+1)
            else:
                self.epsilons = np.linspace(EPSILON_START, 0.01, 7000+1)

        # statistics
        self.win_counter = 0       # number of victory episodes
        self.steps_taken = 0       # steps taken across episodes
        self.steps_per_epi = 0     # steps taken in one episodes   
        self.episode_number = 0
        self.episode_rewards =[]  
        
        #self.epsilon = EPSILON_START  # epsilon init value

    
    def predict(self, state): 
        # state를 넣어 policy에 따라 action을 반환 (epsilon greedy)
        # Hint: network에 state를 input으로 넣기 전에 preprocessing 해야합니다.
        
        if random.random() > self.epsilon:
            state = self.preprocess(state)
            state = torch.FloatTensor(state)
            act = int(torch.argmax(self.mainDQN(state)))
        else:
            act = np.random.randint(0, 4)  # random value between 0 and 3
        self.action = act # save action
        return act
    
    def update_epsilon(self):
        # Exploration 시 사용할 epsilon 값을 업데이트
        try:
            self.epsilon = self.epsilons[self.episode_number]
        except:
            self.epsilon = self.epsilons[-1]
                 
    def step(self, next_state, reward, done):
        # next_state = self.state에 self.action 을 적용하여 나온 state
        # reward = self.state에 self.action을 적용하여 얻어낸 점수.

        # first step
        if self.action is None:
            self.state = self.preprocess(next_state)
        else:
            self.next_state = self.preprocess(next_state)

            if self.multistep:
                if len(self.transition) < self.n_steps:
                    self.transition.append((self.state, self.action, reward, self.next_state, done))
                else:
                    rewards = np.append(np.zeros(self.n_steps), [self.n_steps])
                    for i, (s, a, r, ns, d) in enumerate(self.transition):
                        if i == 0:
                            cur_state = s
                            cur_action = a
                            cur_next_state = ns
                            cur_done = d
                       
                        rewards[i] = r

                        if d == True:
                            rewards[i] = r
                            rewards[-1] = 0
                            self.replay_buffer.append((cur_state, cur_action, rewards, cur_next_state, cur_done))
                            break
                        elif i == self.n_steps - 1:
                            self.replay_buffer.append((cur_state, cur_action, rewards, cur_next_state, cur_done))

                    self.transition.append((self.state, self.action, reward, self.next_state, done))
            else:
                self.replay_buffer.append((self.state, self.action, reward, self.next_state, done))

            self.state = self.next_state
        
        # next
        self.episode_reward += reward
        self.steps_taken += 1
        self.steps_per_epi += 1
        

        if(self.trained_model == False):
            self.train()
            self.update_epsilon()
            if(self.steps_taken % TARGET_UPDATE_ITER == 0):
                # UPDATING target network
                self.targetDQN.load_state_dict(self.mainDQN.state_dict())

            if self.numTraining == self.episode_number:
                torch.save(self.mainDQN.state_dict(), "./mainDQN_state_dict.pt")
        
		
    def train(self):
        # replay_memory로부터 mini batch를 받아 policy를 업데이트

        if (self.steps_taken > LEARNING_STARTS):
            minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

            states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))
                
            states = torch.FloatTensor(states)
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.tensor(dones)
            
            print("rewards: ", rewards)

            if self.double:
                if self.multistep:      # multistep DDQN
                    next_action = torch.argmax(self.mainDQN(next_states).detach().squeeze(), dim=1).view(-1, 1)
                    DQN_target = self.targetDQN(next_states).squeeze().gather(1, next_action)
                    #print("rewards: ", rewards)
                    indices = rewards[:, -1].long()
                    #print("indices: ", indices)
                    
                    for i, reward in enumerate(rewards):
                        if indices[i] == self.n_steps:
                            reward[indices[i]] = (~dones[i]) * DQN_target[i]
                        else:
                            reward[-1] = 0

                    DQN_target = torch.sum(self.discounts * rewards, 1).unsqueeze(1)
                    Q_hat = self.mainDQN(states).squeeze().gather(1, actions)

                    loss = self.loss(DQN_target, Q_hat)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:                   # DDQN
                    next_action = torch.argmax(self.mainDQN(next_states).detach().squeeze(), dim=1).view(-1, 1)
                    rewards += (~dones) * DISCOUNT_RATE * self.targetDQN(next_states).squeeze().gather(1, next_action).squeeze()
                    
                    DQN_target = rewards.unsqueeze(1)
                    Q_hat = self.mainDQN(states).squeeze().gather(1, actions)
                    
                    loss = self.loss(DQN_target, Q_hat)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:                       # DQN
                rewards += (~dones) * DISCOUNT_RATE * self.targetDQN(next_states).squeeze().max(1)[0]

                DQN_target = rewards.unsqueeze(1)
                Q_hat = self.mainDQN(states).squeeze().gather(1, actions)
                
                loss = self.loss(DQN_target, Q_hat)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def reset(self):
        # 새로운 episode 시작시 불러 오는 함수.

        self.last_score = 0
        self.current_score = 0
        self.episode_reward = 0

        self.episode_number += 1
        self.steps_per_epi = 0
    
    def final(self, state):
        # epsidoe 종료시 불려오는 함수. 수정할 필요 없음.
        done = True
        reward = self.getScore(state)
        if reward >= 0: # not eaten by ghost when the game ends
            self.win_counter +=1

        self.step(state, reward, done)
        self.episode_rewards.append(self.episode_reward)
        win_rate = float(self.win_counter) / 500.0
        avg_reward = np.mean(np.array(self.episode_rewards))
		# print episode information
        if(self.episode_number%500 == 0):
            print("Episode no = {:>5}; Win rate {:>5}/500 ({:.2f}); average reward = {:.2f}; epsilon = {:.2f}".format(self.episode_number,
                                                                    self.win_counter, win_rate, avg_reward, self.epsilon))
            self.win_counter = 0
            self.episode_rewards= []

    def preprocess(self, state):
        # pacman.py의 Gamestate 클래스를 참조하여 state로부터 자유롭게 state를 preprocessing 해보세요.

        result = np.zeros((3, 7, 7))
        
        # Pacman state
        pacman_x, pacman_y = state.getPacmanPosition()
        result[0, pacman_x, pacman_y] = 1

        # Ghost state
        ghost_x, ghost_y = map(int, state.getGhostPositions()[0])
        result[1, ghost_x, ghost_y] = 1
        
        # Food
        for x in range(7):
            for y in range(7):
                if state.hasFood(x, y):
                    result[2, x, y] = 1
        
        result = result.reshape(-1, 1)

        # Cpasule
        capsule = np.array(len(state.getCapsules())).reshape(-1 ,1)
        result = np.concatenate((result, capsule))

        # Timer
        scaredTimer = np.array([[state.getScaredTimer()]]) / 40
        result = np.concatenate((result, scaredTimer))
        result = result.reshape(1, -1)

        return result

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_features=149, out_features=32)
        self.layer2 = nn.Linear(in_features=32, out_features=4)

        self.relu = nn.ReLU()

        self.layers = nn.Sequential(self.layer1, self.relu, self.layer2)
    
    def forward(self, x):
        return self.layers(x)