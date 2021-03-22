import numpy as np
import random
import util
import time
import sys

# Pacman game
from pacman import Directions
from game import Agent
import game

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf

params = {
    # Model backups
    'load_file': 'PACMAN',
    'save_file': 'PACMAN',
    'save_interval' : 10000, 

    # Training parameters
    'train_start': 5000,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 500000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0002,            # Learning reate
    'rms_decay': 0.99,      # RMS Prop decay
    'rms_eps': 1e-6,        # RMS Prop epsilon

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}                     

def build_network(params):
        Input_observation = tf.keras.layers.Input(shape = (params['width'] * params['height'], 3) )
        Input_pos = tf.keras.layers.Input(shape = (2,3))
        Input = tf.keras.layers.concatenate([Input_observation, Input_pos], axis = 1)
        layer = tf.keras.layers.Flatten()(Input)
        layer = tf.keras.layers.Dense(units = 512, activation = 'relu', kernel_initializer='he_normal')(layer)
        layer = tf.keras.layers.Dense(units = 512, activation = 'relu', kernel_initializer='he_normal')(layer)
        output = tf.keras.layers.Dense(units = 4, activation = 'linear' )(layer)
        model = tf.keras.models.Model([Input_observation,Input_pos], output)
        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(), metrics=["accuracy"])
        model.summary()
        return model

class PacmanDQN(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        #self.params['depth'] = args['depth']
        self.params['state_dim'] = (args['width'] * args['height'] , 3)
        self.params['pos_dim'] = (2, 3)
        self.params['numActions'] = 4
        self.params['num_training'] = args['numTraining']
        self.params['batch_size'] = 64

        self.qnet = build_network(self.params)

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0     

        # Stats
        if self.params['load_file'] is not None:
            self.cnt= hash(self.params['load_file'].split('_')[-1])
        else:
            self.cnt = 0

        self.local_cnt = 0
        self.gamma = 0.9
        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()
        
        self.i = 0


    def getMove(self, state):
        # Exploit / Explore
        
        if np.random.rand() > self.params['eps']:
            
            state = np.reshape(self.current_state,(1, self.params['width']* self.params['height'], 3))
            pos = np.reshape(self.current_pos, (1, 2, 3))
            self.Q_pred = self.qnet.predict([state, pos])

            self.Q_global.append(max(self.Q_pred))
            a_winner = np.argmax(self.Q_pred)

            move = self.get_direction(a_winner)
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST
            
    def observation_step(self, state):
        if self.last_action is not None:
            # Process current experience state
            self.last_state, self.last_pos = np.copy(self.current_state), np.copy(self.current_pos)
            self.current_state, self.current_pos = self.getStateMatrices(state)

            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.    # Eat ghost   (Yum! Yum!)
            elif reward > 0:
                self.last_reward = 10.    # Eat food    (Yum!)
            elif reward < -10:
                self.last_reward = -500.  # Get eaten   (Ouch!) -500
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # Punish time (Pff..)

            
            if(self.terminal and self.won):
                self.last_reward = 100.
            self.ep_rew += self.last_reward

            # Store last experience into memory 
            experience = (self.last_state, self.last_pos, float(self.last_reward), self.last_action, self.current_state, self.current_pos, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if(params['save_file']):
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    self.qnet.save(r"C:\Users\may32\Desktop\reinforcement\model" + str(self.cnt ))
                    print('Model saved')

            # Train
            self.train()
        else :
           self.registerInitialState( state)     
        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = min(self.params['eps_final'],self.local_cnt*float(self.params['eps_step']))
        
    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        #log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log','a')
        #log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         #(self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        #log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        #sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         #(self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        #sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        #sys.stdout.flush()
    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            minibatch = random.sample(self.replay_mem, self.params['batch_size'])
            state_ = np.zeros(((self.params['batch_size'],) + self.params['state_dim']))
            next_state = np.zeros(((self.params['batch_size'],) + self.params['state_dim']))
            action, reward, done = [], [], []
            next_pos = np.zeros(((self.params['batch_size'],)+self.params['pos_dim']))
            pos_ = np.zeros(((self.params['batch_size'],)+self.params['pos_dim']))
                                
            for i in range(self.params['batch_size']):
                state_[i] = minibatch[i][0]
                pos_[i] = minibatch[i][1]
                reward.append(minibatch[i][2])
                action.append(minibatch[i][3])
                next_state[i] = minibatch[i][4]
                next_pos[i] = minibatch[i][5]
                done.append(minibatch[i][6])

            target = self.qnet.predict([state_,pos_])
            target_next = self.qnet.predict([next_state, next_pos])
 
            for i in range(self.params['batch_size']):
                if done[i]:
                    target[i][int(action[i])] = reward[i]
                else:

                    target[i][int(action[i])] = reward[i] + self.gamma * (np.amax(target_next[i]))

            self.qnet.fit([state_,pos_], target, verbose = 1)

    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width),dtype=np.int)
            #matrix.dtype = int

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanPos(state):
            """ Return matrix with pacman coordinates set to 1 """
 

            #matrix.dtype = int

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition() 
            return pos

        def getGhostPos(state):
            """ Return matrix with ghost coordinates set to 1 """
  
            #matrix.dtype = int

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                    else :
                        pos = (0,0)
            return pos

        def getScaredPos(state):
            """ Return matrix with ghost coordinates set to 1 """

            
            #matrix.dtype = int

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                    else :
                        pos = (0,0)
            return pos

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width),dtype=np.int)
            #matrix.dtype = int

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width),dtype=np.int)
            #matrix.dtype = int

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((3, height*width))
        pos = np.zeros((3,2))
        
        observation[0] = getWallMatrix(state).flatten()
        observation[1] = getFoodMatrix(state).flatten()
        observation[2] = getCapsulesMatrix(state).flatten()

        pos[0] = getPacmanPos(state)
        pos[1] = getGhostPos(state)
        pos[2] = getScaredPos(state)
        
        observation = np.swapaxes(observation, 0, 1)
        pos = np.swapaxes(pos, 0,1)
        
        return observation, pos

    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state, self.current_pos = self.getStateMatrices(state)
        
        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP

        return move