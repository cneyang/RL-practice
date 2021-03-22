import numpy as np

class Agent:

    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.trajectory = []
        self.k = 1
        self.gamma = 0.99
        self.alpha = 0.1
        self.eps = 0.1
        self.mc_eps = np.linspace(0.5, 0.001, 100000)


    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if self.mode == "test_mode":
            return np.argmax(self.Q[state])
        elif self.mode == "mc_control":
            self.eps = self.mc_eps[self.k - 1]
        elif self.mode == "q_learning":
            self.eps = 1/self.k

        if np.random.rand() > self.eps:
            a = np.argmax(self.Q[state])
            return a
        else:
            return np.random.choice(self.n_actions)


    def step(self, state, action, reward, next_state, done):
        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if self.mode == "mc_control":
            if not done:
                self.trajectory.append((state, action, reward))

            if done:
                self.k += 1
                
                for i, s_a_r in enumerate(self.trajectory):
                    s, a, r = s_a_r
                    first_idx = next(i for i, x in enumerate(self.trajectory) if x[0] == s and x[1] == a)
                    G = sum([x[2]*(self.gamma**i) for i, x in enumerate(self.trajectory[first_idx:])])
                    self.Q[s][a] = self.Q[s][a] + (1/10)*self.eps * (G - self.Q[s][a])
                
                self.trajectory = []

        elif self.mode == "q_learning":
            max_a = np.max(self.Q[next_state])

            self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma*max_a - self.Q[state][action])
            
            if done:
                self.k += 1
