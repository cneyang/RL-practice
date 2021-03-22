import numpy as np
import copy


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)

    delta = 100
    while delta > theta:
        delta = 0

        # for all state (16 states)
        for s in range(len(env.MDP)):
            v = V[s]    # copy value of the value function given state s
            sigma1 = 0

            # for all action (LEFT, DOWN, RIGHT, UP)
            for a in range(4):
                pi = policy[s][a]   # policy of a given state s
                q = env.MDP[s][a]   # state action function given s and a

                sigma2 = 0
                # for all action available
                for i in range(len(q)):
                    sigma2 += q[i][0] * (q[i][2] + gamma*V[q[i][1]])

                sigma1 += pi * sigma2

            V[s] = sigma1

            delta = np.maximum(delta, np.abs(v - V[s]))

    return V


def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA

    # for all state
    for s in range(len(env.MDP)):
        argmax_a = 0
        max_q = 0

        # for all action
        for a in range(4):
            q = env.MDP[s][a]
            sigma = 0

            for i in range(len(q)):
                sigma += q[i][0] * (q[i][2] + gamma*V[q[i][1]])

            if sigma > max_q:
                max_q = sigma
                argmax_a = a

        policy[s][argmax_a] = 1

    return policy


def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    policy_stable = False
    while not policy_stable:
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)

        policy_stable = True
        old_policy = copy.deepcopy(policy)
        policy = policy_improvement(env, V, gamma=gamma)

        if not np.array_equal(old_policy, policy):
            policy_stable = False

    return policy, V


def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA

    delta = 100
    while delta > theta:
        delta = 0

        for s in range(len(env.MDP)):
            v = V[s]
            max_a = 0

            for a in range(4):
                q = env.MDP[s][a]
                sigma = 0

                for i in range(len(q)):
                    sigma += q[i][0] * (q[i][2] + gamma*V[q[i][1]])

                if sigma > max_a:
                    max_a = sigma

            V[s] = max_a

            delta = np.maximum(delta, np.abs(v - V[s]))

    for s in range(len(env.MDP)):
        argmax_a = 0
        max_q = 0

        for a in range(4):
            q = env.MDP[s][a]
            sigma = 0

            for i in range(len(q)):
                sigma += q[i][0] * (q[i][2] + gamma * V[q[i][1]])

            if sigma > max_q:
                max_q = sigma
                argmax_a = a

        policy[s].fill(0.0)
        policy[s][argmax_a] = 1

    return policy, V
