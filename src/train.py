from gymnasium.wrappers import TimeLimit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from env_hiv import HIVPatient

import pickle as pkl
import numpy as np

import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False, logscale=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ProjectAgent:

    def __init__(self):
        self.path = "src/model"
        self.gamma = 0.98
        self.Q = ExtraTreesRegressor()

        

    def act(self, observation, use_random=False):
        Qsa = []
        for a in range(self.n_action):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        return np.argmax(Qsa)

    def collect_samples(self, env, horizon, exp = 0):
        s, _ = env.reset()
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in range(horizon):
            if exp == 0:
                a = env.action_space.sample()
            else:
                p = np.random.rand()
                if p < 0.15:
                    a = env.action_space.sample()
                else:
                    a = self.act(s)

            s2, r, done, trunc, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D
    def train(self, env, n_iterations, n_exp, horizon= 30 * 200):
        self.state_dim = env.observation_space.shape[0]
        self.n_action = env.action_space.n
        for exp in range(n_exp):
            S, A, R, S2, D = self.collect_samples(env, exp=exp, horizon=horizon)
            nb_samples = S.shape[0]
            SA = np.append(S,A,axis=1)
            for iter in range(n_iterations):
                if iter==0:
                    value=R.copy()
                else:
                    Q2 = np.zeros((nb_samples,self.n_action))
                    for a2 in range(self.n_action):
                        A2 = a2*np.ones((S.shape[0],1))
                        S2A2 = np.append(S2,A2,axis=1)
                        Q2[:,a2] = self.Q.predict(S2A2)
                    max_Q2 = np.max(Q2,axis=1)
                    value = R + self.gamma*(1-D)*max_Q2

                Q = ExtraTreesRegressor()
                Q.fit(SA,value)
                self.Q = Q
    


    def save(self, path):
        payload = {}
        payload["Q"] = self.Q
        payload["nb_action"] = self.n_action
        with open("model", "wb") as f:
            pkl.dump(payload, f)
        

    def load(self):
        path = os.path.join(os.getcwd(), os.path.abspath(self.path))
        with open(path, "rb") as f:
            payload = pkl.load(f)
            self.Q = payload["Q"]
            self.n_action = payload["nb_action"]

