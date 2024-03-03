from memory import ReplayBuffer
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from nn_models import DDDQNNet
from copy import deepcopy

import numpy as np
import pickle as pkl
import torch
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
        self.model_path = '../models/model.pt'
        self.payload_path = '../models/payload.binary'
        self.device = 'cpu'

    def act(self, observation, use_random = False):
        if use_random:
            return np.random.choice(range(self.n_actions))
        else:
            with torch.no_grad():
                Q = self.network(torch.Tensor(observation).unsqueeze(0).to(self.device))
                return torch.argmax(Q).item()
    
    def save(self, path):
        pass

    def load(self):
        # loading payload containing infos abt the network architecture
        with open(os.path.join(os.getcwd(),self.payload_path), 'rb') as file:
            payload = pkl.load(file)
            self.n_states = payload["n_states"]
            self.n_actions = payload["n_actions"]
            self.n_hidden = payload["n_hidden"]

        # instantiating the newtork and its state dict
        self.network = DDDQNNet(self.n_states, self.n_hidden, self.n_actions)
        self.network.load_state_dict(torch.load(os.path.join(os.getcwd(),self.model_path)))

        # ensuring it's on cpu
        self.network.to(self.device)

class DDDQNAgent:
    def __init__(self, env, config):
        # environment and device
        self.env = env
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # network dimensions
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.hidden_size = config["hidden_size"]

        # instantiating the network
        self.network = DDDQNNet(self.state_size, self.hidden_size, self.action_size).to(self.device)

        # defining backpropagations necessary thingies
        self.learning_rate = config["learning_rate"]
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.nb_gradient_steps = config["gradient_steps"]
        self.batch_size = config["batch_size"]

        # defining the target network
        self.target_network = deepcopy(self.network).to(self.device)
        self.update_target_freq = config["update_target_freq"]
        self.update_target_strategy = config["update_target_strategy"]
        self.update_target_tau = config["update_target_tau"]

        # defining replay buffer
        self.memory = ReplayBuffer(config["buffer_size"], self.device)

        # defining action exploration ctes
        self.epsilon_max = config["epsilon_max"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_stop = config['epsilon_decay_period'] 
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop

        # define qlearning cte
        self.gamma = config["gamma"]

        # defining logging nb trials to use
        self.monitoring_nb_trials = config["monitoring_nb_trials"]

        # save payload and model path
        self.model_path = '../models/model.pt'
        self.payload_path = '../models/payload.binary'

    def greedy_action(self, state):
        with torch.no_grad():
            Q = self.network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    def save(self):
        payload = {
            'n_states':self.state_size,
            'n_hidden':self.hidden_size,
            'n_actions': self.action_size,

        }
        with open(self.payload_path, 'wb') as file:
            pkl.dump(payload, file)
        torch.save(self.network.state_dict(), self.model_path)


    def MC_eval(self, nb_trials):
        MC_total_rewards = []
        MC_discounted_rewards = []

        for _ in range(nb_trials):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            discounted_reward = 0
            step = 0

            while not done:
                action = self.greedy_action(self.model, state)
                next_state, reward, done, _ = self.env.step(action)
                
                total_reward += reward
                discounted_reward += self.gamma ** step * reward
                step += 1

                state = next_state

            MC_total_rewards.append(total_reward)
            MC_discounted_rewards.append(discounted_reward)

        mean_discounted_reward = np.mean(MC_discounted_rewards)
        mean_total_reward = np.mean(MC_total_rewards)

        return mean_discounted_reward, mean_total_reward

    def V_initial_state(self, nb_trials):   
        initial_values = []

        with torch.no_grad():
            for _ in range(nb_trials):
                state, _ = self.env.reset()
                state_tensor = torch.Tensor(state).unsqueeze(0).to(self.device)
                value = self.network(state_tensor).max().item()
                initial_values.append(value)

        mean_initial_value = np.mean(initial_values)
        return mean_initial_value

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            # Sample experiences from memory
            samples = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = samples

            # Compute target Q-values using the target model
            with torch.no_grad():
                next_max_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_max_q_values

            # Compute current Q-values for the sampled actions
            current_q_values = self.network(states).gather(1, actions.long().unsqueeze(1))

            # Compute the loss between current and target Q-values
            loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

            # Perform backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, max_episode):
        episode_returns = []
        MC_avg_total_rewards = []
        MC_avg_discounted_rewards = []
        V_init_states = []

        episode = 0
        best_reward = 0
        episode_cum_reward = 0
        state, _ = self.env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # Update epsilon
            if step > self.epsilon_delay:
                if 50 < episode and episode < 80:
                    epsilon = 0.1
                elif episode > 80:
                    epsilon = 0.05
                else:
                    epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            # Select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.greedy_action(state)

            # Step
            next_state, reward, done, trunc, _ = self.env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # Update target network if needed
            if self.update_target_strategy == 'replace' and step % self.update_target_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            if self.update_target_strategy == 'ema':
                model_state_dict = self.network.state_dict()
                target_state_dict = self.target_network.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau * model_state_dict[key] + (1 - tau) * target_state_dict[key]
                self.target_network.load_state_dict(target_state_dict)

            # Next transition
            step += 1

            if done or trunc:
                episode += 1

                # Monitoring
                if self.monitoring_nb_trials > 0:
                    MC_discounted_reward, MC_total_reward = self.MC_eval(self.monitoring_nb_trials)
                    if MC_total_reward > best_reward:
                        best_reward = MC_total_reward
                        self.save()
                    V0 = self.V_initial_state(self.monitoring_nb_trials)
                    MC_avg_total_rewards.append(MC_total_reward)
                    MC_avg_discounted_rewards.append(MC_discounted_reward)
                    V_init_states.append(V0)
                    episode_returns.append(episode_cum_reward)
                    print(f"Episode {episode:2d}, epsilon {epsilon:6.2f}, batch size {len(self.memory):4d}, "
                        f"ep return {episode_cum_reward:4.1f}, MC tot {MC_total_reward:6.2f}, "
                        f"MC disc {MC_discounted_reward:6.2f}, V0 {V0:6.2f}")
                else:
                    episode_returns.append(episode_cum_reward)
                    print(f"Episode {episode:2d}, epsilon {epsilon:6.2f}, batch size {len(self.memory):4d}, "
                        f"ep return {episode_cum_reward:e}")

                state, _ = self.env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
            

        return episode_returns, MC_avg_discounted_rewards, MC_avg_total_rewards, V_init_states


# setting up configuration
config = {
    'hidden_size' : 300,
    'batch_size'  : 1024,
    'gradient_steps' : 10,
    'learning_rate' : 2.5e-3,
    'update_target_freq' : 100,
    'update_target_tau' : 0.05,
    'update_target_strategy' : 'replace',
    'epsilon_min' : 0.2,
    'epsilon_max' : 1,
    'epsilon_delay_decay' : 200,
    'epsilon_decay_period'  : 2000,
    'gamma' : 0.98,
    'alpha' : 0.5,
    'buffer_size' : int(10e20),
    'monitoring_nb_trials': 0,
}

# training the DDDQN Agent
# trained_agent = DDDQNAgent(env, config)
# trained_agent.train(max_episode=200)
