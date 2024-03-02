import torch.nn as nn
import torch.nn.functional as F

class DDDQNNet(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(DDDQNNet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # First linear layer
        self.linear1 = nn.Linear(state_size, hidden_size)

        # Second linear layer
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        # Third linear layer
        self.linear3 = nn.Linear(hidden_size, 2 * hidden_size)

        # Fully connected layers for value stream
        self.value_fc = nn.Linear(2 * hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

        # Fully connected layers for advantage stream
        self.advantage_fc = nn.Linear(2 * hidden_size, hidden_size)
        self.advantage = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        # Forward pass through first linear layers
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        # Forward pass through value and advantage streams
        value = F.relu(self.value_fc(x))
        value = self.value(value)

        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)

        # Combine value and advantage streams to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
