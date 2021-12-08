from abc import ABC

import torch
import torch.nn as nn
from numpy import ndarray


class QNetwork(nn.Module, ABC):
    """
    QNetwork Model for the banana agent
        state_size: int
            shape of the state of the agent
        action_size: int
            size of the action space of the agent
        seed: int
            random seed for the network
        fc1_units:
            units for first, nn.Linear layer
        fc2_units:
            units for second, nn.Linear layer
        fc3_units:
            units for third, nn.Linear layer
        fc4_units:
            units for fourth, nn.Linear Layer
    """

    def __init__(self, state_size: int, action_size: int, seed: int, fc1_units=512, fc2_units=256, fc3_units=128, fc4_units=64):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(),
            nn.Linear(fc3_units, fc4_units),
            nn.ReLU(),
            nn.Linear(fc4_units, action_size),
        )

    def forward(self, state: ndarray):
        """
        Propagate data forward the Q-Network

        :param state: state to propagate into Network
        """
        return self.layers(state)
