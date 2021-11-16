from abc import ABC

import numpy
import torch
import torch.nn as nn


class QNetwork(nn.Module, ABC):
    """
    QNetwork Model for the banana agent
        state_size: int
            shape of the state of the agent
        action_size: int
            size of the action space of the agent
        seed: int
            random seed for the networkd
    """

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, fc3_units=64):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, fc3_units),
            nn.ReLU(),
            nn.Linear(fc3_units, action_size)
        )

    def forward(self, state):
        """
        Propagate data forward the Q-Network
        """
        state = state.float()
        return self.layers(state)
