import operator
import random
import statistics
from dataclasses import dataclass, field
from statistics import mean

import numpy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from model import QNetwork

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # cpu runs to test


@dataclass
class ReplayBuffer:
    """
    Class representing ReplayBuffer, containing objects of Memory type, used for Experience Replay of the Q-Network

    Attributes:
        memory_buffer: []
            array of all Memory objects
        total_stack_rank: float
            sum of all TD Errors to compute stack of given memory
        smoothing_factor: float
            factor to avoid division by 0
    """
    memory_buffer: [] = field(default_factory=list)
    total_stack_rank: float = 1.0
    smoothing_factor = 1e-8

    def _recompute_stack_ranks(self):
        """
        Recompute priority of the memories
        """
        for memory in self.memory_buffer:
            memory.stack_rank += self.smoothing_factor
            memory.prob = memory.stack_rank / self.total_stack_rank

    def choose_memories(self):
        """
        Choose 10 memories, taking in account memories stack rank
        """
        self._recompute_stack_ranks()
        self.memory_buffer = sorted(self.memory_buffer, key=operator.attrgetter("prob"), reverse=True)
        probabilities = sorted([memory.prob if memory.prob > 0 else (1.0 / len(self.memory_buffer)) for memory in
                                self.memory_buffer], reverse=True)
        memories_to_experience = []
        # Workaround for ValueError: probabilities do not sum to 1 - https://github.com/numpy/numpy/issues/6123
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        # Now we can proceed
        while len(memories_to_experience) != 5:
            memory_to_experience = numpy.random.choice(self.memory_buffer, p=probabilities)
            memories_to_experience.append(memory_to_experience)
        return memories_to_experience

    def forget(self, rate_of_forget):
        f"""
        Forget some of the memory buffer data, after sorting through the stack values.

        param: rate of forget: float - how much memory should be saved
        """
        self.memory_buffer = sorted(self.memory_buffer, key=operator.attrgetter("stack_rank"), reverse=True)
        data_to_cut = int(len(self.memory_buffer) * rate_of_forget)
        self.memory_buffer = self.memory_buffer[:data_to_cut]
        self.total_stack_rank = sum(memory.stack_rank for memory in self.memory_buffer) + 1
        print(f'Forgetting {(1 - rate_of_forget) * 100}% of least important memories.')


@dataclass(order=True)
class Memory:
    """
    Class representing simple Memory object

    Attributes:
        state: int
            State representation of the agent (s)
        action: int
            Executed action (a)
        reward: int
            Reward from the action (a) in state (s)
        next_state: int
            State after taking action (a) in state (s)
        done: int
            If agent has done the episode
        stack_rank: float
            A difference between new reward and currently updated reward, to prioritize the memory
        probability: float
            probability of memory to be chosen for learning

    """
    state: torch.Tensor
    action: int
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    stack_rank: float = 0
    prob: float = 0

    def __post_init__(self):
        """
        Cast objects to Tensors
        """
        tensors = list(
            map(lambda x: torch.from_numpy(np.array(x)), (self.state, self.action, self.reward, self.next_state)))
        self.state = tensors[0].float()
        self.action = tensors[1].long()
        self.reward = tensors[2].float()
        self.next_state = tensors[3].float()
        done = np.array(self.done).astype(np.uint8)
        self.done = torch.from_numpy(done)


@dataclass
class Agent:
    """
    Agent interacting with Banana environment

    Attributes:
        state_size: int
            shape of the state_size
        action_size: int
            a number of possible action
        seed: int
            random seed for replay buffer
        qnetwork_local: QNetwork
            local Q-Network changing weights every step
        qnetwork_target: QNetwork
            target Q-Network which should be changed every some time
        optimizer: Adamax
            optimizer for Q-Networks
        lr_scheduler: ReduceLROnPlateu
            LR scheduler used for optimal change of optimzer's LR
        priority_hyperparameter: float
            an hyperparameter of update, to evaluate prioritized memories


    """
    state_size: int
    action_size: int
    seed: int

    # Q-Network
    qnetwork_local: QNetwork = None
    qnetwork_target: QNetwork = None
    optimizer: optim.Adam = None
    lr_scheduler: ExponentialLR = None
    # Replay memory
    replay_memory: ReplayBuffer = ReplayBuffer()
    priority_hyperparameter: float = 0.5
    losses = []
    decay_lr_mean_check = 10
    checks = 0

    def __post_init__(self):
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed).to(DEVICE)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.02)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def memorize(self, state, action, reward, next_state, done):
        """
        Add memories to the Replay Buffer

        Attributes:
            state: int
                State of the agent
            action: int
                Action taken by the agent
            reward:int
                Reward from the action
            next_state:int
                Next state of the agent
            done: bool
                If Agent has done the episode
        """
        memory = Memory(state, action, reward, next_state, done)
        self.replay_memory.memory_buffer.append(memory)
        if len(self.replay_memory.memory_buffer) % 4 == 0:
            self.learn()

    def act(self, state, eps):
        """Get action basing on current state in the given policy state
        """
        state = torch.from_numpy(state).float().to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, gamma=0.99):
        memories = self.replay_memory.choose_memories()
        for memory in memories:
            # Get max predicted Q value (for next states) from target model Use Double DQNs method. If two Models
            # agree on action, take any of them. If they don't agree on the action, Choose lower value,
            # to not overestimate the reward.
            Q_target_next = self._resolve_DQN_td_target(memory)

            # Compute Q targets for current states
            Q_target = memory.reward + (gamma * Q_target_next * (1-memory.done.float()))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(memory.state).gather(0, memory.action).float()

            # Count diff and assign it to single and total stack ranks for prioritizing
            diff = abs(float(Q_target - Q_expected))
            memory.stack_rank = diff
            self.replay_memory.total_stack_rank += diff

            # Compute loss
            sampling_weight = (1 / len(self.replay_memory.memory_buffer)) * (1 / memory.prob)
            sampling_weight = pow(sampling_weight, self.priority_hyperparameter)
            loss = sampling_weight * F.mse_loss(Q_expected, Q_target)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.losses.append(float(loss))
            self.optimizer.step()
            # LR optimizer step
        try:
            last_loss_mean = mean(self.losses[-self.decay_lr_mean_check:])
            previous_loss_mean = mean(self.losses[-(2 * self.decay_lr_mean_check):-self.decay_lr_mean_check])
            if not last_loss_mean < previous_loss_mean:
                self.checks += 1
                if self.checks > 9:
                    print('Decaying LR')
                    self.lr_scheduler.step()
                    print(self.optimizer.param_groups[0]['lr'])
                    self.checks = 0
            else:
                self.checks = 0
        except (IndexError, statistics.StatisticsError):
            pass
        # Copy parameters
        self.update_target_network()

    def update_target_network(self, tau=1e-3):
        """
        Copy parameters to target network

        param: memory
            memory to get probability of
        param: tau
            smoothing factor
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def _resolve_DQN_td_target(self, memory):
        """
        Basing on DQN principle, resolve expected reward

        memory: State, Action, Reward, Next State, Done basing on which estimation has to be considered
        """
        Q_target_next_target = self.qnetwork_target(memory.next_state).detach().max().float()
        Q_target_next_local = self.qnetwork_local(memory.next_state).detach().max().float()
        if Q_target_next_target != Q_target_next_local:
            # Avoiding overestimation
            return min(Q_target_next_target, Q_target_next_local)
        else:
            return self.qnetwork_target(memory.next_state).detach().max().float()
