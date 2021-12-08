import operator
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy import ndarray
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import QNetwork

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # cpu runs to test


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
    memory_id: int
    state: torch.Tensor
    action: int
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    stack_rank: float = 0
    prob: float = 0
    used: int = 1

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

    def recompute_stack_ranks(self, agent_to_check_diff):
        """
        Recompute priority of the memories
        """
        # Recount stack ranks to avoid getting incorrect probabilities
        self.total_stack_rank = sum(memory.stack_rank for memory in self.memory_buffer)
        if self.total_stack_rank == 0:
            self.total_stack_rank = 1
        for memory in self.memory_buffer:
            if memory.used == 1:
                agent_to_check_diff.learn(memories_to_assume_diff=[memory])
                memory.prob = memory.stack_rank / self.total_stack_rank

    def process_buffer(self, memories_to_experience: [Memory]) -> None:
        """
        Function to adjust Total Stack rank of memory buffer and pop memories gathered for learning from the buffer
        :param memories_to_experience: Memories which should be deleted from memory buffer
        :return: None
        """
        ids_to_pop_out = [memory.memory_id for memory in memories_to_experience]
        self.memory_buffer = [memory for memory in self.memory_buffer if memory.memory_id not in ids_to_pop_out]

    def choose_memories(self, agent, sample_size: int = 20) -> [Memory]:
        """
        Sample memories, taking in account memories stack rank
        :param agent: Learning agent
        :param sample_size: How many samples should be taken from buffer
        :return sample ndarray of memories to learn
        """
        self.recompute_stack_ranks(agent)
        # Workaround for ValueError: probabilities do not sum to 1
        probabilities = np.array([memory.prob for memory in self.memory_buffer])
        probabilities /= probabilities.sum()
        memories_to_experience = np.random.choice(self.memory_buffer, size=sample_size, p=probabilities)
        self.process_buffer(memories_to_experience)
        return memories_to_experience

    def remember(self, rate_of_remember):
        """
        Drop memories with least stack rank values, do not overflow the buffer.
        :param: rate of remember: float - how much memory should be saved
        """
        self.memory_buffer = sorted(self.memory_buffer, key=operator.attrgetter("stack_rank"), reverse=True)
        data_to_cut = int(len(self.memory_buffer) * rate_of_remember)
        self.memory_buffer = self.memory_buffer[:data_to_cut]
        print(f'Forgetting {(1 - rate_of_remember) * 100}% of the least important memories.')


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
        priority_hyperparameter_initial: float
            an hyperparameter of update, to evaluate prioritized memories - initial value
        priority_hyperparameter_current: float
            an hyperparameter of update, to evaluate prioritized memories - momentary value


    """
    state_size: int
    action_size: int
    seed: int

    # Q-Network
    qnetwork_local: QNetwork = None
    qnetwork_target: QNetwork = None
    optimizer: optim.Adamax = None
    lr_scheduler: ReduceLROnPlateau = None
    # Replay memory
    priority_hyperparameter_current: float = 0
    priority_hyperparameter_initial: float = 0.4
    replay_memory: ReplayBuffer = ReplayBuffer()

    def __post_init__(self):
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed).to(DEVICE)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed).to(DEVICE)
        self.optimizer = optim.Adamax(self.qnetwork_local.parameters())
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10)

    def memorize(self, memory_id, state, action, reward, next_state, done):
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
        memory = Memory(memory_id=memory_id, state=state, action=action, reward=reward, next_state=next_state,
                        done=done)
        self.replay_memory.memory_buffer.append(memory)
        if len(self.replay_memory.memory_buffer) % 10 == 0 and len(self.replay_memory.memory_buffer) > 100:
            self.learn()
        return memory

    def act(self, state: ndarray, eps: float) -> ndarray:
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

    def learn(self, gamma=0.9, sampling_weight = 1, memories_to_assume_diff=None):
        already_appended_ids = []
        if not memories_to_assume_diff:
            memories = self.replay_memory.choose_memories(agent=self, sample_size=10)
        else:
            memories = memories_to_assume_diff
        for memory in memories:
            # Get max predicted Q value (for next states) from target model Use Double DQNs method. If two Models
            # agree on action, take any of them. If they don't agree on the action, Choose lower value,
            # to not overestimate the reward.
            Q_target_next = self.resolve_DQN_td_target(memory)

            # Compute Q targets for current states
            Q_target = memory.reward + (gamma * Q_target_next * (1 - memory.done.float()))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(memory.state).gather(0, memory.action).float()

            # Count diff and assign it to single and total stack ranks for prioritizing.
            diff = abs(float(Q_target - Q_expected))
            memory.stack_rank = diff
            memory.used += 1
            if not memories_to_assume_diff:
                if memory.memory_id not in already_appended_ids:
                    self.replay_memory.memory_buffer.append(memory)
                already_appended_ids.append(memory.memory_id)
                # Compute loss
                sampling_weight = (1 / len(self.replay_memory.memory_buffer)) * (1 / memory.prob)
                sampling_weight = pow(sampling_weight, self.priority_hyperparameter_current)
            loss = sampling_weight * F.mse_loss(Q_expected, Q_target)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # LR optimizer step
        # Copy parameters
        self.update_target_network()

    def update_target_network(self, tau=0.001):
        """
        Copy parameters to target network

        param: tau
            factor of intensity of params update
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def resolve_DQN_td_target(self, memory):
        """
        Basing on DQN principle, resolve expected reward

        :param memory: Memory object basing on which estimation has to be considered
        :return lower value of maxes between Local and Target network, considering most valuable action A in State S
        """
        considered_Q_target_next_target = self.qnetwork_target(memory.next_state).detach().max().float()
        considered_Q_target_next_local = self.qnetwork_local(memory.next_state).detach().max().float()
        if considered_Q_target_next_target != considered_Q_target_next_local:
            # Avoiding overestimation
            return min(considered_Q_target_next_target, considered_Q_target_next_local)
        else:
            return self.qnetwork_target(memory.next_state).detach().max().float()
