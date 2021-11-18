import configparser
import os
import statistics
from statistics import mean

import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment

from Agent import Agent

# Get Configs from config.ini
config = configparser.ConfigParser()
config.read('./config.ini')
env_path = config.get('config', 'env_path')
model_weights_filename = config.get('config', 'model_weights_filename')
checkpoint_filename = config.get('config', 'checkpoint_filename')

env = UnityEnvironment(file_name="C:\\Users\\Damian\\BananaCatcher\\Banana_Windows_x86_64\\Banana.exe")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment

# number of actions
action_size = brain.vector_action_space_size
# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)

env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
state = env_info.vector_observations[0]  # get the current state
score = 0  # initialize the score
agent = Agent(action_size=action_size, state_size=state_size, seed=42)  # Create agent object
num_of_episodes = 1800  # Assume number of episodes

scores = []  # Scores array for plotting
over_13_counter = 0  # Counter to check when the model will achieve score over 13 for 100 episodes


def resolve_backup(agent):
    """
    Resolve backup save and load of the model

    :param agent: Created Agent
    """
    if os.path.isfile(model_weights_filename):
        agent.qnetwork_local.load_state_dict(torch.load(model_weights_filename))
        agent.qnetwork_target.load_state_dict(torch.load(model_weights_filename))
        print('Backup loaded')
        torch.save(agent.qnetwork_target.state_dict(), f"{model_weights_filename}_backup")


def evaluate_episode(episode_epsilon, episode_initial_state):
    """
    Do episode of enviroment
    :param episode_epsilon: Epsilon for episode
    :param episode_initial_state: Initial state of episode
    :return: Tuple of agent state in episode - state,action,reward,next_state,done
    """
    episode_action = agent.act(episode_initial_state, eps=episode_epsilon).astype(int)  # select an action
    episode_env_info = env.step(episode_action)[brain_name]  # send the action to the environment
    episode_next_state = episode_env_info.vector_observations[0]  # get the next state
    episode_reward = episode_env_info.rewards[0]  # get the reward
    episode_done = episode_env_info.local_done[0]  # see if episode has finished
    return episode_initial_state, episode_action, episode_reward, episode_next_state, episode_done


def process_score(score, scores, over_13_counter):
    """
    :param score: Score for current episode
    :param scores: All scores
    :param over_13_counter: number of consecutive episodes with score over 13
    :return: tuple of scores and over_13_counter after resolving the scores
    """
    if score >= 13.0:
        over_13_counter += 1
        if over_13_counter > 100:
            print(f"Achieved over 13 scores for 100 consecutive episodes at episode {episode}")
            exit()
    else:
        over_13_counter = 0

    if len(scores) > 101:
        print(f"Average scores of last 100 episodes {mean(scores[-100:])}")
    print(f"Score: {score} for episode {episode}")
    scores.append(score)
    return scores, over_13_counter


def evaluate_model(num_of_evaluated_episodes, state_for_evaluation):
    """
    Evaluate the model with 0 epsilon, to check how model do without randomness
    :param num_of_evaluated_episodes: For how many episodes should model be evaluated
    :param state_for_evaluation: Start state for evaluation
    """
    print(f'Evaluate for {num_of_evaluated_episodes}')
    eval_scores = []
    for eval_episode in range(1, num_of_evaluated_episodes):
        eval_score = 0
        eval_epsilon = 0
        while True:
            '''
            Resolve env info
            '''
            _, _, eval_reward, eval_next_state, eval_done = evaluate_episode(eval_epsilon, state_for_evaluation)
            state_for_evaluation = eval_next_state
            eval_score += eval_reward
            if eval_done:  # exit loop if episode finished
                eval_scores.append(eval_score)
                print(f"Evaluation score is {eval_score}")
                env.reset(train_mode=True)
                break
    print(f"Mean of evaluation of 10 episodes is {mean(eval_scores)}")


# If saved model weights are found load them
resolve_backup(agent)
for episode in range(1, num_of_episodes):
    backup_and_hyperparameters_decay_step = 5  # forget every this step
    epsilon = 1 - (episode / num_of_episodes)
    epsilon = max(0.05, epsilon)
    score = 0
    decay_lr = int(num_of_episodes/6)
    if episode % backup_and_hyperparameters_decay_step == 0:
        '''
        Resolve parameters decays.
        
        Also resolve backup.
        '''
        print('Decaying')
        agent.replay_memory.forget(0.2)
        agent.priority_hyperparameter = agent.priority_hyperparameter + (
                (1 - agent.priority_hyperparameter) * (episode / num_of_episodes))
    if episode % decay_lr == 0:
        #agent.lr_scheduler.step()
        print(agent.lr_scheduler.get_lr())
        torch.save(agent.qnetwork_target.state_dict(), checkpoint_filename)
        print('Saving.')
        plt.plot(scores)
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.savefig('scores.png')
    while True:
        '''
        Resolve env info
        '''
        epsilon *= 0.6
        epsilon = max(epsilon, 0.05)
        state, action, reward, next_state, done = evaluate_episode(epsilon, state)
        agent.memorize(state=state, action=action, reward=reward, next_state=next_state, done=done)
        state = next_state
        score += reward
        if done:  # exit loop if episode finished
            scores, over_13_counter = process_score(score, scores, over_13_counter)
            env.reset(train_mode=True)
            break
env.close()
plt.plot(scores)
plt.ylabel('Score')
plt.xlabel('Episode')
plt.savefig('scores.png')
torch.save(agent.qnetwork_target.state_dict(), model_weights_filename)
plt.show()
