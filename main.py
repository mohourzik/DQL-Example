import gym
from agent import Agent
import numpy as np

if __name__ == "__main__":
    ENV_NAME = "LunarLander-v2"

    env = gym.make(ENV_NAME)

    n_inputs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    scores, epsilons= [], []
    agent = Agent(n_inputs, n_actions)

    n_games = 500
    for i in range(n_games):
        state = env.reset()[0]
        score = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info, _ = env.step(action)
            score += reward
            agent.store_transition(state, action, reward, state_, done)
            agent.learn()
            state = state_
        scores.append(score)
        epsilons.append(agent.eps)

        avg_score = np.mean(scores[-100:])

        print(f"episode = {i}, score = {score:.2f}, avg_score = {avg_score:.2f}, eps = {agent.eps:.2f}")