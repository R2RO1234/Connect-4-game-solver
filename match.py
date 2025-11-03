# For our model to play vs other algo
# match.py
from logic import Connect4
from neural_network import DQNAgent
import torch as th
import random

def play_models(agent1, agent2, num_games=20):
    wins = {'agent1':0, 'agent2':0, 'draws':0}
    for g in range(num_games):
        game = Connect4()
        state = game.reset()
        # randomize who starts: 0->agent1 starts, 1->agent2 starts
        if random.random() < 0.5:
            current = agent1
            current_name = 'agent1'
        else:
            current = agent2
            current_name = 'agent2'
        while True:
            valid = game.get_valid_moves()
            action = current.select_action(state, valid)
            state, reward, done = game.make_move(action)
            if done != 0:
                if reward == 1:
                    wins[current_name] += 1
                else:
                    wins['draws'] += 1
                break
            # switch current
            if current is agent1:
                current = agent2
                current_name = 'agent2'
            else:
                current = agent1
                current_name = 'agent1'
    print(wins)
    return wins

if __name__ == "__main__":
    # load trained DQN agent (adjust path if saved differently)
    dqn = DQNAgent(input_shape=(2,6,7), move_count=7)
    # dqnBetter = DQNAgent(input_shape=(2,6,7), move_count=7)
    try:
        dqn.policy_net.load_state_dict(th.load("trained_agent.pt"))
        dqn.policy_net.eval()
    except Exception as e:
        print("Could not load trained_agent.pt:", e)
        # If you don't have a saved model, train or point to existing file.
    # play_models(dqn, heuristic, num_games=20)
    