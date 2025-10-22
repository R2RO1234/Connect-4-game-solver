from logic import Connect4

import numpy as np
import matplotlib.pyplot as plt
import random
import torch as th
import torch.nn as nn
import torch.optim as optim
from collections import deque   # NOTE: deque is a double-ended queue where you can add/remove from both ends of the list


class DQN(nn.Module): # Base class for all neural networks in PyTorch (nn.Module = parent class of DQN)
    
    def __init__(self, input_shape: int, move_count: int):
        super(DQN, self).__init__()   # Calls parent class of DQN (nn.Module) (Initializes self as nn.Module object?)

        self.input_shape = input_shape
        self.move_count = move_count

        self.flatten = nn.Flatten() # Associates the nn.Flatten() function with self.flatten (to be used later in forward function). 
                                    # Note: It does not actually change the input np.shape at initialization.

        # NOTE: Here assigns FUNCTIONS to these attributes (to be used later), not the results of calling those functions!
        # Also NOTE: calling nn.Linear() automatically initializes the random weights in PyTorch. 
        self.fc1 = nn.Linear(84, 256)   # First hidden layer of the NN with 256 neurons (Note: first argument connects our size 84 input vector to the first hidden layer)

        self.fc2 = nn.Linear(256, 128)  # Second hidden layer of the NN with 128 neurons

        self.fc3 = nn.Linear(128, 64)   # Third hidden layer of the NN with 64 neurons

        self.fc4 = nn.Linear(64, 7)     # Connect last hidden layer to output layer with 7 neurons (one for each playable column in the game)

        self.relu = nn.ReLU()    # Chooses activation function as ReLU

    def forward(self, x):   # Obligatory function that defines how data flows through the NN. Outputs actual numbers (as a vector) not an object.
                            # If you look at the transformation of x from top to bottom, you can imagine the input flowing through the network.
        
        x = self.flatten(x)   # Calls the flatten function which converts our 2x6x7 state input into a vector of size 2*6*7 = 84 (NN train better on vector inputs)

        # First layer
        x = self.fc1(x)   # Passes the flattened vector through the first hidden layer
        x = self.relu(x)  # Applies ReLU activation function to the output of the first hidden layer

        # Second layer
        x = self.fc2(x)
        x = self.relu(x)

        # Third layer
        x = self.fc3(x)
        x = self.relu(x)

        # Output vector
        x = self.fc4(x)

        return x
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # We choose the size of the deque buffer for the NN

    # NOTE: MIGHT NEED TO ADD REWARD, NEXT STATE HERE LATER
    def push(self, state):
        self.buffer.append(state)  # Saves the state to the buffer. When buffer is full, the left-most (oldest) state is removed.
        
    def sample(self, batch_size):   # To sample a batch of items/states from replay buffer
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)
        
class DQNAgent:
    def __init__(self, input_shape, move_count, lr=0.001): # lr = learning rate
        self.input_shape = input_shape
        self.move_count = move_count
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") # For GPU/CPU management if available

        # Main network (policy network), ie. the one that will do the training
        self.policy_net = DQN(input_shape, move_count).to(self.device)  # .to() moves network to GPU if available

        # Target network (intermediate copy of policy network). This is the one that will help compute the Q-values for the policy network during training.
        self.target_net = DQN(input_shape, move_count).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Copies weights of policy network to target network

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr) # Popular NN optimizer (Adam) that adjusts weights using backpropagation

        self.criterion = nn.MSELoss() # Sets loss function to Mean Squared Error (good for Q-value prediction)

        self.buffer = ReplayBuffer(10000) # Buffer of size 10000

        self.epsilon = 1.0          # Initial epsilon rate (start with random moves)
        self.epsilon_min = 0.1      # Minimum epsilon rate for exploration during training
        self.epsilon_decay = 0.995  # Decay rate for epsilon

        self.batch_size = 32 # Batch size for training
        self.gamma = 0.99    # Discount factor for future rewards

    def select_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        else:
            with th.no_grad():  # Stops gradient calculation to save memory and compute during training
                state = th.FloatTensor(state).unsqueeze(0).to(self.device)  # Converts numpy array of state to PyTorch tensor. ".unsqueeze(0)" adds an extra dimension (called 
                                                                            # batch dimension) because NN expects batches of inputs, ie. (2, 6, 7) -> (1, 2, 6, 7)
                q_values = self.policy_net(state) # Passes state through policy network to get Q-values for each possible action (column)
                q_values = q_values.cpu().numpy()[0] # Moves tensor back to CPU and converts to numpy array. "[0]" removes batch dimension.

                # Only consider valid moves
                valid_q_values = {move: q_values[move] for move in valid_moves}
                best_move = max(valid_q_values, key=valid_q_values.get) # Chooses the move with the highest Q-value among valid moves
                return best_move
    
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return  # Not enough samples in buffer to train
        
        batch = self.buffer.sample(self.batch_size)

        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        states = th.FloatTensor(states).to(self.device)
        actions = th.LongTensor(actions).to(self.device)
        rewards = th.FloatTensor(rewards).to(self.device)
        next_states = th.FloatTensor(next_states).to(self.device)
        dones = th.BoolTensor(dones).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute next Q values from target network
        with th.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            # If done, no next Q value
            target_q_values = rewards + ~dones * self.gamma * next_q_values

        loss = self.criterion(current_q_values.squeeze(), target_q_values) # Compute loss between current and target Q values

        self.optimizer.zero_grad() # Zero the gradients before backpropagation
        loss.backward() # Backpropagation
        self.optimizer.step() # Update the weights

        # Gradually decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_experience(self, state, action, reward, next_state, done):
        self.buffer.push((state, action, reward, next_state, done))
    

def train_dqn_agent():
    game = Connect4()
    agent = DQNAgent(input_shape=(2, 6, 7), move_count=7)

    stats = {
        'episodes': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0
    }

    for episode in range(1000): # Train for 1000 games
        state = game.reset()
        total_reward = 0
        steps = 0

        while True:
            # Agent selects action
            valid_moves = game.get_valid_moves()
            action = agent.select_action(state, valid_moves)

            # Execute action
            next_state, reward, done = game.make_move(action)

            # Store experience in replay buffer
            agent.save_experience(state, action, reward, next_state, done)

            # Train the agent
            agent.train_step()

            # Update statistics
            total_reward += reward
            steps += 1
            state = next_state

            if done != 0:
                # Update win/loss statistics
                if reward == 1:
                    stats['wins'] += 1
                elif reward == 0 and game.game_over == -1:
                    stats['draws'] += 1
                else:
                    stats['losses'] += 1
                stats['episodes'] += 1
                break

        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Print stats every 100 episodes
        if episode % 100 == 0:
            win_rate = stats['wins'] / max(1, stats['episodes']) * 100
            print(f"Episode {episode}: Win Rate = {win_rate:.2f}%, Epsilon = {agent.epsilon:.3f}")
    
    print("Training Done.")
    return agent


def test_agent(agent, num_games=10):
    print(f"\nTest for {num_games} games:")
    
    wins = 0
    for game_num in range(num_games):
        game = Connect4()
        state = game.reset()
        
        # Randomly decide who goes first
        if random.random() < 0.5:
            current_player = 'agent'
        else:
            current_player = 'random'
        
        while True:
            if current_player == 'agent':
                # Agent's turn
                valid_moves = game.get_valid_moves()
                action = agent.select_action(state, valid_moves)
                # Force exploitation (no exploration during test)
                agent.epsilon = 0
            else:
                # Random opponent's turn
                valid_moves = game.get_valid_moves()
                action = random.choice(valid_moves)
            
            state, reward, done = game.make_move(action)
            
            if done:
                if reward == 1 and current_player == 'agent':
                    wins += 1
                break
            
            # Switch players
            current_player = 'random' if current_player == 'agent' else 'agent'
    
    win_rate = wins / num_games * 100
    print(f"Agent win rate: {win_rate:.1f}")
    return win_rate

if __name__ == "__main__":
    trained_agent = train_dqn_agent()
    test_agent(trained_agent)

