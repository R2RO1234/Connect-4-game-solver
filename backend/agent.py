from MonteCarloSearchTree import MCTS
from CNN_architecture import CNN
import torch
from logic import Connect4
import numpy as np


class AIPlayer:
    def __init__(self,filename):
        trained_network = CNN(num_res_blocks=5, channels=128)
        trained_network.load_state_dict(torch.load(filename, map_location='cpu'))
        trained_network.eval()
        self.mc = MCTS(trained_network, num_simulations=400)
        self.game = Connect4()
        

      
    
    
    def choose_move(self, board ,current_player ):
        self.game.board = board
        move = np.argmax(self.mc.search(self.game, temp=0))
        return int(move)